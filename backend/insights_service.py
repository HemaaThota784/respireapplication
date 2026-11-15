"""

Gemini-powered, plain-language explanations of the model predictions.

* Uses ONLY the predictions the frontend already received from /api/predict/*.

* Never changes the predictions: the service returns an explanation, nothing else.

* Gemini's general medical knowledge is used for the explanation (there is no

  medical knowledge base / RAG in this project), so the prompt forces cautious,

  educational wording.

* The result page keeps working if Gemini fails: errors are mapped to a small

  set of exceptions that the route turns into friendly HTTP errors.

"""

from __future__ import annotations

import hashlib

import json

import logging

import re

import threading

import time

from collections import OrderedDict, deque

from typing import Deque, Dict, List, Literal, Optional

import httpx

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

import config

log = logging.getLogger("respira")

# ======================================================================

# 1. Supported classes (small, maintainable mapping)

#    These must match what audio_service.py / xray_service.py return.

# ======================================================================

AUDIO_CLASSES: Dict[str, Dict[str, str]] = {

    "normal": {"label": "Normal",

               "description": "No crackles or wheezes were detected in the recording."},

    "crackle": {"label": "Crackles",

                "description": "Crackles are short, discontinuous popping or crackling lung sounds, "

                               "usually heard when breathing in."},

    "wheeze": {"label": "Wheezes",

               "description": "Wheezes are continuous, high-pitched whistling lung sounds, usually "

                              "linked to narrowed airways."},

    "both": {"label": "Crackles and wheezes",

             "description": "Both crackle-type and wheeze-type sounds were detected."},

}

XRAY_CLASSES: Dict[str, Dict[str, str]] = {

    "Normal": {"label": "No pneumonia pattern",

               "description": "The X-ray model did not predict a pneumonia pattern."},

    "Bacterial": {"label": "Bacterial pneumonia",

                  "description": "The X-ray model predicted the category 'bacterial pneumonia'."},

    "Viral": {"label": "Viral pneumonia",

              "description": "The X-ray model predicted the category 'viral pneumonia'."},

}

AUDIO_PROB_KEYS = set(AUDIO_CLASSES)

XRAY_PROB_KEYS = {"Normal", "Bacterial Pneumonia", "Viral Pneumonia"}   # keys used by xray_service.py

SOURCE_NAME = {"audio": "chest sound recording", "xray": "chest X-ray"}

# ======================================================================

# 2. Request validation (what the frontend sends)

# ======================================================================

def _check_probs(v: Dict[str, float], allowed: set) -> Dict[str, float]:

    for k, p in v.items():

        if k not in allowed:

            raise ValueError(f"Unexpected class '{k}' in probabilities.")

        if not 0 <= p <= 1.0001:

            raise ValueError("Probabilities must be between 0 and 1.")

    return v

class AudioPrediction(BaseModel):

    predicted_class: Literal["normal", "crackle", "wheeze", "both"]

    confidence: float = Field(ge=0, le=100)

    probabilities: Dict[str, float] = Field(default_factory=dict, max_length=8)

    @field_validator("probabilities")

    @classmethod

    def _probs(cls, v):

        return _check_probs(v, AUDIO_PROB_KEYS)

class XrayPrediction(BaseModel):

    predicted_class: Literal["Normal", "Bacterial", "Viral"]

    confidence: float = Field(ge=0, le=100)

    probabilities: Dict[str, float] = Field(default_factory=dict, max_length=8)

    @field_validator("probabilities")

    @classmethod

    def _probs(cls, v):

        return _check_probs(v, XRAY_PROB_KEYS)

class InsightsRequest(BaseModel):

    audio: Optional[AudioPrediction] = None

    xray: Optional[XrayPrediction] = None

    @model_validator(mode="after")

    def _need_one(self):

        if self.audio is None and self.xray is None:

            raise ValueError("Provide an audio and/or X-ray prediction.")

        return self

# ======================================================================

# 3. Schemas: what Gemini must return, and what the frontend receives

# ======================================================================

class GFinding(BaseModel):

    source: Literal["audio", "xray"]

    meaning: str = Field(description="2-3 simple sentences: what this predicted category means "

                                     "and what it does NOT mean.")

class GAssociation(BaseModel):

    source: Literal["audio", "xray"]

    condition: str = Field(description="Name of a condition that may be associated with the finding.")

    why: str = Field(description="1-2 simple sentences on why this condition may produce or relate to "

                                 "the finding. Use cautious wording such as 'can', 'may', 'is sometimes'.")

class GeminiInsights(BaseModel):

    summary: str = Field(description="2-3 sentence plain-English overview of the available results.")

    findings: List[GFinding] = Field(description="Exactly one entry per provided input.")

    possible_associations: List[GAssociation] = Field(

        description="Possible associations only (never diagnoses). 2-5 per abnormal finding; "

                    "empty for a normal finding.")

    combined_interpretation: str = Field(

        description="Only when BOTH inputs are provided: how the two predictions relate "

                    "(agree or differ). Otherwise an empty string.")

    recommended_next_steps: List[str] = Field(description="3-5 general, non-medication next steps.")

    limitations: List[str] = Field(description="3-4 short limitations of these predictions.")

class FindingOut(BaseModel):

    source: Literal["audio", "xray"]

    detected: str            # canonical label from OUR mapping, not from Gemini

    confidence: float        # copied from the request, never from Gemini

    meaning: str

class AssociationOut(BaseModel):

    source: Literal["audio", "xray"]

    condition: str

    why: str

class InsightsResponse(BaseModel):

    summary: str

    findings: List[FindingOut]

    possible_associations: List[AssociationOut]

    combined_interpretation: Optional[str] = None

    agreement: Optional[Literal["both_normal", "both_abnormal", "differ"]] = None

    recommended_next_steps: List[str]

    limitations: List[str]

    model: str

    cached: bool = False

# ======================================================================

# 4. Errors (the route turns these into HTTP responses)

# ======================================================================

class InsightsError(Exception):

    status_code = 502

    public_message = "AI insights are temporarily unavailable."

    def __init__(self, detail: str = ""):

        super().__init__(detail)

class NotConfigured(InsightsError):

    status_code = 503

    public_message = "AI insights are not configured on this server."

class RateLimited(InsightsError):

    status_code = 429

    public_message = "Too many insight requests. Please try again later."

class UpstreamBusy(InsightsError):

    status_code = 503

    public_message = "The AI service is busy right now. Please try again in a moment."

class UpstreamTimeout(InsightsError):

    status_code = 504

    public_message = "The AI service took too long to respond. Please try again."

class InvalidAIResponse(InsightsError):

    status_code = 502

    public_message = "The AI returned an unusable response. Please try again."

# ======================================================================

# 5. Prompt

# ======================================================================

SYSTEM_PROMPT = """You are a careful medical-education assistant inside a respiratory screening app.

Two machine-learning models may have produced predictions: a lung-sound classifier (normal, crackles,

wheezes, both) and a chest X-ray classifier (no pneumonia pattern, bacterial pneumonia, viral pneumonia).

Your job is to explain the predictions that are provided, in simple English (about a 12-year-old reading level).

STRICT RULES

1. Use ONLY the predictions given in the user message. Never invent or assume a prediction for an input

   marked "not provided". Never change a prediction or its confidence.

2. These are screening-model outputs, NOT diagnoses. Present every condition as a possible association

   only ("can be associated with", "may", "is sometimes seen in"). Never say the person has a disease.

3. Explain mechanisms cautiously and briefly, in simple language.

4. For X-ray predictions, explain only the predicted category in general terms. Do NOT describe specific

   radiological findings (e.g. opacities, consolidation, effusions) as if you saw them in this image, and

   do not claim the model confirmed pneumonia. You may say that other conditions can look similar and that

   only a clinician reviewing the real X-ray and symptoms can tell them apart.

5. Model confidence is the model's score for its predicted class. It is NOT the probability that the person

   has a disease. Say this plainly if you mention confidence. If confidence is below about 65%, or the top

   two class scores are close, say the model was less certain.

6. If both inputs are provided, explain each separately and then how they relate. A normal result from one

   model does not rule out a problem found by the other. Different results are possible and are not

   necessarily an error. If only one input is provided, leave combined_interpretation as an empty string.

7. NEVER recommend starting, stopping, changing or choosing any medication, dose or treatment.

8. Next steps must be general: see a qualified healthcare professional, share the original recording/X-ray

   and symptoms, mention symptoms to a clinician, seek urgent care if breathing becomes difficult.

9. If information is insufficient for a point, say so instead of guessing.

10. For a normal finding, return no possible associations for it, and explain that a normal model result

    does not rule out every respiratory condition.

11. Plain text only: no markdown, no HTML, no links, no emojis. Keep every string concise.

"""

def _scores(probs: Dict[str, float], names: Dict[str, str]) -> str:

    items = sorted(probs.items(), key=lambda kv: -kv[1])

    return ", ".join(f"{names.get(k, k)}: {v * 100:.1f}%" for k, v in items) or "not available"

def build_prompt(req: InsightsRequest) -> str:

    """Built only from validated enums and numbers; no free text from the client reaches Gemini."""

    lines = ["Predictions available for this request (use only these):", ""]

    if req.audio:

        a = req.audio

        info = AUDIO_CLASSES[a.predicted_class]

        lines += ["INPUT 'audio' - chest sound recording classifier",

                  f"- Predicted class: {info['label']}",

                  f"- What the class means: {info['description']}",

                  f"- Model confidence for this class: {a.confidence:.1f}%",

                  f"- All class scores: {_scores(a.probabilities, {k: v['label'] for k, v in AUDIO_CLASSES.items()})}",

                  ""]

    else:

        lines += ["INPUT 'audio': not provided", ""]

    if req.xray:

        x = req.xray

        info = XRAY_CLASSES[x.predicted_class]

        lines += ["INPUT 'xray' - chest X-ray classifier",

                  f"- Predicted class: {info['label']}",

                  f"- What the class means: {info['description']}",

                  f"- Model confidence for this class: {x.confidence:.1f}%",

                  f"- All class scores: {_scores(x.probabilities, {})}",

                  ""]

    else:

        lines += ["INPUT 'xray': not provided", ""]

    provided = [k for k, v in (("audio", req.audio), ("xray", req.xray)) if v]

    lines.append(f"Provided inputs: {', '.join(provided)}. Return JSON that follows the schema exactly.")

    return "\n".join(lines)

# ======================================================================

# 6. Response validation / cleaning

# ======================================================================

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _clean(text: str, max_len: int) -> str:

    text = _CTRL.sub("", text or "")

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_len:

        return text

    cut = text[:max_len]

    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))

    return cut[: end + 1] if end > max_len * 0.5 else cut.rstrip() + "…"

def _is_abnormal(req: InsightsRequest, source: str) -> bool:

    return (req.audio.predicted_class != "normal") if source == "audio" else (req.xray.predicted_class != "Normal")

def validate_and_clean(g: GeminiInsights, req: InsightsRequest, model: str) -> InsightsResponse:

    provided = [s for s, v in (("audio", req.audio), ("xray", req.xray)) if v]

    summary = _clean(g.summary, 700)

    if not summary:

        raise InvalidAIResponse("empty summary")

    findings: List[FindingOut] = []

    for src in provided:                                   # exactly one per provided input, fixed order

        match = next((f for f in g.findings if f.source == src), None)

        meaning = _clean(match.meaning, 600) if match else ""

        if not meaning:

            raise InvalidAIResponse(f"missing finding for {src}")

        if src == "audio":

            detected, conf = AUDIO_CLASSES[req.audio.predicted_class]["label"], req.audio.confidence

        else:

            detected, conf = XRAY_CLASSES[req.xray.predicted_class]["label"], req.xray.confidence

        findings.append(FindingOut(source=src, detected=detected, confidence=conf, meaning=meaning))

    assoc: List[AssociationOut] = []

    seen = set()

    for a in g.possible_associations:

        if a.source not in provided:                       # drop anything about a non-provided input

            continue

        cond, why = _clean(a.condition, 80), _clean(a.why, 400)

        key = (a.source, cond.lower())

        if not cond or not why or key in seen:

            continue

        if not _is_abnormal(req, a.source):                # normal finding -> no associations

            continue

        seen.add(key)

        assoc.append(AssociationOut(source=a.source, condition=cond, why=why))

    assoc = assoc[:8]

    combined: Optional[str] = None

    agreement = None

    if len(provided) == 2:

        combined = _clean(g.combined_interpretation, 800)

        if not combined:

            raise InvalidAIResponse("missing combined interpretation")

        a_ab, x_ab = _is_abnormal(req, "audio"), _is_abnormal(req, "xray")

        agreement = "both_abnormal" if a_ab and x_ab else "both_normal" if not a_ab and not x_ab else "differ"

    steps = [s for s in (_clean(s, 300) for s in g.recommended_next_steps) if s][:6]

    limits = [s for s in (_clean(s, 300) for s in g.limitations) if s][:6]

    if not steps or not limits:

        raise InvalidAIResponse("missing next steps or limitations")

    return InsightsResponse(summary=summary, findings=findings, possible_associations=assoc,

                            combined_interpretation=combined, agreement=agreement,

                            recommended_next_steps=steps, limitations=limits, model=model)

# ======================================================================

# 7. Cache + per-user rate limit (avoid repeated paid calls)

# ======================================================================

_CACHE_MAX = 500

_CACHE_TTL = 24 * 3600

_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()

_hits: Dict[str, Deque[float]] = {}

_lock = threading.Lock()

_working_model: Dict[str, Optional[str]] = {"name": None}

def _cache_key(req: InsightsRequest) -> str:

    def part(p, probs_round=3):

        return None if p is None else {"c": p.predicted_class, "conf": round(p.confidence, 1),

                                       "p": {k: round(v, probs_round) for k, v in sorted(p.probabilities.items())}}

    blob = json.dumps({"a": part(req.audio), "x": part(req.xray), "m": config.GEMINI_MODEL}, sort_keys=True)

    return hashlib.sha256(blob.encode()).hexdigest()

def _cache_get(key: str) -> Optional[dict]:

    with _lock:

        item = _cache.get(key)

        if not item:

            return None

        ts, data = item

        if time.time() - ts > _CACHE_TTL:

            _cache.pop(key, None)

            return None

        _cache.move_to_end(key)

        return data

def _cache_put(key: str, data: dict) -> None:

    with _lock:

        _cache[key] = (time.time(), data)

        _cache.move_to_end(key)

        while len(_cache) > _CACHE_MAX:

            _cache.popitem(last=False)

def _check_rate_limit(user_id: str) -> None:

    now = time.time()

    with _lock:

        q = _hits.setdefault(user_id, deque())

        while q and now - q[0] > 3600:

            q.popleft()

        if len(q) >= config.INSIGHTS_RATE_LIMIT_PER_HOUR:

            raise RateLimited(user_id)

        q.append(now)

# ======================================================================

# 8. Gemini call

# ======================================================================

def _make_client():

    """Separate function so tests can point the real SDK at a fake server."""

    from google import genai

    from google.genai import types

    return genai.Client(api_key=config.GEMINI_API_KEY,

                        http_options=types.HttpOptions(timeout=config.GEMINI_TIMEOUT_SECONDS * 1000))

def _generate(client, model: str, prompt: str) -> str:

    from google.genai import types

    resp = client.models.generate_content(

        model=model,

        contents=prompt,

        config=types.GenerateContentConfig(

            system_instruction=SYSTEM_PROMPT,

            temperature=0.3,

            max_output_tokens=2048,          # generous: newer models spend tokens on thinking

            response_mime_type="application/json",

            response_json_schema=GeminiInsights.model_json_schema(),

        ),

    )

    text = resp.text           # None when blocked or cut off (e.g. finish_reason MAX_TOKENS / SAFETY)

    if not text:

        reason = None

        try:

            reason = resp.candidates[0].finish_reason if resp.candidates else None

        except Exception:

            pass

        raise InvalidAIResponse(f"empty response (finish_reason={reason})")

    return text

def _call_gemini(prompt: str) -> tuple[str, str]:
    """Return (JSON text, model used), trying the fallback after transient failures."""
    from google.genai import errors

    if not config.GEMINI_API_KEY:
        raise NotConfigured("GEMINI_API_KEY is empty")

    client = _make_client()
    candidates: List[str] = []
    for model_name in (
        _working_model["name"],
        config.GEMINI_MODEL,
        config.GEMINI_FALLBACK_MODEL,
    ):
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    last_error: Optional[Exception] = None
    saw_timeout = False

    for model_name in candidates:
        # Two attempts per model, with a short backoff before retrying.
        for attempt in (1, 2):
            try:
                response_text = _generate(client, model_name, prompt)
                _working_model["name"] = model_name
                return response_text, model_name

            except errors.ServerError as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                log.warning(
                    "Gemini server error %s on %s (attempt %d/2)",
                    code if code is not None else "?",
                    model_name,
                    attempt,
                )
                if attempt == 1:
                    time.sleep(2.0)
                    continue
                # After two failures, move on to the next configured model.
                break

            except errors.ClientError as exc:
                code = getattr(exc, "code", None)
                last_error = exc
                if code == 404:
                    log.warning(
                        "Gemini model '%s' was not found for this API key; trying next candidate.",
                        model_name,
                    )
                    break
                if code == 429:
                    raise UpstreamBusy(str(exc)) from exc
                if code in (400, 401, 403):
                    log.error(
                        "Gemini rejected the request (%s). Check the API key and model access.",
                        code,
                    )
                raise InsightsError(str(exc)) from exc

            except (httpx.TimeoutException, TimeoutError) as exc:
                last_error = exc
                saw_timeout = True
                log.warning(
                    "Gemini request timed out on %s (attempt %d/2)",
                    model_name,
                    attempt,
                )
                if attempt == 1:
                    time.sleep(2.0)
                    continue
                break

            except InvalidAIResponse:
                # An empty/blocked response is not fixed by trying another model.
                raise

            except InsightsError:
                raise

            except Exception as exc:
                last_error = exc
                log.exception("Unexpected Gemini failure on %s", model_name)
                # Allow the configured fallback to handle transient/network failures.
                break

    if saw_timeout or getattr(last_error, "code", None) == 504:
        raise UpstreamTimeout(str(last_error)) from last_error
    if last_error is not None:
        if getattr(last_error, "code", None) in (429, 503):
            raise UpstreamBusy(str(last_error)) from last_error
        raise InsightsError(str(last_error)) from last_error
    raise InsightsError("No usable Gemini model is configured.")


def get_insights(req: InsightsRequest, user_id: str) -> InsightsResponse:

    key = _cache_key(req)

    hit = _cache_get(key)

    if hit is not None:

        return InsightsResponse(**{**hit, "cached": True})

    if not config.GEMINI_API_KEY:

        raise NotConfigured("GEMINI_API_KEY is empty")

    _check_rate_limit(user_id)

    prompt = build_prompt(req)

    text, model = _call_gemini(prompt)

    try:

        parsed = GeminiInsights.model_validate_json(text)

    except (ValidationError, ValueError) as e:

        log.warning("Gemini returned invalid JSON/schema: %s", str(e)[:300])

        raise InvalidAIResponse("schema validation failed")

    result = validate_and_clean(parsed, req, model)

    _cache_put(key, result.model_dump())

    return result