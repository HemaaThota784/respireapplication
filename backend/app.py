"""Respira backend (FastAPI).   Run:  uvicorn app:app --port 8000"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from starlette.middleware.sessions import SessionMiddleware

import config
import db
from assessment_routes import router as assessment_router
from auth_routes import get_current_user, router as auth_router
from insights_routes import router as insights_router

log = logging.getLogger("respira")
logging.basicConfig(level=logging.INFO)

MODELS = Path(os.getenv("MODELS_DIR", Path(__file__).parent / "models"))


def _first(*names):
    """First existing file in models/, else the first name (for a clear error message)."""
    for n in names:
        if (MODELS / n).exists():
            return MODELS / n
    for pattern in ("respire_logits*.keras", "respireai*.keras"):   # any other seed, e.g. _main_s43
        found = sorted(MODELS.glob(pattern))
        if found and names[0].endswith(".keras"):
            return found[0]
    return MODELS / names[0]


AUDIO_MODEL = Path(os.getenv("AUDIO_MODEL", _first("respire_logits.keras", "respireai_main_s77.keras")))
NORMALIZER = Path(os.getenv("AUDIO_NORMALIZER", _first("norm_stats.json", "normalizer.json")))
PREPROC_CFG = MODELS / "preproc_config.json"
META = MODELS / "meta.json"
XRAY_MODEL = Path(os.getenv("XRAY_MODEL", MODELS / "xray_pneumonia_model.keras"))
_T = os.getenv("AUDIO_TEMPERATURE")
TEMPERATURE = float(_T) if _T else None   # None -> read from meta.json
MAX_BYTES = 25 * 1024 * 1024

state = {"audio": None, "xray": None, "audio_error": None, "xray_error": None, "db_error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        db.init_db()
        log.info("MongoDB connected")
    except Exception as e:                      # keep the server up; auth routes return 503
        state["db_error"] = f"{type(e).__name__}: {e}"
        log.exception("MongoDB connection failed")
    try:
        from audio_service import AudioPredictor
        state["audio"] = AudioPredictor(AUDIO_MODEL, NORMALIZER, PREPROC_CFG, META, TEMPERATURE)
        log.info("Audio model loaded")
    except Exception as e:
        state["audio_error"] = f"{type(e).__name__}: {e}"
        log.exception("Audio model failed to load")
    try:
        from xray_service import XrayPredictor
        state["xray"] = XrayPredictor(XRAY_MODEL)
        log.info("X-ray model loaded")
    except Exception as e:
        state["xray_error"] = f"{type(e).__name__}: {e}"
        log.exception("X-ray model failed to load")
    yield


app = FastAPI(title="Respira API", lifespan=lifespan)

# Temporary cookie used only during the Google OAuth redirect (state check).
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET, same_site="lax",
                   https_only=config.COOKIE_SECURE, max_age=600)
app.add_middleware(CORSMiddleware, allow_origins=[config.FRONTEND_URL, "http://127.0.0.1:5173"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router)
app.include_router(assessment_router)
app.include_router(insights_router)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(_, exc: PyMongoError):
    log.error("MongoDB error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Database unavailable. Please try again later."})


@app.get("/api/health")
def health():
    return {"audio_ready": state["audio"] is not None, "xray_ready": state["xray"] is not None,
            "db_ready": state["db_error"] is None,
            "gemini_configured": bool(config.GEMINI_API_KEY),
            "audio_error": state["audio_error"], "xray_error": state["xray_error"],
            "db_error": state["db_error"]}


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 25 MB.")
    return data


@app.post("/api/predict/audio")
async def predict_audio(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if state["audio"] is None:
        raise HTTPException(503, f"Audio model unavailable: {state['audio_error']}")
    data = await _read(file)
    try:
        return state["audio"].predict(data)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Audio prediction failed")
        raise HTTPException(400, f"Could not process audio (upload a valid .wav file): {e}")


@app.post("/api/predict/xray")
async def predict_xray(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if state["xray"] is None:
        raise HTTPException(503, f"X-ray model unavailable: {state['xray_error']}")
    data = await _read(file)
    try:
        return state["xray"].predict(data)
    except Exception as e:
        log.exception("X-ray prediction failed")
        raise HTTPException(400, f"Could not process image: {e}")