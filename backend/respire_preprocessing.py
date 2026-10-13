"""
respire_preprocessing.py
========================
Single source of truth for lung-sound preprocessing.

Import THIS module from both the training notebook and the Streamlit app, so
there is no training/serving skew (the most common silent failure in deployed
audio models).

Pipeline (applied to the WHOLE recording first, then cut into cycles/windows):
    load -> mono -> DC removal -> anti-aliased resample -> zero-phase band-pass
    -> (optional) edge-silence trim -> RMS normalisation
    -> cycle / window extraction (repeat-padding with edge fades, no zero-padding)
    -> log-mel spectrogram -> dataset-level per-band standardisation (Normalizer)

Dependencies: numpy, scipy, soundfile  (no TensorFlow / librosa needed, so the
Streamlit app stays light).
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal as sps
import soundfile as sf

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PreprocConfig:
    # signal
    sr: int = 8000                 # lung-sound energy lives < 2 kHz; 8 kHz keeps headroom
    band_low: float = 50.0         # removes handling noise / DC drift
    band_high: float = 2000.0      # upper limit of clinically relevant lung-sound content
    filter_order: int = 4
    # segmentation
    segment_s: float = 5.0         # fixed model input length
    window_hop_s: float = 2.5      # hop for sliding windows (inference / long cycles)
    min_cycle_s: float = 0.1       # drop degenerate annotations shorter than this
    # time-frequency
    n_fft: int = 512               # 64 ms @ 8 kHz
    hop_length: int = 64           # 8 ms -> resolves crackle-scale transients (~5-20 ms)
    n_mels: int = 64
    mel_fmin: float = 0.0
    mel_fmax: float = 2000.0
    log_eps: float = 1e-6

    @property
    def segment_samples(self) -> int:
        return int(round(self.segment_s * self.sr))

    @property
    def n_frames(self) -> int:
        return 1 + self.segment_samples // self.hop_length

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.n_mels, self.n_frames, 1)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "PreprocConfig":
        with open(path) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------------------- #
# Audio I/O + signal conditioning
# --------------------------------------------------------------------------- #
AudioSource = Union[str, bytes, bytearray, io.IOBase]


def load_audio(src: AudioSource) -> Tuple[np.ndarray, int]:
    """Read WAV/FLAC/OGG (8/16/24/32-bit PCM or float) -> mono float32 in [-1, 1].

    `src` may be a path, raw bytes, or a file-like object (e.g. Streamlit's
    UploadedFile). soundfile scales every PCM width correctly (the old
    hand-rolled reader mis-scaled 8-bit and 32-bit audio).
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    if hasattr(src, "seek"):
        src.seek(0)
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    return x.mean(axis=1).astype(np.float32), int(sr)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling WITH anti-alias filtering (np.interp has none)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    return sps.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


@lru_cache(maxsize=8)
def _band_sos(sr: int, lo: float, hi: float, order: int) -> np.ndarray:
    hi = min(hi, 0.98 * sr / 2)
    return sps.butter(order, [lo, hi], btype="band", fs=sr, output="sos")


def bandpass(x: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    sos = _band_sos(cfg.sr, cfg.band_low, cfg.band_high, cfg.filter_order)
    try:
        y = sps.sosfiltfilt(sos, x)  # zero-phase: does not smear crackle onsets
    except ValueError:               # signal shorter than the filter pad length
        y = sps.sosfilt(sos, x)
    return y.astype(np.float32)


def rms_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Unit-RMS scaling per RECORDING (keeps relative loudness between cycles)."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return (x / rms).astype(np.float32) if rms > eps else np.zeros_like(x)


def trim_silence_edges(x: np.ndarray, sr: int, top_db: float = -45.0,
                       frame_s: float = 0.05) -> Tuple[np.ndarray, int]:
    """Trim only leading/trailing near-silence (never interior pauses).

    Returns (trimmed, start_sample). Use for uploaded audio only; annotated
    training recordings must NOT be trimmed (annotation times would shift).
    """
    frame = max(1, int(sr * frame_s))
    n = len(x) // frame
    if n < 3:
        return x, 0
    e = (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12
    db = 10 * np.log10(e / np.percentile(e, 95))
    keep = np.where(db > top_db)[0]
    if len(keep) == 0:
        return x, 0
    s, t = keep[0] * frame, (keep[-1] + 1) * frame
    return x[s:t], int(s)


def preprocess_recording(x: np.ndarray, sr_in: int, cfg: PreprocConfig) -> np.ndarray:
    """Whole-recording conditioning. Filtering BEFORE cutting avoids edge
    transients on short cycles."""
    x = x - np.mean(x)
    x = resample(x, sr_in, cfg.sr)
    x = bandpass(x, cfg)
    return rms_normalize(x)


def assess_quality(x: np.ndarray, sr: int) -> List[str]:
    """Cheap sanity checks to show as warnings in the UI (raw, pre-conditioning)."""
    w: List[str] = []
    dur = len(x) / sr
    if dur < 2.0:
        w.append(f"Recording is very short ({dur:.1f}s); at least one full breath (~3-5 s) is recommended.")
    if np.mean(np.abs(x) > 0.99) > 0.01:
        w.append("Audio is clipped (>1% of samples at full scale); predictions may be unreliable.")
    if np.sqrt(np.mean(x ** 2)) < 1e-4:
        w.append("Signal is almost silent.")
    if sr < 4000:
        w.append(f"Low sampling rate ({sr} Hz) limits usable bandwidth.")
    return w


# --------------------------------------------------------------------------- #
# ICBHI annotations
# --------------------------------------------------------------------------- #
def parse_recording_name(name: str) -> Dict[str, str]:
    """'101_1b1_Al_sc_Meditron' -> patient / rec idx / chest location / mode / device."""
    p = name.split("_")
    return dict(file=name, patient=p[0], rec_idx=p[1], chest_loc=p[2], acq_mode=p[3],
                device=p[4] if len(p) > 4 else "unknown")


def read_annotations(txt_path: str) -> np.ndarray:
    """Returns array [n_cycles, 4]: start, end, crackle, wheeze."""
    a = np.loadtxt(txt_path, dtype=np.float64, ndmin=2)
    return a[:, :4] if a.size else np.zeros((0, 4))


def cycle_label(crackle: int, wheeze: int) -> int:
    return int(crackle) + 2 * int(wheeze)  # 0 normal, 1 crackle, 2 wheeze, 3 both


def extract_cycles(x_proc: np.ndarray, ann: np.ndarray, cfg: PreprocConfig) -> List[dict]:
    out = []
    for s, e, c, w in ann:
        if e - s < cfg.min_cycle_s:
            continue
        seg = x_proc[int(s * cfg.sr): int(e * cfg.sr)]
        if len(seg) < int(cfg.min_cycle_s * cfg.sr):
            continue
        out.append(dict(wave=seg.astype(np.float32), start=float(s), end=float(e),
                        label=cycle_label(c, w)))
    return out


# --------------------------------------------------------------------------- #
# Fixed-length handling (NO zero padding: zeros become a constant floor in
# log-mel and distort any per-sample scaling)
# --------------------------------------------------------------------------- #
def _edge_fade(x: np.ndarray, n: int) -> np.ndarray:
    if n < 2 or len(x) < 2 * n:
        return x
    y = x.copy()
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def fix_length(x: np.ndarray, n: int, sr: int,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Crop (random if rng else centre) or cyclically repeat to exactly n samples."""
    L = len(x)
    if L == n:
        return x
    if L > n:
        s = int(rng.integers(0, L - n + 1)) if rng is not None else (L - n) // 2
        return x[s: s + n]
    x = _edge_fade(x, max(2, int(0.003 * sr)))
    reps = int(math.ceil((n + L) / L))
    tiled = np.tile(x, reps)
    off = int(rng.integers(0, L)) if rng is not None else 0
    return tiled[off: off + n]


def cycle_windows(x: np.ndarray, n: int, hop: int, sr: int) -> np.ndarray:
    """Deterministic windows covering a cycle/recording. Shape [k, n]."""
    if len(x) <= n:
        return fix_length(x, n, sr)[None, :]
    starts = list(range(0, len(x) - n + 1, hop))
    if starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    return np.stack([x[s: s + n] for s in starts])


# --------------------------------------------------------------------------- #
# Log-mel spectrogram (pure numpy)
# --------------------------------------------------------------------------- #
def _hz2mel(f):  # HTK
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel2hz(np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - l) / max(c - l, 1e-9)
        down = (r - freqs) / max(r - c, 1e-9)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


@lru_cache(maxsize=4)
def _hann(n: int) -> np.ndarray:
    return sps.get_window("hann", n, fftbins=True).astype(np.float32)


def log_mel(x: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """[n_samples] -> [n_mels, n_frames] natural-log mel power (float32)."""
    pad = cfg.n_fft // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(xp, cfg.n_fft)[:: cfg.hop_length]
    spec = np.abs(np.fft.rfft(frames * _hann(cfg.n_fft), axis=1)) ** 2   # [T, F]
    fb = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.mel_fmin, cfg.mel_fmax)
    mel = fb @ spec.T.astype(np.float32)                                  # [M, T]
    return np.log(mel + cfg.log_eps).astype(np.float32)


class Normalizer:
    """Dataset-level per-mel-band standardisation. Fit on TRAIN only; reuse in app.

    (Per-sample min-max, as in the original notebook, discards absolute level
    and lets zero-padding dictate the scale.)
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32).reshape(-1, 1)
        self.std = np.maximum(std.astype(np.float32).reshape(-1, 1), 1e-6)

    @classmethod
    def fit(cls, mels: Sequence[np.ndarray]) -> "Normalizer":
        s = s2 = 0.0
        n = 0
        for m in mels:
            s = s + m.sum(axis=1)
            s2 = s2 + (m.astype(np.float64) ** 2).sum(axis=1)
            n += m.shape[1]
        mean = s / n
        return cls(mean, np.sqrt(np.maximum(s2 / n - mean ** 2, 0)))

    def __call__(self, m: np.ndarray) -> np.ndarray:
        return ((m - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dict(mean=self.mean.ravel().tolist(), std=self.std.ravel().tolist()), f)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["mean"]), np.array(d["std"]))


# --------------------------------------------------------------------------- #
# Augmentation (training only)
# --------------------------------------------------------------------------- #
def augment_waveform(x: np.ndarray, cfg: PreprocConfig, rng: np.random.Generator) -> np.ndarray:
    """Speed perturbation (+-10 %, also shifts pitch), gain, band-limited noise."""
    if rng.random() < 0.5:
        f = float(rng.uniform(0.9, 1.1))
        x = sps.resample_poly(x, 100, int(round(100 * f))).astype(np.float32)
    x = x * float(10 ** (rng.uniform(-3, 3) / 20))
    if rng.random() < 0.5:
        snr_db = float(rng.uniform(15, 35))
        noise = sps.sosfilt(_band_sos(cfg.sr, cfg.band_low, cfg.band_high, cfg.filter_order),
                            rng.standard_normal(len(x))).astype(np.float32)
        p_sig = np.mean(x ** 2) + 1e-12
        noise *= math.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
        x = x + noise
    return x.astype(np.float32)


def channel_augment(m: np.ndarray, rng: np.random.Generator, max_ln: float = 0.5) -> np.ndarray:
    """Random smooth spectral tilt in the log-mel domain: simulates different
    stethoscope / microphone frequency responses (ICBHI uses 4 devices)."""
    knots = rng.uniform(-max_ln, max_ln, size=5)
    curve = np.interp(np.linspace(0, 4, m.shape[0]), np.arange(5), knots)
    return (m + curve[:, None]).astype(np.float32)


def spec_augment(m: np.ndarray, rng: np.random.Generator, n_freq: int = 2, max_f: int = 8,
                 n_time: int = 2, max_t: int = 40) -> np.ndarray:
    """SpecAugment on a STANDARDISED spectrogram (mask value 0 == dataset mean).
    Applied identically to every class (the old code applied it to 'none' only)."""
    m = m.copy()
    M, T = m.shape
    for _ in range(n_freq):
        f = int(rng.integers(0, max_f + 1))
        if f:
            f0 = int(rng.integers(0, M - f + 1)); m[f0:f0 + f] = 0
    for _ in range(n_time):
        t = int(rng.integers(0, max_t + 1))
        if t:
            t0 = int(rng.integers(0, T - t + 1)); m[:, t0:t0 + t] = 0
    return m


def augment_waveform_light(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gain only (+-2 dB). Keeps pitch, speed and device/noise characteristics intact."""
    return (x * float(10 ** (rng.uniform(-2, 2) / 20))).astype(np.float32)


def make_model_input(wave: np.ndarray, cfg: PreprocConfig, norm: Normalizer,
                     rng: Optional[np.random.Generator] = None, augment=False) -> np.ndarray:
    """One cycle/window waveform -> model tensor [n_mels, n_frames, 1].

    augment: False/None -> deterministic evaluation path
             "light"     -> gain + random crop/offset + mild SpecAugment (best for accuracy)
             True/"full" -> speed, gain, noise, spectral tilt, SpecAugment (strong regulariser)
    """
    mode = "none" if (not augment or rng is None) else ("full" if augment is True else str(augment))
    if mode == "full":
        wave = augment_waveform(wave, cfg, rng)
    elif mode == "light":
        wave = augment_waveform_light(wave, rng)
    x = fix_length(wave, cfg.segment_samples, cfg.sr, rng if mode != "none" else None)
    m = log_mel(x, cfg)
    if mode == "full":
        m = channel_augment(m, rng)
    m = norm(m)
    if mode == "full" and rng.random() < 0.8:
        m = spec_augment(m, rng)
    elif mode == "light" and rng.random() < 0.5:
        m = spec_augment(m, rng, n_freq=1, max_f=6, n_time=1, max_t=30)
    return m[..., None]


# --------------------------------------------------------------------------- #
# Inference helpers (used by Streamlit)
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def prepare_inference(src: AudioSource, cfg: PreprocConfig, norm: Normalizer,
                      trim: bool = True) -> dict:
    """Uploaded recording -> everything the app needs.

    Returns dict(wave, sr, quality_warnings, x [N, n_mels, n_frames, 1],
                 starts_s [N], full_log_mel [n_mels, T], duration_s)
    """
    raw, sr0 = load_audio(src)
    warnings = assess_quality(raw, sr0)
    wave = preprocess_recording(raw, sr0, cfg)
    t0 = 0.0
    if trim:
        wave, s0 = trim_silence_edges(wave, cfg.sr)
        t0 = s0 / cfg.sr
    n, hop = cfg.segment_samples, int(cfg.window_hop_s * cfg.sr)
    wins = cycle_windows(wave, n, hop, cfg.sr)
    starts = (np.arange(len(wins)) * hop / cfg.sr + t0) if len(wave) > n else np.array([t0])
    x = np.stack([make_model_input(w, cfg, norm) for w in wins])
    return dict(wave=wave, sr=cfg.sr, quality_warnings=warnings, x=x, starts_s=starts,
                full_log_mel=norm(log_mel(wave, cfg)), duration_s=len(wave) / cfg.sr)


def aggregate_window_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Mean of per-window probabilities -> recording-level class probabilities."""
    return softmax(logits, temperature).mean(axis=0)


def to_multilabel(p: np.ndarray) -> Dict[str, float]:
    """4-class probs -> clinically intuitive marginals."""
    return dict(crackle=float(p[1] + p[3]), wheeze=float(p[2] + p[3]))
