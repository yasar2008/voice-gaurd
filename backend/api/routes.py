"""
REST API routes for the Voice Clone Detector.

Endpoints:
    GET  /health       — Health check + model status
    POST /analyze      — Upload a WAV/FLAC file → full risk analysis
    POST /enroll       — Upload reference voice → store speaker embedding
    DELETE /enroll      — Clear enrolled speaker
    GET  /config       — View current fusion weights and thresholds
    PUT  /config       — Update fusion weights and thresholds
"""

import io
import logging
import struct
import time

import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.api.result_format import detection_summary
from backend.config import PROJECT_ROOT, config
from backend.data.audio_io import load_audio
from backend.features.audio_quality import assess

#: Folders under data/eval that /capture is allowed to write to. A fixed set,
#: not user-supplied path components — the label never touches the filesystem
#: except by lookup here.
CAPTURE_LABELS = {"bonafide", "spoof"}

router = APIRouter()

logger = logging.getLogger("voiceguard.api")


# ---------------------------------------------------------------------------
# Pydantic models for request/response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    device: str
    enrolled_speaker: bool
    spoof_detector: dict


class AnalysisResponse(BaseModel):
    risk_score: float
    alert: bool
    confidence: str
    breakdown: dict
    prosody_detail: dict
    anomalies: list[str]
    latency_ms: float
    calibrated: bool
    quality: dict
    # Flat summary, identical keys to the WebSocket risk_update frame.
    label: str
    genuine_probability: float
    synthetic_probability: float
    risk: str


class ChunkResponse(BaseModel):
    """Low-latency response for streaming clients (Meet integration)."""

    label: str
    genuine_probability: float
    synthetic_probability: float
    risk: str
    risk_score: float
    reliable: bool
    reasons: list[str]
    inference_ms: float


class ConfigResponse(BaseModel):
    fusion_weights: dict[str, float]
    alert_threshold: float
    prosody_ranges: dict


class ConfigUpdateRequest(BaseModel):
    fusion_weights: dict[str, float] | None = None
    alert_threshold: float | None = None


# ---------------------------------------------------------------------------
# Helper: load audio from uploaded file
# ---------------------------------------------------------------------------

def _spoof_over_windows(detector, waveform: torch.Tensor, sr: int, k: int = 5) -> float:
    """Median spoof score across up to `k` evenly spaced analysis windows."""
    import statistics

    window = config.model.aasist_input_length
    mono = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    # Window length is defined at the model's rate, so compare like with like.
    scale = sr / config.audio.sample_rate
    span = int(window * scale)

    if mono.shape[1] <= span:
        return detector.predict(mono, sr)

    starts = [int(i * (mono.shape[1] - span) / (k - 1)) for i in range(k)]
    scores = [detector.predict(mono[:, s : s + span], sr) for s in starts]
    return float(statistics.median(scores))


async def _load_audio_from_upload(file: UploadFile) -> tuple[torch.Tensor, int]:
    """Read an uploaded audio file and return (waveform, sample_rate)."""
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        buffer = io.BytesIO(contents)
        waveform, sr = load_audio(buffer)
        return waveform, sr
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode audio file: {e}. Supported formats: WAV, FLAC, MP3, OGG.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check — reports model status and enrollment state."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "spoof_detector": request.app.state.spoof_detector is not None,
            "speaker_verifier": request.app.state.speaker_verifier is not None,
            "prosody_analyzer": request.app.state.prosody_analyzer is not None,
            "risk_scorer": request.app.state.risk_scorer is not None,
        },
        device=config.device,
        enrolled_speaker=request.app.state.speaker_verifier.is_enrolled,
        spoof_detector=request.app.state.spoof_detector.status,
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    """
    Upload an audio file and receive a full risk analysis.
    
    The analysis runs three models in parallel:
    1. AASIST-L spoof detection → bonafide confidence
    2. ECAPA-TDNN speaker verification → cosine similarity (if enrolled)
    3. Prosody analysis → naturalness score
    
    These are fused into a single 0–100 risk score.
    """
    start_time = time.time()

    # Load audio
    waveform, sr = await _load_audio_from_upload(file)

    # 1. Spoof detection. A single centre crop of a long file is a lottery —
    #    measured on one 40s recording, consecutive windows scored 0.003 and
    #    0.999. Score several spread across the file and take the median, which
    #    is what the offline evaluation does and what the live stream approximates
    #    by scoring repeatedly. Short files still yield exactly one window.
    spoof_score = _spoof_over_windows(request.app.state.spoof_detector, waveform, sr)

    # 2. Speaker verification — None (not 0.5) when nothing is enrolled, so the
    #    fusion layer drops the signal instead of charging constant risk for it.
    verifier = request.app.state.speaker_verifier
    speaker_score = verifier.verify(waveform, sr) if verifier.is_enrolled else None

    # 3. Prosody analysis
    waveform_np = waveform.squeeze().numpy().astype(np.float64)
    quality = assess(waveform_np, sr)
    prosody_result = request.app.state.prosody_analyzer.extract(waveform_np, sr)
    prosody_score = prosody_result["prosody_score"]

    # 4. Fusion
    risk_result = request.app.state.risk_scorer.compute(
        spoof_score=spoof_score,
        speaker_score=speaker_score,
        prosody_score=prosody_score,
    )
    risk_result.anomalies = prosody_result.get("anomalies", [])

    latency_ms = (time.time() - start_time) * 1000

    result_dict = risk_result.to_dict()

    return AnalysisResponse(
        risk_score=result_dict["risk_score"],
        alert=result_dict["alert"],
        confidence=result_dict["confidence"],
        breakdown=result_dict["breakdown"],
        prosody_detail={
            "pitch_mean": round(prosody_result["pitch_mean"], 2),
            "pitch_std": round(prosody_result["pitch_std"], 2),
            "jitter": round(prosody_result["jitter"], 6),
            "shimmer": round(prosody_result["shimmer"], 4),
            "hnr": round(prosody_result["hnr"], 2),
            "pause_ratio": round(prosody_result["pause_ratio"], 3),
            "speaking_rate": round(prosody_result["speaking_rate"], 2),
        },
        anomalies=result_dict["anomalies"],
        latency_ms=round(latency_ms, 1),
        calibrated=request.app.state.spoof_detector.is_calibrated,
        quality=quality,
        **detection_summary(
            result_dict["risk_score"],
            result_dict["alert"],
            spoof_score,
            reliable=quality["reliable"],
        ),
    )


@router.post("/enroll")
async def enroll_speaker(request: Request, file: UploadFile = File(...)):
    """
    Enroll a reference speaker for verification.
    
    Upload 5–15 seconds of clean speech from the target speaker.
    Future analyses will compare against this enrollment.
    """
    waveform, sr = await _load_audio_from_upload(file)

    duration_s = waveform.shape[1] / sr
    if duration_s < 1.0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Audio too short ({duration_s:.1f}s). "
                f"Provide at least 1 second, ideally 5-15 seconds."
            ),
        )

    embedding = request.app.state.speaker_verifier.enroll(waveform, sr)

    return {
        "status": "enrolled",
        "duration_s": round(duration_s, 2),
        "embedding_dim": len(embedding),
        "embedding_norm": round(float(np.linalg.norm(embedding)), 4),
    }


@router.delete("/enroll")
async def clear_enrollment(request: Request):
    """Clear the enrolled speaker, disabling speaker verification."""
    request.app.state.speaker_verifier.clear_enrollment()
    return {"status": "enrollment_cleared"}


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """View current fusion configuration."""
    scorer = request.app.state.risk_scorer
    return ConfigResponse(
        fusion_weights=scorer.weights,
        alert_threshold=scorer.alert_threshold,
        prosody_ranges={
            "jitter": list(config.prosody.jitter_range),
            "shimmer": list(config.prosody.shimmer_range),
            "pitch_std": list(config.prosody.pitch_std_range),
            "hnr": list(config.prosody.hnr_range),
            "pause_ratio": list(config.prosody.pause_ratio_range),
        },
    )


@router.put("/config")
async def update_config(request: Request, update: ConfigUpdateRequest):
    """
    Update fusion weights and/or alert threshold at runtime.
    
    This does NOT persist across restarts — it only affects the current session.
    """
    scorer = request.app.state.risk_scorer

    if update.fusion_weights is not None:
        total = sum(update.fusion_weights.values())
        if abs(total - 1.0) > 0.01:
            raise HTTPException(
                status_code=400,
                detail=f"Fusion weights must sum to 1.0, got {total:.3f}",
            )
        scorer.weights = update.fusion_weights

    if update.alert_threshold is not None:
        if not 0 <= update.alert_threshold <= 100:
            raise HTTPException(
                status_code=400,
                detail="Alert threshold must be between 0 and 100",
            )
        scorer.alert_threshold = update.alert_threshold

    return {
        "status": "updated",
        "fusion_weights": scorer.weights,
        "alert_threshold": scorer.alert_threshold,
    }


@router.post("/capture")
async def capture_sample(
    request: Request,
    file: UploadFile = File(...),
    label: str = Form("bonafide"),
):
    """
    Save a recording into the calibration set and score it in one step.

    Exists because collecting genuine speech is the bottleneck for tuning the
    detector, and the obvious manual route is a trap: Windows Voice Recorder
    writes M4A, which libsndfile cannot decode. The browser records 16 kHz WAV
    through the same capture path the live analyser uses, so what lands on disk
    is exactly what the detector sees in production.
    """
    if label not in CAPTURE_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"label must be one of {sorted(CAPTURE_LABELS)}",
        )

    waveform, sr = await _load_audio_from_upload(file)
    duration_s = waveform.shape[1] / sr
    min_s = config.model.aasist_input_length / config.audio.sample_rate
    if duration_s < min_s:
        raise HTTPException(
            status_code=400,
            detail=f"Clip is {duration_s:.1f}s; the spoof model needs at least {min_s:.1f}s.",
        )

    dest_dir = PROJECT_ROOT / "data" / "eval" / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(str(dest), waveform.mean(dim=0).numpy(), sr)

    # Score it with the same pipeline /analyze uses, so the number shown next to
    # the saved file is the number calibrate.py will reproduce.
    spoof_score = request.app.state.spoof_detector.predict(waveform, sr)
    verifier = request.app.state.speaker_verifier
    speaker_score = verifier.verify(waveform, sr) if verifier.is_enrolled else None
    prosody_result = request.app.state.prosody_analyzer.extract(
        waveform.squeeze().numpy().astype(np.float64), sr
    )
    risk = request.app.state.risk_scorer.compute(
        spoof_score=spoof_score,
        speaker_score=speaker_score,
        prosody_score=prosody_result["prosody_score"],
    )

    return {
        "saved_as": dest.name,
        "folder": f"data/eval/{label}",
        "duration_s": round(duration_s, 2),
        "sample_rate": sr,
        "peak": round(float(waveform.abs().max()), 4),
        "rms": round(float(waveform.pow(2).mean().sqrt()), 4),
        "risk_score": round(risk.risk_score, 1),
        "alert": risk.alert,
        "bonafide_score": round(spoof_score, 4),
        "prosody_score": round(prosody_result["prosody_score"], 4),
        "calibrated": request.app.state.spoof_detector.is_calibrated,
        "count_in_folder": len(list(dest_dir.glob("*.wav"))),
    }


@router.post("/analyze/chunk", response_model=ChunkResponse)
async def analyze_chunk(
    request: Request,
    sample_rate: int = 16000,
    enforce_quality: bool = True,
):
    """
    Score one short window of raw PCM. Built for near-real-time callers.

    Body: raw 16-bit signed little-endian mono PCM at `sample_rate`. No
    container, no multipart — a streaming client should not pay WAV-header or
    form-parsing cost per chunk.

    Why this exists alongside /ws/analyze: the WebSocket is the better transport
    for a continuous stream, but a Meet add-on iframe or any stateless caller may
    only be able to POST. Both share the same model instance from app.state and
    the same result shape, so neither path is a second implementation.

    Send at least ~4 seconds of audio; shorter input is repeat-padded by the
    detector's preprocessing and scores less reliably.
    """
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body; expected raw PCM16 bytes")
    if len(raw) < 2:
        raise HTTPException(status_code=400, detail="Body too short to contain audio")

    start = time.time()

    count = len(raw) // 2
    samples = np.array(struct.unpack(f"<{count}h", raw[: count * 2]), dtype=np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)

    quality = assess(samples.astype(np.float64), sample_rate)
    if enforce_quality and config.quality.enforce and not quality["reliable"]:
        # Decline rather than guess — same policy as the WebSocket path.
        summary = detection_summary(0.0, False, 0.5, reliable=False)
        return ChunkResponse(
            **summary,
            reliable=False,
            reasons=quality["reasons"],
            inference_ms=round((time.time() - start) * 1000, 1),
        )

    spoof_score = request.app.state.spoof_detector.predict(waveform, sample_rate)
    verifier = request.app.state.speaker_verifier
    speaker_score = verifier.verify(waveform, sample_rate) if verifier.is_enrolled else None
    prosody = request.app.state.prosody_analyzer.extract(
        samples.astype(np.float64), sample_rate
    )
    risk = request.app.state.risk_scorer.compute(
        spoof_score=spoof_score,
        speaker_score=speaker_score,
        prosody_score=prosody["prosody_score"],
    )

    inference_ms = (time.time() - start) * 1000
    summary = detection_summary(risk.risk_score, risk.alert, spoof_score)
    logger.info(
        "chunk label=%s genuine=%.3f risk=%s score=%.1f inference_ms=%.1f samples=%d",
        summary["label"], summary["genuine_probability"], summary["risk"],
        summary["risk_score"], inference_ms, count,
    )
    return ChunkResponse(
        **summary,
        reliable=quality["reliable"],
        reasons=quality["reasons"],
        inference_ms=round(inference_ms, 1),
    )
