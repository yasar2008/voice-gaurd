"""
WebSocket handler for real-time streaming audio analysis.

Protocol:
    Client → Server: Binary audio chunks (PCM 16-bit signed LE, 16kHz mono)
    Server → Client: JSON risk score updates every ~1–3 seconds

The handler maintains a rolling buffer of audio data. When enough data
accumulates (configurable via CHUNK_DURATION_S), it runs the full
detection pipeline and sends a risk update back to the client.

Architecture:
    ┌──────────┐   binary PCM    ┌──────────────┐   risk JSON   ┌──────────┐
    │  Client  │ ──────────────→ │  WS Handler  │ ────────────→ │  Client  │
    │ (browser)│                 │ (ring buffer  │               │ (browser)│
    │          │ ←────────────── │  + pipeline)  │               │          │
    └──────────┘                 └──────────────┘               └──────────┘
"""

import logging
import struct
import time

import numpy as np
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.result_format import detection_summary
from backend.config import config
from backend.features.audio_quality import assess

router = APIRouter()

logger = logging.getLogger("voiceguard.stream")

#: Windows quieter than this (RMS over the analysis window) are treated as
#: silence and skipped rather than scored.
SILENCE_RMS_THRESHOLD = 0.004


class AudioRingBuffer:
    """
    Rolling ring buffer for streaming audio data.
    
    Stores raw PCM samples as float32 numpy array. Automatically drops
    oldest data when the buffer exceeds max_duration_s.
    """

    def __init__(self, sample_rate: int, max_duration_s: float):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.buffer = np.zeros(0, dtype=np.float32)

    def add(self, samples: np.ndarray) -> None:
        """Append new samples, dropping oldest if buffer exceeds max size."""
        self.buffer = np.concatenate([self.buffer, samples])
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples :]

    def get_latest(self, duration_s: float) -> np.ndarray:
        """Get the most recent `duration_s` seconds of audio."""
        num_samples = int(duration_s * self.sample_rate)
        if len(self.buffer) < num_samples:
            # Pad with zeros if not enough data yet
            pad = np.zeros(num_samples - len(self.buffer), dtype=np.float32)
            return np.concatenate([pad, self.buffer])
        return self.buffer[-num_samples:]

    @property
    def duration_s(self) -> float:
        """Current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.zeros(0, dtype=np.float32)


def pcm16_to_float(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert raw PCM 16-bit signed little-endian bytes to float32 numpy array.
    
    The browser's ScriptProcessorNode or AudioWorklet sends PCM data in this format.
    """
    # Unpack as int16
    num_samples = len(pcm_bytes) // 2
    if num_samples == 0:
        return np.zeros(0, dtype=np.float32)

    samples = struct.unpack(f"<{num_samples}h", pcm_bytes[:num_samples * 2])
    # Normalize to [-1.0, 1.0]
    return np.array(samples, dtype=np.float32) / 32768.0


@router.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    Real-time audio analysis over WebSocket.
    
    Client sends binary PCM audio chunks (16-bit signed LE, 16kHz, mono).
    Server responds with JSON risk updates when enough audio accumulates.
    
    Connection lifecycle:
        1. Client connects
        2. Server sends initial config (sample rate, chunk size)
        3. Client streams audio chunks
        4. Server responds with risk scores periodically
        5. Client disconnects (or server closes on error)
    """
    await websocket.accept()

    # Access models from app state
    app = websocket.app
    spoof_detector = app.state.spoof_detector
    speaker_verifier = app.state.speaker_verifier
    prosody_analyzer = app.state.prosody_analyzer
    risk_scorer = app.state.risk_scorer

    # Initialize ring buffer
    ring_buffer = AudioRingBuffer(
        sample_rate=config.audio.sample_rate,
        max_duration_s=config.audio.max_buffer_duration_s,
    )

    # Track inference timing
    last_inference_time = 0.0
    chunk_interval_s = config.audio.chunk_duration_s
    # Minimum audio needed before first inference (AASIST needs ~4s)
    min_audio_s = config.model.aasist_input_length / config.audio.sample_rate

    # Send initial config to client
    await websocket.send_json(
        {
            "type": "config",
            "sample_rate": config.audio.sample_rate,
            "chunk_duration_s": chunk_interval_s,
            "alert_threshold": risk_scorer.alert_threshold,
            "enrolled_speaker": speaker_verifier.is_enrolled,
            "calibrated": spoof_detector.is_calibrated,
        }
    )

    try:
        while True:
            # Receive binary audio data
            data = await websocket.receive_bytes()

            # Convert PCM bytes to float samples and add to buffer
            samples = pcm16_to_float(data)
            ring_buffer.add(samples)

            # Check if enough time has elapsed and we have enough audio
            current_time = time.time()
            time_since_last = current_time - last_inference_time

            if time_since_last >= chunk_interval_s and ring_buffer.duration_s >= min_audio_s:
                # Skip windows that are essentially silence. Scoring them is worse
                # than useless: with no voiced frames the prosody extractor reports
                # pitch_std = 0, which reads as flat synthetic pitch and pushes the
                # risk score up whenever the speaker simply stops talking.
                window_rms = float(np.sqrt(np.mean(ring_buffer.get_latest(min_audio_s) ** 2)))
                if window_rms < SILENCE_RMS_THRESHOLD:
                    last_inference_time = current_time
                    await websocket.send_json(
                        {"type": "idle", "reason": "no_speech", "rms": round(window_rms, 5)}
                    )
                    continue

                # Refuse to answer on audio the detector cannot judge. Measured:
                # synthetic speech evades below ~25 dB SNR, and genuine speech
                # reads as fake on telephone-band audio. Without this the app
                # reports "Real voice" for everything in a noisy room.
                quality = assess(ring_buffer.get_latest(min_audio_s), config.audio.sample_rate)
                if config.quality.enforce and not quality["reliable"]:
                    last_inference_time = current_time
                    await websocket.send_json(
                        {
                            "type": "unreliable",
                            "quality": quality,
                            "reasons": quality["reasons"],
                            # Same keys as risk_update so consumers need one parser.
                            **detection_summary(0.0, False, 0.5, reliable=False),
                        }
                    )
                    continue

                last_inference_time = current_time
                inference_start = time.time()

                # Get audio window for inference
                # AASIST needs exactly 64600 samples (~4.04s)
                aasist_audio = ring_buffer.get_latest(min_audio_s)
                # Prosody and speaker verify can use the full chunk
                full_audio = ring_buffer.get_latest(
                    min(ring_buffer.duration_s, config.audio.max_buffer_duration_s)
                )

                # 1. Spoof detection
                aasist_tensor = torch.from_numpy(aasist_audio).unsqueeze(0)  # [1, T]
                spoof_score = spoof_detector.predict(aasist_tensor, config.audio.sample_rate)

                # 2. Speaker verification — None when nothing is enrolled
                full_tensor = torch.from_numpy(full_audio).unsqueeze(0)
                speaker_score = (
                    speaker_verifier.verify(full_tensor, config.audio.sample_rate)
                    if speaker_verifier.is_enrolled
                    else None
                )

                # 3. Prosody analysis
                prosody_result = prosody_analyzer.extract(
                    full_audio.astype(np.float64), config.audio.sample_rate
                )
                prosody_score = prosody_result["prosody_score"]

                # 4. Fusion
                risk_result = risk_scorer.compute(
                    spoof_score=spoof_score,
                    speaker_score=speaker_score,
                    prosody_score=prosody_score,
                )
                risk_result.anomalies = prosody_result.get("anomalies", [])

                inference_ms = (time.time() - inference_start) * 1000

                # Send result to client
                result_dict = risk_result.to_dict()
                summary = detection_summary(
                    result_dict["risk_score"], result_dict["alert"], spoof_score
                )
                logger.info(
                    "stream label=%s genuine=%.3f risk=%s score=%.1f inference_ms=%.1f",
                    summary["label"], summary["genuine_probability"], summary["risk"],
                    summary["risk_score"], inference_ms,
                )
                await websocket.send_json(
                    {
                        "type": "risk_update",
                        **summary,
                        "timestamp": current_time,
                        "risk_score": result_dict["risk_score"],
                        "alert": result_dict["alert"],
                        "confidence": result_dict["confidence"],
                        "breakdown": result_dict["breakdown"],
                        "anomalies": result_dict["anomalies"],
                        "prosody_detail": {
                            "pitch_mean": round(prosody_result["pitch_mean"], 2),
                            "pitch_std": round(prosody_result["pitch_std"], 2),
                            "jitter": round(prosody_result["jitter"], 6),
                            "shimmer": round(prosody_result["shimmer"], 4),
                            "hnr": round(prosody_result["hnr"], 2),
                            "pause_ratio": round(prosody_result["pause_ratio"], 3),
                        },
                        "calibrated": spoof_detector.is_calibrated,
                        "quality": quality,
                        "buffer_duration_s": round(ring_buffer.duration_s, 2),
                        "inference_ms": round(inference_ms, 1),
                    }
                )

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        ring_buffer.clear()
