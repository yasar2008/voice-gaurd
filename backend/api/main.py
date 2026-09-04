"""
FastAPI application entry point.

Sets up the app with:
- Lifespan handler for loading models once at startup
- CORS middleware for frontend communication
- Router includes for REST and WebSocket endpoints
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as rest_router
from backend.api.ws import router as ws_router
from backend.config import config
from backend.features.prosody import ProsodyAnalyzer
from backend.fusion.risk_scorer import RiskScorer
from backend.models.aasist import AASISTDetector
from backend.models.finetuned_spoof import FineTunedSpoofDetector
from backend.models.speaker_verify import SpeakerVerifier
from backend.models.trained_spoof import TrainedSpoofDetector
from backend.models.wav2vec_spoof import Wav2VecSpoofDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all ML models once at startup; clean up on shutdown.
    
    Models are stored in app.state so route handlers can access them
    without re-loading on every request.
    """
    start = time.time()
    print("[Startup] Loading models...")

    # Spoof detection model. Default is the wav2vec2 classifier; AASIST-L stays
    # available via VCD_MODEL__SPOOF_BACKEND=aasist for comparison.
    if config.model.spoof_backend == "finetuned":
        detector = FineTunedSpoofDetector(device=config.device)
        # Fall back rather than serve an uncalibrated detector.
        app.state.spoof_detector = (
            detector if detector.is_calibrated else Wav2VecSpoofDetector(device=config.device)
        )
    elif config.model.spoof_backend == "aasist":
        app.state.spoof_detector = AASISTDetector(
            checkpoint_path=str(config.model.aasist_checkpoint),
            device=config.device,
        )
    elif config.model.spoof_backend == "trained":
        detector = TrainedSpoofDetector(device=config.device)
        # Fall back rather than run uncalibrated if the head was never trained.
        app.state.spoof_detector = (
            detector if detector.is_calibrated else Wav2VecSpoofDetector(device=config.device)
        )
    else:
        app.state.spoof_detector = Wav2VecSpoofDetector(device=config.device)

    # Speaker verification model (ECAPA-TDNN)
    app.state.speaker_verifier = SpeakerVerifier(device=config.device)

    # Prosody analyzer (CPU-based, no model loading needed)
    app.state.prosody_analyzer = ProsodyAnalyzer()

    # Risk scorer (fusion layer)
    app.state.risk_scorer = RiskScorer()

    elapsed = time.time() - start
    print(f"[Startup] All models loaded in {elapsed:.1f}s")

    yield  # App is running

    # Cleanup
    print("[Shutdown] Cleaning up...")
    del app.state.spoof_detector
    del app.state.speaker_verifier


# Create the FastAPI app
app = FastAPI(
    title="Voice Clone Detector API",
    description=(
        "Real-time detection of voice-cloning impersonation attacks. "
        "Fuses spoof detection (AASIST-L), speaker verification (ECAPA-TDNN), "
        "and prosody analysis into a 0–100 risk score."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rest_router)
app.include_router(ws_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=True,
    )
