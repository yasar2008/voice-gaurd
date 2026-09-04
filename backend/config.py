"""
Centralized configuration for the Voice Clone Detector.

All tunable parameters — model paths, audio processing constants,
fusion weights, and server settings — live here. This makes the
system's behavior transparent and easily adjustable.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cuda_available() -> bool:
    """Probed lazily so importing config never hard-depends on torch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class ModelConfig(BaseSettings):
    """Model checkpoint paths and parameters."""

    # AASIST-L spoof detection
    aasist_checkpoint: Path = Field(
        default=PROJECT_ROOT / "checkpoints" / "AASIST-L.pth",
        description="Path to AASIST-L pretrained checkpoint",
    )
    aasist_input_length: int = Field(
        default=64600,
        description="Expected input length in samples (~4.04s at 16kHz)",
    )

    # ECAPA-TDNN speaker verification
    ecapa_source: str = Field(
        default="speechbrain/spkrec-ecapa-voxceleb",
        description="HuggingFace model ID or local path for ECAPA-TDNN",
    )
    ecapa_savedir: Path = Field(
        default=PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb",
        description="Local cache directory for ECAPA-TDNN weights",
    )
    ecapa_embedding_dim: int = Field(
        default=192,
        description="Dimensionality of speaker embeddings",
    )

    # Spoof detector selection. "wav2vec2" is the default on measured grounds:
    # against 11 genuine human recordings and 5 TTS renders it scored EER 4.5%
    # vs AASIST-L's 23.6%, and crucially it rates genuine speech at 0.728 where
    # AASIST rates it 0.152 (i.e. AASIST calls real voices fake).
    spoof_backend: str = Field(
        default="finetuned",
        description=(
            "Which spoof detector to run. 'finetuned' is the fine-tuned encoder: "
            "87-93% on unseen ElevenLabs variants, 86.7% mean genuine. "
            "'trained' is a head on frozen features, 'wav2vec2' the stock "
            "pretrained classifier (0% on ElevenLabs), 'aasist' the legacy model."
        ),
    )
    wav2vec_model_id: str = Field(
        default="mo-thecreator/Deepfake-audio-detection",
        description="HuggingFace id for the wav2vec2 spoof classifier (Apache-2.0)",
    )


class AudioConfig(BaseSettings):
    """Audio processing parameters."""

    sample_rate: int = Field(
        default=16000,
        description="Target sample rate for all models (Hz)",
    )
    chunk_duration_s: float = Field(
        default=3.0,
        description="Duration of each WebSocket audio chunk (seconds)",
    )
    max_buffer_duration_s: float = Field(
        default=10.0,
        description="Maximum rolling buffer duration before dropping old frames (seconds)",
    )
    pre_emphasis_coeff: float = Field(
        default=0.97,
        description="Pre-emphasis filter coefficient for AASIST preprocessing",
    )

    # Level normalisation. AASIST is NOT level-invariant: the same clip scored at
    # peak 0.70 vs peak 0.035 moves from bonafide 0.000 to 0.830. Training data
    # (ASVspoof 2019 LA) sits around RMS 0.1, while a laptop mic with AGC off is
    # often 10-20x quieter, so without this the recording level dominates the
    # verdict. Measured, not assumed - see scripts/level_sensitivity.py.
    normalize_input: bool = Field(
        default=True,
        description="RMS-normalise audio before spoof inference",
    )
    target_rms: float = Field(
        default=0.10,
        description="Target RMS for input normalisation (matches ASVspoof-like levels)",
    )
    min_rms_for_normalize: float = Field(
        default=0.0005,
        description="Below this RMS the window is treated as silence and left alone",
    )
    normalize_peak_ceiling: float = Field(
        default=0.95,
        description="Post-normalisation peak clamp, to avoid clipping on transients",
    )


class FusionConfig(BaseSettings):
    """Fusion layer weights and thresholds."""

    # Component weights (must sum to 1.0).
    #
    # Set from measurement, not intuition. Scored against every speech-like clip
    # available (5 SAPI TTS renders + the 3 synthetic fixtures):
    #   AASIST  spread 0.454, tracks the label
    #   prosody spread 0.209, and points the WRONG WAY - it returned its highest
    #           naturalness score (1.000) for all five genuine TTS renders, whose
    #           jitter/shimmer/HNR all sit inside the "human" reference bands.
    # So the spoof model carries the decision and prosody is kept only as a small
    # tie-breaker until its ranges are recalibrated against real recordings.
    #
    # Speaker similarity answers a different question ("is this the enrolled
    # person?") and is dropped entirely when nobody is enrolled, so for plain
    # synthetic-vs-natural the effective split is 0.875 / 0.125.
    weight_spoof_detection: float = Field(
        default=0.70,
        description="Weight for AASIST-L spoof detection score",
    )
    weight_speaker_similarity: float = Field(
        default=0.20,
        description="Weight for ECAPA-TDNN speaker verification score (enrolled only)",
    )
    weight_prosody_naturalness: float = Field(
        default=0.10,
        description="Weight for prosody-based naturalness score",
    )

    # Alert threshold
    alert_threshold: float = Field(
        default=65.0,
        description="Risk score threshold (0-100) above which an alert is raised",
    )
    # Lower band boundary. This value already governed the UI's "Uncertain"
    # state, hardcoded in frontend/src/lib/verdict.ts; naming it here lets the
    # backend report the same bands rather than the two drifting apart.
    suspicious_threshold: float = Field(
        default=35.0,
        description="Risk score above which a result is 'medium' rather than 'low'",
    )

    # Confidence calibration
    high_confidence_agreement: float = Field(
        default=0.8,
        description="Min agreement ratio between signals for 'high' confidence",
    )
    low_confidence_agreement: float = Field(
        default=0.4,
        description="Below this agreement ratio, confidence is 'low'",
    )

    @property
    def weights(self) -> dict[str, float]:
        """Return weights as a dict for the RiskScorer."""
        return {
            "spoof_detection": self.weight_spoof_detection,
            "speaker_similarity": self.weight_speaker_similarity,
            "prosody_naturalness": self.weight_prosody_naturalness,
        }


class ProsodyConfig(BaseSettings):
    """Reference ranges for prosody features (derived from bonafide speech)."""

    # Normal ranges — values outside these suggest synthetic speech
    # NOTE: these bands are inherited defaults and have NOT held up under test.
    # Measured on Windows SAPI renders: jitter 0.016-0.018, shimmer 0.078-0.104,
    # pitch_std 13.8-47.3, HNR 9.5-14.4 - every value comfortably inside the
    # "human" band, so the clips scored 1.000 natural. The "TTS often < X" claims
    # below describe older parametric synthesis, not what these voices produce.
    # Recalibrate from real recordings before trusting this signal.
    jitter_range: tuple[float, float] = Field(
        default=(0.001, 0.03),
        description="Normal jitter range (fraction). Unverified against modern TTS.",
    )
    shimmer_range: tuple[float, float] = Field(
        default=(0.02, 0.15),
        description="Normal shimmer range (dB). Unverified against modern TTS.",
    )
    pitch_std_range: tuple[float, float] = Field(
        default=(10.0, 80.0),
        description="Normal pitch std range (Hz). Expressive real speech can exceed 80.",
    )
    hnr_range: tuple[float, float] = Field(
        default=(5.0, 25.0),
        description="Normal harmonics-to-noise ratio range (dB).",
    )
    pause_ratio_range: tuple[float, float] = Field(
        default=(0.05, 0.40),
        description="Normal pause ratio (fraction of silence).",
    )


class QualityConfig(BaseSettings):
    """
    Input conditions the spoof detector needs to give a meaningful answer.
    Measured limits, not guesses — see backend/features/audio_quality.py.
    """

    min_snr_db: float = Field(
        default=30.0,
        description="Below this SNR synthetic speech evades detection (measured: fails at 25 dB)",
    )
    min_high_freq_db: float = Field(
        default=-60.0,
        description=(
            "Energy above 4 kHz relative to total, in dB. Full-band speech is -18 to -32; "
            "telephone-band is about -80, where genuine voices score 0.000."
        ),
    )
    min_duration_s: float = Field(
        default=2.0,
        description="Shorter windows get padded and the score degrades",
    )
    enforce: bool = Field(
        default=True,
        description="Withhold a verdict when input is outside the reliable range",
    )


class ServerConfig(BaseSettings):
    """FastAPI server settings."""

    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, description="Server port")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins (frontend dev server)",
    )
    log_level: str = Field(default="info", description="Uvicorn log level")


class Config(BaseSettings):
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Device selection. Auto-detects CUDA so the same config works on a laptop
    # with a GPU and on a CPU-only box; override with VCD_DEVICE=cpu.
    device: str = Field(
        default_factory=lambda: "cuda" if _cuda_available() else "cpu",
        description="Torch device: 'cpu', 'cuda', or 'cuda:0'",
    )

    model_config = SettingsConfigDict(
        env_prefix="VCD_",
        env_nested_delimiter="__",
    )


# Singleton instance — import this throughout the codebase
config = Config()
