"""
Locally-trained spoof detector: frozen wav2vec2 encoder + a trained head.

Supports both legacy heads (trained on Kaggle corpus) and new MLAAD-trained heads
with deeper architecture (LayerNorm + GELU).
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from backend.config import PROJECT_ROOT, config

HEAD_PATH = PROJECT_ROOT / "checkpoints" / "detector_head.pt"


class LegacyHead(nn.Module):
    """Legacy 2-layer MLP head from scripts/train_detector.py."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


class ImprovedHead(nn.Module):
    """Deeper head with LayerNorm + GELU from scripts/train_mlaad.py."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


# Backward compatibility alias
Head = LegacyHead


class TrainedSpoofDetector:
    """
    Same surface as the other detectors: ``predict``, ``predict_batch``,
    ``preprocess``, ``is_calibrated``, ``status``.
    """

    def __init__(self, head_path: Path | None = None, device: str | None = None):
        self.device = torch.device(device or config.device)
        self.target_sr = config.audio.sample_rate
        self.input_length = config.model.aasist_input_length
        self.head_path = Path(head_path or HEAD_PATH)

        self.is_calibrated = False
        self.backend = "uncalibrated"
        self.encoder = None
        self.feature_extractor = None
        self.head = None

        self._load()

    def _load(self) -> None:
        if not self.head_path.exists():
            print(
                f"[TrainedSpoofDetector] No head at {self.head_path}. "
                f"Train one with: python scripts/train_mlaad.py"
            )
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModel

            model_id = config.model.wav2vec_model_id
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            encoder = AutoModel.from_pretrained(model_id)
            encoder.to(self.device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            self.encoder = encoder

            blob = torch.load(self.head_path, map_location=self.device, weights_only=False)
            version = blob.get("version", "legacy")
            if version == "mlaad_v1":
                head = ImprovedHead(blob["dim"])
            else:
                head = LegacyHead(blob["dim"])
            head.load_state_dict(blob["state_dict"])
            head.to(self.device)
            head.eval()
            self.head = head

            self.is_calibrated = True
            self.backend = f"trained-head:{model_id} ({version})"
            print(f"[TrainedSpoofDetector] Loaded head ({version}) from {self.head_path.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[TrainedSpoofDetector] Could not load: {e}")

    @property
    def status(self) -> dict:
        return {
            "backend": self.backend,
            "calibrated": self.is_calibrated,
            "checkpoint": self.head_path.name if self.is_calibrated else None,
        }

    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Mono, 16 kHz, centre window. The feature extractor handles gain."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sr:
            waveform = torchaudio.transforms.Resample(sample_rate, self.target_sr)(waveform)

        length = waveform.shape[1]
        if length == 0:
            waveform = torch.zeros(1, self.input_length)
        elif length < self.input_length:
            repeats = self.input_length // length + 1
            waveform = waveform.repeat(1, repeats)[:, : self.input_length]
        elif length > self.input_length:
            start = (length - self.input_length) // 2
            waveform = waveform[:, start : start + self.input_length]
        return waveform

    @torch.no_grad()
    def _embed(self, signal: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(
            signal.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.encoder(**inputs).last_hidden_state.mean(dim=1)

    @torch.no_grad()
    def predict(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """Probability the audio is genuine human speech. 1.0 = genuine."""
        if not self.is_calibrated:
            return 0.5
        x = self.preprocess(waveform, sample_rate).squeeze(0)
        logits = self.head(self._embed(x))
        return float(F.softmax(logits, dim=-1)[0, 1].item())

    @torch.no_grad()
    def predict_batch(self, waveforms: list[torch.Tensor], sample_rate: int) -> list[float]:
        if not self.is_calibrated:
            return [0.5] * len(waveforms)
        return [self.predict(w, sample_rate) for w in waveforms]
