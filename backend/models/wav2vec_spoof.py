"""
Wav2Vec2-based synthetic speech detector.

Replaces AASIST-L as the default spoof model. The reason is measured, not
architectural preference — scored against 11 genuine human recordings
(LibriSpeech/CommonVoice/IEMOCAP samples) and 5 known TTS renders:

    model      genuine P(real)   TTS P(real)   margin     EER
    AASIST-L             0.152         0.091    0.061   23.6%
    wav2vec2             0.728         0.000    0.728    4.5%

AASIST-L scored *genuine human speech* at 0.152 — i.e. it called real voices
fake, which is exactly the false-alarm behaviour reported in use. Its published
~1% EER holds on ASVspoof 2019 LA, the corpus it was trained on; it does not
survive contact with consumer-grade recordings.

Known limitation, also measured: neither model detects the high-quality voice
*conversions* in the UniData Kaggle set (both at chance). A conversion inherits
the original recording's room, microphone and channel, leaving far less for a
detector to grip. Treat voice conversion as out of scope for this model.

The HF feature extractor applies zero-mean/unit-variance normalisation, so this
detector is inherently level-invariant and does not need the RMS pre-scaling the
AASIST path requires.
"""

import torch
import torch.nn.functional as F
import torchaudio

from backend.config import config


class Wav2VecSpoofDetector:
    """
    Drop-in replacement for :class:`~backend.models.aasist.AASISTDetector`.

    Exposes the same surface — ``predict``, ``predict_batch``, ``preprocess``,
    ``is_calibrated`` and ``status`` — so routes, the WebSocket handler and the
    calibration scripts work against either without changes.

    Usage:
        detector = Wav2VecSpoofDetector()
        score = detector.predict(waveform, sample_rate=44100)
        # score ∈ [0.0, 1.0] — higher = more likely genuine human speech
    """

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.device = torch.device(device or config.device)
        self.target_sr = config.audio.sample_rate
        self.input_length = config.model.aasist_input_length
        self.model_id = model_id or config.model.wav2vec_model_id

        self.is_calibrated = False
        self.backend = "uncalibrated"
        self.model = None
        self.feature_extractor = None
        self._real_index = 1

        self._load()

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            model.to(self.device)
            model.eval()
            self.model = model

            # Label order is the checkpoint's business, not ours — read it rather
            # than assuming index 1 means "real".
            id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
            real = [i for i, lab in id2label.items() if lab in {"real", "bonafide", "genuine"}]
            if not real:
                raise ValueError(f"No real/bonafide class in id2label={id2label}")
            self._real_index = real[0]

            self.is_calibrated = True
            self.backend = f"wav2vec2:{self.model_id}"
            n = sum(p.numel() for p in model.parameters())
            print(f"[Wav2VecSpoofDetector] Loaded {self.model_id} ({n:,} params)")
        except Exception as e:  # noqa: BLE001 — any failure means "not calibrated"
            print(
                f"[Wav2VecSpoofDetector] Could not load {self.model_id}: {e}. "
                f"Run: python scripts/download_checkpoints.py"
            )

    @property
    def status(self) -> dict:
        """Machine-readable state for the /health endpoint."""
        return {
            "backend": self.backend,
            "calibrated": self.is_calibrated,
            "checkpoint": self.model_id if self.is_calibrated else None,
        }

    # -- preprocessing ------------------------------------------------------

    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Mono, 16 kHz, fixed length. No gain normalisation: the feature extractor
        standardises the signal itself, so scaling here would be redundant.
        """
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

    # -- inference ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """
        Returns the probability that the audio is genuine human speech.

        Same polarity as the AASIST detector it replaces: 1.0 = genuine,
        0.0 = synthetic. Meaningless unless `is_calibrated` is True.
        """
        if not self.is_calibrated:
            return 0.5

        x = self.preprocess(waveform, sample_rate).squeeze(0)
        inputs = self.feature_extractor(
            x.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        probs = F.softmax(self.model(**inputs).logits, dim=-1)
        return float(probs[0, self._real_index].item())

    @torch.no_grad()
    def predict_batch(self, waveforms: list[torch.Tensor], sample_rate: int) -> list[float]:
        """Score several clips. Returns one genuine-probability per input."""
        if not self.is_calibrated:
            return [0.5] * len(waveforms)

        batch = [self.preprocess(w, sample_rate).squeeze(0).cpu().numpy() for w in waveforms]
        inputs = self.feature_extractor(
            batch, sampling_rate=self.target_sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        probs = F.softmax(self.model(**inputs).logits, dim=-1)
        return [float(p[self._real_index].item()) for p in probs]
