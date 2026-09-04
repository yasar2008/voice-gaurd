"""
Fine-tuned wav2vec2 spoof detector — the project's default.

Unlike every earlier detector here, the encoder itself was trained, not just a
classifier on top of frozen features. That distinction is the whole result.
Measured on held-out generators never seen during training:

    source                        kind        correct    n
    Edge-TTS                      synthetic     99.4%   161
    jay15k_synth                  synthetic     99.5%   200
    ElevenLabs-Turbo-v2.5         synthetic     88.7%   159
    ElevenLabs-v2-Multilingual    synthetic     88.5%   156
    ChatTTS                       synthetic     85.1%   174
    ElevenLabs-v3                 synthetic     84.5%   148
    jay15k_genuine                genuine       99.2%   400
    LibriSpeech (unseen speakers) genuine       85.1%   370

    mean genuine 92.2%   mean synthetic 90.9%

Three training choices produced this, each measured against the alternative:

* **Random crops.** Earlier training cached one centre window per file, so the
  model only ever saw the middle 4 seconds while the app scores every window.
  Fixing that mismatch took ChatTTS from 79.3% to 90.2%.
* **Speech-gated crops.** The live path skips anything below RMS 0.004, so
  training on silence is label noise -- pauses sound the same whoever made them.
* **Noise on both classes.** jay15k genuine is noisy consumer audio while
  LibriSpeech and MLAAD's TTS are both clean, so "clean" leaked label
  information. LibriSpeech genuine sat at 76.5% because of it; adding noise
  symmetrically brought it to 85.1%.

Two things that did NOT work, recorded so they are not retried: balancing the
genuine sources 50/50 moved LibriSpeech 77.0% -> 76.5% (the dilution theory was
wrong), and MP3 at 64k shifts synthetic scores by -0.010, so codec compression
is not a shortcut this model relies on.

Operating point: this checkpoint leans genuine (92.2% / 90.9%), which suppresses
false "synthetic" calls on real speech. checkpoints/finetuned_encoder_synthetic_leaning
is the same recipe without noise augmentation and inverts the trade (87.6% /
95.4%) -- swap it in via config if missing spoofs costs more than false alarms.

Known limit, measured: an independently produced ElevenLabs voice clone -- a
clone of a real person built from a phone recording, rather than TTS reading
text -- scored 0.812 P(genuine), i.e. wrong and confident. Four separate
retrains, including an exact rerun of the earlier recipe, all failed it while
holdout stayed at 90-92%. MLAAD is entirely TTS-from-text with library voices
and contains no cloned-speaker audio, so this category is absent from training.
Holdout cannot see this failure; do not read these numbers as covering it.

Trained by scripts/finetune_encoder.py; evaluated by scripts/eval_finetuned.py.
Training data is CC BY-NC (MLAAD) — keep derived weights non-commercial.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from backend.config import PROJECT_ROOT, config

MODEL_DIR = PROJECT_ROOT / "checkpoints" / "finetuned_encoder"


class FineTunedSpoofDetector:
    """
    Same surface as the other detectors: ``predict``, ``predict_batch``,
    ``preprocess``, ``is_calibrated``, ``status``.
    """

    def __init__(self, model_dir: Path | None = None, device: str | None = None):
        self.device = torch.device(device or config.device)
        self.target_sr = config.audio.sample_rate
        self.input_length = config.model.aasist_input_length
        self.model_dir = Path(model_dir or MODEL_DIR)

        self.is_calibrated = False
        self.backend = "uncalibrated"
        self.model = None
        self.feature_extractor = None
        self._real_index = 1

        self._load()

    def _load(self) -> None:
        if not (self.model_dir / "config.json").exists():
            print(
                f"[FineTunedSpoofDetector] No model at {self.model_dir}. "
                f"Train with: python scripts/finetune_encoder.py"
            )
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                config.model.wav2vec_model_id
            )
            model = AutoModelForAudioClassification.from_pretrained(self.model_dir)
            model.to(self.device)
            model.eval()
            self.model = model

            # Label 1 = genuine, set by scripts/finetune_encoder.py. The base
            # checkpoint's id2label is stale after re-heading, so it is not
            # consulted here — the training script's convention is the contract.
            self._real_index = 1

            self.is_calibrated = True
            self.backend = "finetuned-wav2vec2"
            n = sum(p.numel() for p in model.parameters())
            print(f"[FineTunedSpoofDetector] Loaded fine-tuned encoder ({n:,} params)")
        except Exception as e:  # noqa: BLE001 — any failure means "not calibrated"
            print(f"[FineTunedSpoofDetector] Could not load: {e}")

    @property
    def status(self) -> dict:
        return {
            "backend": self.backend,
            "calibrated": self.is_calibrated,
            "checkpoint": self.model_dir.name if self.is_calibrated else None,
        }

    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Mono, 16 kHz, fixed window. Gain is handled by the feature extractor."""
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
    def predict(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """Probability the audio is genuine human speech. 1.0 = genuine."""
        if not self.is_calibrated:
            return 0.5
        x = self.preprocess(waveform, sample_rate).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = self.model(input_values=x).logits
        return float(F.softmax(logits.float(), dim=-1)[0, self._real_index].item())

    @torch.no_grad()
    def predict_batch(self, waveforms: list[torch.Tensor], sample_rate: int) -> list[float]:
        if not self.is_calibrated:
            return [0.5] * len(waveforms)
        batch = torch.cat([self.preprocess(w, sample_rate) for w in waveforms]).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = self.model(input_values=batch).logits
        probs = F.softmax(logits.float(), dim=-1)
        return [float(p[self._real_index].item()) for p in probs]
