"""
AASIST-L: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (Lightweight).

This module wraps the AASIST-L architecture for spoof/synthesis detection.
The model operates on raw 16kHz waveforms and outputs a bonafide confidence score.

Architecture Reference:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks", ICASSP 2022.
    GitHub: https://github.com/clovaai/aasist

The AASIST-L variant has ~85k parameters — small enough for real-time CPU inference.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from backend.config import config
from backend.models.aasist_pretrained import build_aasist_l


# ---------------------------------------------------------------------------
# Sinc convolution layer (from RawNet2 / SincNet, used as AASIST front-end)
# ---------------------------------------------------------------------------

class SincConv(nn.Module):
    """Parameterized sinc-function-based bandpass filter bank.
    
    Learns the low and high cutoff frequencies of each filter,
    operating directly on raw waveforms.
    """

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 128,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialize filterbank with mel-spaced frequencies
        low_hz = min_low_hz
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel_low = 2595 * math.log10(1 + low_hz / 700)
        mel_high = 2595 * math.log10(1 + high_hz / 700)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)

        self.low_hz_ = nn.Parameter(hz_points[:-1].unsqueeze(1))
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).unsqueeze(1))

        # Hamming window (fixed)
        n = (kernel_size - 1) / 2.0
        self.register_buffer(
            "n_", (2 * math.pi * torch.arange(-n, 0).unsqueeze(0) / sample_rate)
        )
        self.register_buffer("window_", torch.hamming_window(kernel_size // 2, periodic=False))

    def _sinc(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized sinc function."""
        sinc_x = torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)
        return sinc_x

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), max=self.sample_rate / 2)
        
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate

        # Compute bandpass filters
        band_pass_left = (
            (self._sinc(f_high * self.n_) - self._sinc(f_low * self.n_)) * self.window_
        )
        band_pass_center = 2 * (f_high - f_low)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band_pass_center)

        filters = band_pass.unsqueeze(1)
        return F.conv1d(waveform, filters, stride=1, padding=self.kernel_size // 2)


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer for spectro-temporal modeling."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: Node features [batch, num_nodes, in_dim]
            adj: Adjacency matrix [batch, num_nodes, num_nodes] or None (fully connected)
        Returns:
            Updated node features [batch, num_nodes, out_dim]
        """
        z = self.fc(h)  # [B, N, out_dim]
        B, N, D = z.shape

        # Compute attention coefficients
        a_input_i = z.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        a_input_j = z.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        e = self.leaky_relu(self.attn_fc(torch.cat([a_input_i, a_input_j], dim=-1)).squeeze(-1))

        if adj is not None:
            e = e.masked_fill(adj == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        h_prime = torch.bmm(alpha, z)  # [B, N, out_dim]
        return F.elu(h_prime)


# ---------------------------------------------------------------------------
# AASIST-L Model
# ---------------------------------------------------------------------------

class AASISTModel(nn.Module):
    """
    AASIST-L (Lightweight) model architecture.
    
    Simplified version of the full AASIST with reduced channel dimensions
    to achieve ~85k parameters while maintaining strong ASVspoof performance.
    
    Pipeline:
        Raw waveform → SincConv → Encoder blocks → Graph Attention → Readout → Score
    """

    def __init__(
        self,
        sinc_channels: int = 70,
        sinc_kernel: int = 128,
        encoder_channels: list[int] = None,
        gat_hidden: int = 32,
        gat_out: int = 16,
        num_classes: int = 2,
    ):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [128, 64, 32]  # Lightweight channel progression

        # Front-end: learnable sinc filters on raw waveform
        self.sinc_conv = SincConv(
            out_channels=sinc_channels,
            kernel_size=sinc_kernel,
            sample_rate=config.audio.sample_rate,
        )
        self.sinc_bn = nn.BatchNorm1d(sinc_channels)

        # Encoder: series of conv blocks with residual connections
        self.encoder = nn.ModuleList()
        in_ch = sinc_channels
        for out_ch in encoder_channels:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.SiLU(),
                    nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.SiLU(),
                    nn.MaxPool1d(kernel_size=4),
                )
            )
            in_ch = out_ch

        # Adaptive pooling to create graph nodes
        self.num_nodes = 8
        self.node_proj = nn.Linear(in_ch, gat_hidden)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.num_nodes)

        # Graph attention layers for spectro-temporal reasoning
        self.gat1 = GraphAttentionLayer(gat_hidden, gat_hidden)
        self.gat2 = GraphAttentionLayer(gat_hidden, gat_out)

        # Readout: aggregate graph node features → classification
        self.readout = nn.Sequential(
            nn.Linear(gat_out * self.num_nodes, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform tensor [batch, 1, samples] or [batch, samples]
        Returns:
            Logits [batch, 2] — index 0: spoof, index 1: bonafide
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        # Sinc front-end
        x = self.sinc_conv(x)  # [B, sinc_channels, T]
        x = self.sinc_bn(x)
        x = F.silu(x)

        # Encoder
        for block in self.encoder:
            x = block(x)  # [B, C, T'] — T shrinks via MaxPool

        # Create graph nodes via adaptive pooling
        x = self.adaptive_pool(x)  # [B, C, num_nodes]
        x = x.permute(0, 2, 1)  # [B, num_nodes, C]
        x = self.node_proj(x)  # [B, num_nodes, gat_hidden]

        # Graph attention
        x = self.gat1(x)  # [B, num_nodes, gat_hidden]
        x = self.gat2(x)  # [B, num_nodes, gat_out]

        # Readout
        x = x.reshape(x.size(0), -1)  # [B, num_nodes * gat_out]
        logits = self.readout(x)  # [B, 2]
        return logits


# ---------------------------------------------------------------------------
# High-level detector wrapper
# ---------------------------------------------------------------------------

class AASISTDetector:
    """
    High-level wrapper for AASIST-L inference.

    Two execution modes, and the difference matters:

    * **calibrated** — `checkpoints/AASIST-L.pth` is present, so the official
      AASIST-L network (`backend/models/aasist_pretrained.py`) runs with weights
      trained on ASVspoof 2019 LA. Scores are meaningful.
    * **uncalibrated** — no usable checkpoint. The local :class:`AASISTModel`
      runs with *random* weights so the rest of the pipeline still works, but its
      output is noise. `is_calibrated` is False and every caller is expected to
      surface that rather than present the score as a real verdict.

    Run ``python scripts/download_checkpoints.py`` to get the checkpoint.

    Usage:
        detector = AASISTDetector("checkpoints/AASIST-L.pth")
        score = detector.predict(waveform, sample_rate=44100)
        # score ∈ [0.0, 1.0] — higher = more likely bonafide
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or config.device)
        self.target_sr = config.audio.sample_rate
        self.input_length = config.model.aasist_input_length

        self.is_calibrated = False
        self.backend = "uncalibrated"
        self.checkpoint_path = checkpoint_path

        self.model: nn.Module | None = None
        if checkpoint_path:
            self._load_pretrained(checkpoint_path)

        if self.model is None:
            # Fallback so the API still starts on a fresh clone — but the caller
            # can (and should) tell the user the detector is not calibrated.
            self.model = AASISTModel()
            print(
                "[AASISTDetector] Running UNCALIBRATED with randomly initialized "
                "weights - spoof scores are meaningless. Fix with: "
                "python scripts/download_checkpoints.py"
            )

        self.model.to(self.device)
        self.model.eval()

    # -- loading ------------------------------------------------------------

    def _load_pretrained(self, path: str) -> None:
        """
        Load the official AASIST-L checkpoint.

        Loads strictly: a checkpoint that does not match the architecture is a
        hard failure, not a silently half-initialised model.
        """
        import os

        if not os.path.exists(path):
            print(
                f"[AASISTDetector] Checkpoint not found at {path}. "
                f"Run: python scripts/download_checkpoints.py"
            )
            return

        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)

            # Handle checkpoints wrapped in {'model': state_dict, ...}
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model = build_aasist_l()
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:  # noqa: BLE001 — any failure means "not calibrated"
            print(
                f"[AASISTDetector] Could not load checkpoint {path}: {e}. "
                f"Falling back to uncalibrated weights."
            )
            return

        self.model = model
        self.is_calibrated = True
        self.backend = "aasist-l-pretrained"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[AASISTDetector] Loaded AASIST-L checkpoint ({n_params:,} params) from {path}")

    @property
    def status(self) -> dict:
        """Machine-readable state for the /health endpoint."""
        import os

        return {
            "backend": self.backend,
            "calibrated": self.is_calibrated,
            # Name only — the absolute path is server-side detail.
            "checkpoint": os.path.basename(self.checkpoint_path) if self.checkpoint_path else None,
        }

    # -- preprocessing ------------------------------------------------------

    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio for AASIST-L inference.

        Steps:
            1. Convert to mono (if stereo)
            2. Resample to 16kHz
            3. RMS-normalise toward the level the model was trained on, unless the
               window is essentially silent. AASIST is strongly level-sensitive,
               so skipping this lets microphone gain decide the verdict.
            4. Apply pre-emphasis — only for the uncalibrated local model; the
               pretrained network was trained on raw waveforms, so applying it
               there would shift the input off its training distribution.
            5. Pad (by repeating the signal, as in the reference implementation)
               or truncate to exactly 64,600 samples

        Args:
            waveform: Audio tensor [channels, samples] or [samples]
            sample_rate: Original sample rate

        Returns:
            Preprocessed tensor [1, input_length] on target device
        """
        # Ensure 2D: [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)

        # Level normalisation — before anything else touches the samples.
        if config.audio.normalize_input:
            rms = waveform.pow(2).mean().sqrt()
            # Leave near-silence alone: scaling a noise floor up to speech level
            # would manufacture a signal out of nothing.
            if rms > config.audio.min_rms_for_normalize:
                waveform = waveform * (config.audio.target_rms / rms)
                peak = waveform.abs().max()
                ceiling = config.audio.normalize_peak_ceiling
                if peak > ceiling:
                    waveform = waveform * (ceiling / peak)

        # Pre-emphasis: y[n] = x[n] - α * x[n-1]
        if not self.is_calibrated:
            coeff = config.audio.pre_emphasis_coeff
            waveform = torch.cat(
                [waveform[:, :1], waveform[:, 1:] - coeff * waveform[:, :-1]], dim=1
            )

        # Pad or truncate to fixed length
        current_length = waveform.shape[1]
        if current_length == 0:
            waveform = torch.zeros(1, self.input_length)
        elif current_length < self.input_length:
            # Repeat-pad, matching the reference AASIST data pipeline
            repeats = self.input_length // current_length + 1
            waveform = waveform.repeat(1, repeats)[:, : self.input_length]
        elif current_length > self.input_length:
            # Truncate (take from center for better representation)
            start = (current_length - self.input_length) // 2
            waveform = waveform[:, start : start + self.input_length]

        return waveform.to(self.device)

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _logits(output) -> torch.Tensor:
        """The pretrained network returns (last_hidden, logits); ours returns logits."""
        return output[1] if isinstance(output, tuple) else output

    @torch.no_grad()
    def predict(self, waveform: torch.Tensor, sample_rate: int) -> float:
        """
        Run spoof detection on an audio waveform.

        Args:
            waveform: Audio tensor [channels, samples] or [samples]
            sample_rate: Sample rate of the input audio

        Returns:
            Bonafide confidence score ∈ [0.0, 1.0]
            - 1.0 = high confidence the audio is genuine human speech
            - 0.0 = high confidence the audio is synthetic/spoofed

            Meaningless unless `is_calibrated` is True.
        """
        x = self.preprocess(waveform, sample_rate)  # [1, T]
        if not self.is_calibrated:
            x = x.unsqueeze(0)  # local model expects [B, 1, T]

        logits = self._logits(self.model(x))  # [1, 2]
        probs = F.softmax(logits, dim=-1)

        # Index 1 = bonafide probability
        bonafide_score = probs[0, 1].item()
        return bonafide_score

    @torch.no_grad()
    def predict_batch(self, waveforms: list[torch.Tensor], sample_rate: int) -> list[float]:
        """
        Run spoof detection on a batch of waveforms.

        Args:
            waveforms: List of audio tensors
            sample_rate: Common sample rate

        Returns:
            List of bonafide confidence scores
        """
        batch = torch.stack([self.preprocess(w, sample_rate).squeeze(0) for w in waveforms])
        if not self.is_calibrated:
            batch = batch.unsqueeze(1)  # [B, 1, T]

        logits = self._logits(self.model(batch))
        probs = F.softmax(logits, dim=-1)

        return [p[1].item() for p in probs]
