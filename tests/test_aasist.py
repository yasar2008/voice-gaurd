"""
Tests for the AASIST-L spoof detection model.
"""

import torch
import pytest

from backend.models.aasist import AASISTModel, AASISTDetector


class TestAASISTModel:
    """Test the raw AASIST-L model architecture."""

    def test_model_forward_shape(self):
        """Model output should be [batch, 2] logits."""
        model = AASISTModel()
        x = torch.randn(2, 1, 64600)  # batch=2, mono, 64600 samples
        out = model(x)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_model_forward_2d_input(self):
        """Model should handle [batch, samples] input (no channel dim)."""
        model = AASISTModel()
        x = torch.randn(1, 64600)
        out = model(x)
        assert out.shape == (1, 2)

    def test_model_parameter_count(self):
        """AASIST-L should be lightweight (< 500k params)."""
        model = AASISTModel()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 500_000, f"Too many parameters: {total_params}"
        print(f"AASIST-L parameter count: {total_params:,}")


class TestAASISTDetector:
    """Test the high-level detector wrapper."""

    def setup_method(self):
        """Create detector with random weights (no checkpoint)."""
        self.detector = AASISTDetector(checkpoint_path=None)

    def test_predict_returns_float(self):
        """Predict should return a float score."""
        waveform = torch.randn(1, 16000 * 4)  # 4 seconds at 16kHz
        score = self.detector.predict(waveform, sample_rate=16000)
        assert isinstance(score, float)

    def test_predict_score_range(self):
        """Score should be in [0.0, 1.0]."""
        waveform = torch.randn(1, 16000 * 4)
        score = self.detector.predict(waveform, sample_rate=16000)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_preprocess_resamples(self):
        """Preprocessor should handle different sample rates."""
        # Input at 44100 Hz
        waveform = torch.randn(1, 44100 * 4)
        processed = self.detector.preprocess(waveform, sample_rate=44100)
        assert processed.shape[1] == 64600, f"Expected 64600 samples, got {processed.shape[1]}"

    def test_preprocess_pads_short_audio(self):
        """Short audio should be zero-padded."""
        waveform = torch.randn(1, 8000)  # 0.5 seconds
        processed = self.detector.preprocess(waveform, sample_rate=16000)
        assert processed.shape[1] == 64600

    def test_preprocess_truncates_long_audio(self):
        """Long audio should be truncated (center crop)."""
        waveform = torch.randn(1, 16000 * 10)  # 10 seconds
        processed = self.detector.preprocess(waveform, sample_rate=16000)
        assert processed.shape[1] == 64600

    def test_preprocess_mono_conversion(self):
        """Stereo input should be converted to mono."""
        waveform = torch.randn(2, 16000 * 4)  # Stereo
        processed = self.detector.preprocess(waveform, sample_rate=16000)
        assert processed.shape[0] == 1

    def test_predict_batch(self):
        """Batch prediction should return a list of scores."""
        waveforms = [torch.randn(1, 16000 * 4) for _ in range(3)]
        scores = self.detector.predict_batch(waveforms, sample_rate=16000)
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
