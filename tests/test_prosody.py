"""
Tests for Prosody Feature Extraction.
"""

import numpy as np
import pytest

from backend.features.prosody import ProsodyAnalyzer


class TestProsodyAnalyzer:
    """Test prosody feature extraction and naturalness scoring."""

    def setup_method(self):
        self.analyzer = ProsodyAnalyzer()

    def test_extract_synthetic_sine_wave(self):
        """Pure sine wave (unnatural, zero jitter/shimmer) should return low/anomaly score."""
        sr = 16000
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        # 220Hz pure tone
        waveform = 0.5 * np.sin(2 * np.pi * 220 * t)

        result = self.analyzer.extract(waveform, sample_rate=sr)

        assert "pitch_mean" in result
        assert "jitter" in result
        assert "shimmer" in result
        assert "hnr" in result
        assert "pause_ratio" in result
        assert "prosody_score" in result
        assert "anomalies" in result
        assert 0.0 <= result["prosody_score"] <= 1.0

    def test_extract_silence(self):
        """Near-silence should handle gracefully without crashing."""
        sr = 16000
        waveform = np.zeros(sr * 2, dtype=np.float32)

        result = self.analyzer.extract(waveform, sample_rate=sr)
        assert 0.0 <= result["prosody_score"] <= 1.0
        assert result["pitch_mean"] == 0.0

    def test_range_score_boundaries(self):
        """Test _range_score helper for within/outside reference bounds."""
        # Inside bounds [10, 50]
        assert ProsodyAnalyzer._range_score(25.0, 10.0, 50.0) == 1.0
        assert ProsodyAnalyzer._range_score(10.0, 10.0, 50.0) == 1.0
        assert ProsodyAnalyzer._range_score(50.0, 10.0, 50.0) == 1.0

        # Outside bounds
        score_low = ProsodyAnalyzer._range_score(5.0, 10.0, 50.0)
        score_high = ProsodyAnalyzer._range_score(100.0, 10.0, 50.0)
        assert 0.0 <= score_low < 1.0
        assert 0.0 <= score_high < 1.0
