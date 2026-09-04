"""
Prosody Feature Extraction for synthetic speech detection.

Extracts acoustic/prosodic features that synthetic speech often gets wrong:
- Pitch contour variability (TTS tends to be unnaturally smooth)
- Jitter (micro-perturbations in pitch — too regular in TTS)
- Shimmer (micro-perturbations in amplitude — too regular in TTS)
- Harmonics-to-Noise Ratio (different noise profiles in synthetic speech)
- Pause ratio (unnatural pause patterns in cloned speech)

Uses Parselmouth (Python wrapper for Praat) for clinical-grade voice features
and librosa for general audio analysis.
"""

from typing import Optional

import numpy as np

from backend.config import config


class ProsodyAnalyzer:
    """
    Extracts prosody features and computes a naturalness anomaly score.
    
    The naturalness score is rule-based: each feature is checked against
    "normal human speech" reference ranges (from ASVspoof bonafide stats).
    Features falling outside normal ranges reduce the naturalness score.
    
    Usage:
        analyzer = ProsodyAnalyzer()
        result = analyzer.extract(waveform_numpy, sample_rate=16000)
        # result["prosody_score"] ∈ [0.0, 1.0] — 1.0 = sounds natural
    """

    def __init__(self):
        self.prosody_config = config.prosody
        self.target_sr = config.audio.sample_rate

    def extract(self, waveform: np.ndarray, sample_rate: int) -> dict:
        """
        Extract all prosody features from an audio waveform.
        
        Args:
            waveform: Audio samples as numpy array [samples,] (mono, float)
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary containing:
                - pitch_mean: Mean F0 (Hz)
                - pitch_std: Std dev of F0 (Hz) — low in TTS
                - jitter: Local jitter (fraction) — low in TTS
                - shimmer: Local shimmer (dB) — low in TTS
                - hnr: Harmonics-to-noise ratio (dB)
                - pause_ratio: Fraction of silence in the signal
                - speaking_rate: Estimated syllables per second
                - prosody_score: Aggregated naturalness score [0.0, 1.0]
                - anomalies: List of detected anomaly descriptions
        """
        # Ensure 1D float array
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        waveform = waveform.astype(np.float64)

        # Extract individual features
        pitch_features = self._extract_pitch(waveform, sample_rate)
        jitter_val = self._extract_jitter(waveform, sample_rate)
        shimmer_val = self._extract_shimmer(waveform, sample_rate)
        hnr_val = self._extract_hnr(waveform, sample_rate)
        pause_ratio = self._extract_pause_ratio(waveform, sample_rate)
        speaking_rate = self._extract_speaking_rate(waveform, sample_rate)

        # Compute anomaly-based naturalness score
        prosody_score, anomalies = self._compute_naturalness_score(
            pitch_std=pitch_features["pitch_std"],
            jitter=jitter_val,
            shimmer=shimmer_val,
            hnr=hnr_val,
            pause_ratio=pause_ratio,
        )

        return {
            "pitch_mean": pitch_features["pitch_mean"],
            "pitch_std": pitch_features["pitch_std"],
            "pitch_range": pitch_features["pitch_range"],
            "jitter": jitter_val,
            "shimmer": shimmer_val,
            "hnr": hnr_val,
            "pause_ratio": pause_ratio,
            "speaking_rate": speaking_rate,
            "prosody_score": prosody_score,
            "anomalies": anomalies,
        }

    def _extract_pitch(self, waveform: np.ndarray, sample_rate: int) -> dict:
        """Extract pitch (F0) statistics using Parselmouth/Praat."""
        try:
            import parselmouth

            sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
            pitch = sound.to_pitch(time_step=0.01)
            pitch_values = pitch.selected_array["frequency"]

            # Filter out unvoiced frames (0 Hz)
            voiced = pitch_values[pitch_values > 0]

            if len(voiced) < 3:
                return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0}

            return {
                "pitch_mean": float(np.mean(voiced)),
                "pitch_std": float(np.std(voiced)),
                "pitch_range": float(np.max(voiced) - np.min(voiced)),
            }
        except ImportError:
            return self._extract_pitch_fallback(waveform, sample_rate)

    def _extract_pitch_fallback(self, waveform: np.ndarray, sample_rate: int) -> dict:
        """Fallback pitch extraction using librosa's pYIN."""
        try:
            import librosa

            f0, voiced_flag, _ = librosa.pyin(
                waveform.astype(np.float32),
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

            if len(voiced) < 3:
                return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0}

            return {
                "pitch_mean": float(np.nanmean(voiced)),
                "pitch_std": float(np.nanstd(voiced)),
                "pitch_range": float(np.nanmax(voiced) - np.nanmin(voiced)),
            }
        except ImportError:
            return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_range": 0.0}

    def _extract_jitter(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Extract local jitter using Parselmouth/Praat."""
        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

            return float(jitter) if not np.isnan(jitter) else 0.0
        except (ImportError, Exception):
            return 0.0

    def _extract_shimmer(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Extract local shimmer using Parselmouth/Praat."""
        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            shimmer = call(
                [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )

            return float(shimmer) if not np.isnan(shimmer) else 0.0
        except (ImportError, Exception):
            return 0.0

    def _extract_hnr(self, waveform: np.ndarray, sample_rate: int) -> float:
        """Extract Harmonics-to-Noise Ratio using Parselmouth/Praat."""
        try:
            import parselmouth
            from parselmouth.praat import call

            sound = parselmouth.Sound(waveform, sampling_frequency=sample_rate)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)

            return float(hnr) if not np.isnan(hnr) else 0.0
        except (ImportError, Exception):
            return 0.0

    def _extract_pause_ratio(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Estimate pause ratio (fraction of silence) using energy-based VAD.
        
        Uses RMS energy with a threshold to detect voiced vs silent frames.
        """
        try:
            import librosa

            # Compute RMS energy in short frames
            rms = librosa.feature.rms(
                y=waveform.astype(np.float32),
                frame_length=int(0.025 * sample_rate),  # 25ms frames
                hop_length=int(0.010 * sample_rate),  # 10ms hop
            )[0]

            if len(rms) == 0:
                return 0.0

            # Dynamic threshold: 10% of mean RMS
            threshold = 0.1 * np.mean(rms)
            silent_frames = np.sum(rms < threshold)
            pause_ratio = silent_frames / len(rms)

            return float(np.clip(pause_ratio, 0.0, 1.0))
        except ImportError:
            # Simple fallback: amplitude threshold
            threshold = 0.01 * np.max(np.abs(waveform))
            silent_samples = np.sum(np.abs(waveform) < threshold)
            return float(silent_samples / len(waveform))

    def _extract_speaking_rate(self, waveform: np.ndarray, sample_rate: int) -> float:
        """
        Estimate speaking rate using onset detection (approximate syllables/sec).
        """
        try:
            import librosa

            onset_env = librosa.onset.onset_strength(
                y=waveform.astype(np.float32), sr=sample_rate
            )
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env, sr=sample_rate, units="time"
            )

            if len(onsets) < 2:
                return 0.0

            duration = len(waveform) / sample_rate
            return float(len(onsets) / duration)
        except ImportError:
            return 0.0

    def _compute_naturalness_score(
        self,
        pitch_std: float,
        jitter: float,
        shimmer: float,
        hnr: float,
        pause_ratio: float,
    ) -> tuple[float, list[str]]:
        """
        Compute aggregated naturalness score from prosody features.
        
        Each feature contributes equally. A feature scores 1.0 if within
        the normal range, and degrades toward 0.0 as it deviates.
        
        Returns:
            (score, anomalies) — score ∈ [0.0, 1.0], anomalies = list of strings
        """
        scores = []
        anomalies = []
        cfg = self.prosody_config

        # Pitch variability
        score_pitch = self._range_score(pitch_std, cfg.pitch_std_range[0], cfg.pitch_std_range[1])
        scores.append(score_pitch)
        if score_pitch < 0.5:
            anomalies.append(
                f"Pitch variability ({pitch_std:.1f} Hz) outside normal range "
                f"[{cfg.pitch_std_range[0]}, {cfg.pitch_std_range[1]}]"
            )

        # Jitter
        score_jitter = self._range_score(jitter, cfg.jitter_range[0], cfg.jitter_range[1])
        scores.append(score_jitter)
        if score_jitter < 0.5:
            anomalies.append(
                f"Jitter ({jitter:.5f}) outside normal range "
                f"[{cfg.jitter_range[0]}, {cfg.jitter_range[1]}]"
            )

        # Shimmer
        score_shimmer = self._range_score(shimmer, cfg.shimmer_range[0], cfg.shimmer_range[1])
        scores.append(score_shimmer)
        if score_shimmer < 0.5:
            anomalies.append(
                f"Shimmer ({shimmer:.4f}) outside normal range "
                f"[{cfg.shimmer_range[0]}, {cfg.shimmer_range[1]}]"
            )

        # HNR
        score_hnr = self._range_score(hnr, cfg.hnr_range[0], cfg.hnr_range[1])
        scores.append(score_hnr)
        if score_hnr < 0.5:
            anomalies.append(
                f"HNR ({hnr:.1f} dB) outside normal range "
                f"[{cfg.hnr_range[0]}, {cfg.hnr_range[1]}]"
            )

        # Pause ratio
        score_pause = self._range_score(
            pause_ratio, cfg.pause_ratio_range[0], cfg.pause_ratio_range[1]
        )
        scores.append(score_pause)
        if score_pause < 0.5:
            anomalies.append(
                f"Pause ratio ({pause_ratio:.2f}) outside normal range "
                f"[{cfg.pause_ratio_range[0]}, {cfg.pause_ratio_range[1]}]"
            )

        # Average all feature scores
        naturalness_score = float(np.mean(scores)) if scores else 0.5
        return naturalness_score, anomalies

    @staticmethod
    def _range_score(value: float, low: float, high: float) -> float:
        """
        Score a value based on how well it falls within [low, high].
        
        Returns:
            1.0 if within range
            Decays toward 0.0 as distance from range increases
        """
        if value == 0.0:
            # Missing feature — return neutral
            return 0.5

        if low <= value <= high:
            return 1.0

        # Distance from nearest boundary, normalized by range width
        range_width = max(high - low, 1e-6)
        if value < low:
            distance = (low - value) / range_width
        else:
            distance = (value - high) / range_width

        # Exponential decay: score = exp(-2 * distance)
        return float(np.exp(-2.0 * distance))
