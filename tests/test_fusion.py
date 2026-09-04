"""
Tests for the fusion risk scoring layer.
"""

import pytest

from backend.fusion.risk_scorer import RiskScorer, RiskResult


class TestRiskScorer:
    """Test the weighted fusion risk scorer."""

    def setup_method(self):
        self.scorer = RiskScorer()

    def test_all_safe_low_risk(self):
        """All signals safe → risk should be near 0."""
        result = self.scorer.compute(
            spoof_score=1.0,  # Definitely genuine
            speaker_score=1.0,  # Perfect speaker match
            prosody_score=1.0,  # Perfectly natural
        )
        assert result.risk_score < 5.0, f"Expected low risk, got {result.risk_score}"
        assert not result.alert

    def test_all_dangerous_high_risk(self):
        """All signals dangerous → risk should be near 100."""
        result = self.scorer.compute(
            spoof_score=0.0,  # Definitely spoofed
            speaker_score=0.0,  # No speaker match
            prosody_score=0.0,  # Unnatural prosody
        )
        assert result.risk_score > 95.0, f"Expected high risk, got {result.risk_score}"
        assert result.alert

    def test_spoof_only_high_risk(self):
        """
        Spoof detected while the speaker matches → still high risk.

        This is the defining case for clone detection: a successful voice clone
        matches its target speaker by construction, so speaker agreement must
        never be allowed to talk the spoof signal down. Asserted structurally
        rather than against a specific arithmetic result, so retuning the weights
        doesn't silently invalidate the intent.
        """
        result = self.scorer.compute(
            spoof_score=0.1,
            speaker_score=0.9,
            prosody_score=0.8,
        )
        contributions = {b.component: b.contribution for b in result.breakdown}

        assert result.risk_score > 50, "confident spoof evidence must produce high risk"
        assert contributions["spoof_detection"] == max(contributions.values()), (
            "spoof detection must dominate when it is confident"
        )

    def test_no_enrollment_neutral_speaker(self):
        """Speaker score 0.5 (neutral) should not dominate risk."""
        result = self.scorer.compute(
            spoof_score=0.9,
            speaker_score=0.5,  # Neutral — no enrolled speaker
            prosody_score=0.8,
        )
        # Should be relatively low risk since spoof and prosody are fine
        assert result.risk_score < 30

    def test_risk_score_range(self):
        """Risk score should always be in [0, 100]."""
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for sp in [0.0, 0.5, 1.0]:
                for pr in [0.0, 0.5, 1.0]:
                    result = self.scorer.compute(s, sp, pr)
                    assert 0.0 <= result.risk_score <= 100.0

    def test_breakdown_components(self):
        """Result should contain breakdown for all 3 components."""
        result = self.scorer.compute(0.5, 0.5, 0.5)
        breakdown_dict = result.to_dict()["breakdown"]
        assert "spoof_detection" in breakdown_dict
        assert "speaker_similarity" in breakdown_dict
        assert "prosody_naturalness" in breakdown_dict

    def test_breakdown_contributions_sum_to_risk(self):
        """Sum of contributions should equal total risk score."""
        result = self.scorer.compute(0.3, 0.6, 0.4)
        total_contribution = sum(b.contribution for b in result.breakdown)
        assert abs(total_contribution - result.risk_score) < 0.1

    def test_confidence_high_agreement(self):
        """All signals agreeing should produce high confidence."""
        result = self.scorer.compute(0.1, 0.1, 0.1)  # All say dangerous
        assert result.confidence == "high"

    def test_confidence_low_agreement(self):
        """Conflicting signals should produce low confidence."""
        result = self.scorer.compute(0.0, 1.0, 0.5)  # Mixed signals
        assert result.confidence in ("low", "medium")

    def test_alert_threshold(self):
        """Alert should fire only above threshold."""
        # Default threshold is 65
        low_risk = self.scorer.compute(0.8, 0.8, 0.8)
        assert not low_risk.alert

        high_risk = self.scorer.compute(0.1, 0.1, 0.1)
        assert high_risk.alert

    def test_custom_weights(self):
        """Custom weights should change risk score."""
        # All weight on spoof detection
        custom_scorer = RiskScorer(
            weights={
                "spoof_detection": 1.0,
                "speaker_similarity": 0.0,
                "prosody_naturalness": 0.0,
            }
        )
        result = custom_scorer.compute(0.0, 1.0, 1.0)
        assert result.risk_score > 90  # Only spoof matters

    def test_invalid_weights_rejected(self):
        """Weights not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RiskScorer(
                weights={
                    "spoof_detection": 0.5,
                    "speaker_similarity": 0.5,
                    "prosody_naturalness": 0.5,
                }
            )

    def test_to_dict_serialization(self):
        """Result should serialize to a clean dict."""
        result = self.scorer.compute(0.7, 0.6, 0.8)
        d = result.to_dict()
        assert isinstance(d["risk_score"], float)
        assert isinstance(d["alert"], bool)
        assert isinstance(d["confidence"], str)
        assert isinstance(d["breakdown"], dict)
