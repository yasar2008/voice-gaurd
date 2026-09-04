"""
Fusion Layer: Multi-signal risk scoring.

Combines three independent detection signals into a single explainable risk score:
1. AASIST-L spoof detection (is the audio synthetic?)
2. ECAPA-TDNN speaker similarity (does it match the enrolled speaker?)
3. Prosody naturalness (do the acoustic features look human?)

Design decision — why NOT a learned fusion model:
    Explainability. Portfolio reviewers (and real users) should see exactly
    why a risk score is high: "spoof detection contributed 42/78, speaker
    mismatch contributed 24/78, prosody anomaly contributed 12/78."
    A logistic regression or MLP here would be a black box.
"""

from dataclasses import dataclass, field

from backend.config import config


@dataclass
class RiskBreakdown:
    """Per-component contribution to the overall risk score."""

    component: str
    raw_score: float  # Original signal value [0, 1]
    weight: float  # Fusion weight
    contribution: float  # Weighted contribution to risk score (0–100 scale)
    interpretation: str  # Human-readable explanation


@dataclass
class RiskResult:
    """Complete risk assessment result."""

    risk_score: float  # Overall risk score [0, 100]
    alert: bool  # Whether score exceeds alert threshold
    confidence: str  # "low", "medium", or "high"
    breakdown: list[RiskBreakdown] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)  # From prosody analysis

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for API responses."""
        return {
            "risk_score": round(self.risk_score, 1),
            "alert": self.alert,
            "confidence": self.confidence,
            "breakdown": {
                b.component: {
                    "raw_score": round(b.raw_score, 4),
                    "weight": b.weight,
                    "contribution": round(b.contribution, 1),
                    "interpretation": b.interpretation,
                }
                for b in self.breakdown
            },
            "anomalies": self.anomalies,
        }


class RiskScorer:
    """
    Weighted fusion of detection signals into a 0–100 risk score.
    
    Signal semantics (important — each signal has different polarity):
        - spoof_score: AASIST bonafide confidence [0,1]. HIGH = safe. Must INVERT for risk.
        - speaker_score: cosine similarity [0,1]. HIGH = matches enrolled speaker = safe. INVERT.
        - prosody_score: naturalness [0,1]. HIGH = sounds human = safe. INVERT.
    
    So: risk = weighted_sum(1 - signal_i) * 100
    
    Usage:
        scorer = RiskScorer()
        result = scorer.compute(spoof_score=0.2, speaker_score=0.8, prosody_score=0.6)
        print(result.risk_score)  # e.g., 52.0
        print(result.alert)       # True if above threshold
    """

    def __init__(self, weights: dict[str, float] | None = None):
        fusion_config = config.fusion
        self.weights = weights or fusion_config.weights
        self.alert_threshold = fusion_config.alert_threshold
        self.high_agreement = fusion_config.high_confidence_agreement
        self.low_agreement = fusion_config.low_confidence_agreement

        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total:.3f}: {self.weights}"
            )

    def compute(
        self,
        spoof_score: float,
        speaker_score: float | None,
        prosody_score: float,
    ) -> RiskResult:
        """
        Compute the fused risk score from three independent signals.

        Args:
            spoof_score: AASIST bonafide confidence [0, 1]. 1.0 = genuine.
            speaker_score: Speaker cosine similarity [0, 1]. 1.0 = matches enrolled.
                Pass None when no speaker is enrolled — the signal is then dropped
                and the remaining weights are renormalised. Passing 0.5 as a
                "neutral" value instead would add a constant 15 points of unearned
                risk to every result, which both lifts the floor under genuine
                audio and caps the ceiling: 0.50 + 0.30 × 0.5 = 0.65, exactly the
                default alert threshold, so a certain spoof could never alert.
            prosody_score: Prosody naturalness [0, 1]. 1.0 = sounds human.

        Returns:
            RiskResult with score, alert status, confidence, and per-component breakdown.
        """
        speaker_available = speaker_score is not None

        # Invert signals: high safety → low risk
        risk_spoof = 1.0 - spoof_score
        risk_speaker = (1.0 - speaker_score) if speaker_available else 0.0
        risk_prosody = 1.0 - prosody_score

        # Weighted combination → raw risk [0, 1]. When the speaker signal is
        # absent its weight is redistributed proportionally over the rest.
        w = dict(self.weights)
        if not speaker_available:
            remaining = w["spoof_detection"] + w["prosody_naturalness"]
            if remaining > 0:
                scale = 1.0 / remaining
                w["spoof_detection"] *= scale
                w["prosody_naturalness"] *= scale
            w["speaker_similarity"] = 0.0

        raw_risk = (
            w["spoof_detection"] * risk_spoof
            + w["speaker_similarity"] * risk_speaker
            + w["prosody_naturalness"] * risk_prosody
        )

        # Scale to [0, 100]
        risk_score = raw_risk * 100.0
        risk_score = max(0.0, min(100.0, risk_score))

        # Alert check
        alert = risk_score >= self.alert_threshold

        # Per-component breakdown
        breakdown = [
            RiskBreakdown(
                component="spoof_detection",
                raw_score=spoof_score,
                weight=w["spoof_detection"],
                contribution=w["spoof_detection"] * risk_spoof * 100.0,
                interpretation=self._interpret_spoof(spoof_score),
            ),
            RiskBreakdown(
                component="speaker_similarity",
                raw_score=speaker_score if speaker_available else 0.0,
                weight=w["speaker_similarity"],
                contribution=w["speaker_similarity"] * risk_speaker * 100.0,
                interpretation=(
                    self._interpret_speaker(speaker_score)
                    if speaker_available
                    else "No enrolled speaker - signal excluded, weights renormalised"
                ),
            ),
            RiskBreakdown(
                component="prosody_naturalness",
                raw_score=prosody_score,
                weight=w["prosody_naturalness"],
                contribution=w["prosody_naturalness"] * risk_prosody * 100.0,
                interpretation=self._interpret_prosody(prosody_score),
            ),
        ]

        # Confidence based on signal agreement
        confidence = (
            self._compute_confidence(risk_spoof, risk_speaker, risk_prosody)
            if speaker_available
            else self._compute_confidence(risk_spoof, risk_prosody)
        )

        return RiskResult(
            risk_score=risk_score,
            alert=alert,
            confidence=confidence,
            breakdown=breakdown,
        )

    def _compute_confidence(self, *risks: float) -> str:
        """
        Compute confidence level based on agreement between signals.
        
        If all signals agree (all high risk or all low risk), confidence is high.
        Mixed signals → lower confidence.
        """
        signals = list(risks)
        mean_risk = sum(signals) / len(signals)

        # Compute agreement: how close is each signal to the mean?
        deviations = [abs(s - mean_risk) for s in signals]
        avg_deviation = sum(deviations) / len(deviations)

        # Agreement = 1 - normalized deviation
        agreement = 1.0 - min(avg_deviation / 0.5, 1.0)

        if agreement >= self.high_agreement:
            return "high"
        elif agreement >= self.low_agreement:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _interpret_spoof(score: float) -> str:
        """Human-readable interpretation of spoof detection score."""
        if score >= 0.8:
            return "Audio appears genuine (high bonafide confidence)"
        elif score >= 0.5:
            return "Audio has some synthetic characteristics"
        elif score >= 0.2:
            return "Audio likely contains synthetic speech"
        else:
            return "Audio is very likely synthetic or spoofed"

    @staticmethod
    def _interpret_speaker(score: float) -> str:
        """Human-readable interpretation of speaker similarity."""
        if score == 0.5:
            return "No enrolled speaker — cannot verify identity"
        elif score >= 0.8:
            return "Strong match with enrolled speaker"
        elif score >= 0.5:
            return "Moderate match with enrolled speaker"
        elif score >= 0.3:
            return "Weak match — possibly different speaker"
        else:
            return "Speaker does not match enrollment"

    @staticmethod
    def _interpret_prosody(score: float) -> str:
        """Human-readable interpretation of prosody naturalness."""
        if score >= 0.8:
            return "Prosody sounds natural and human-like"
        elif score >= 0.5:
            return "Some prosodic features appear unusual"
        elif score >= 0.2:
            return "Multiple prosodic anomalies detected"
        else:
            return "Prosody is highly unnatural — likely synthetic"
