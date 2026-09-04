"""
Shared result shape for streaming consumers (Meet integration, monitoring UI).

One builder used by both the WebSocket handler and the REST endpoints, so a
client cannot get different field semantics depending on which door it came
through.

The bands reuse the thresholds the app already had — `alert_threshold` (65) and
`suspicious_threshold` (35, previously hardcoded in the frontend's verdict.ts).
Nothing is re-tuned here: the same audio produces the same verdict it did
before, this only names it.
"""

from backend.config import config


def risk_band(risk_score: float) -> str:
    """low / medium / high, using the configured thresholds."""
    fusion = config.fusion
    if risk_score >= fusion.alert_threshold:
        return "high"
    if risk_score >= fusion.suspicious_threshold:
        return "medium"
    return "low"


def detection_summary(
    risk_score: float,
    alert: bool,
    genuine_probability: float,
    *,
    reliable: bool = True,
) -> dict:
    """
    Flat, client-friendly summary of one analysis window.

    Args:
        risk_score: fused 0-100 risk.
        alert: whether the fused score crossed the alert threshold.
        genuine_probability: the spoof model's P(genuine) in [0, 1]. This is the
            model's own output, not the fused score — a caller that wants the
            detector's raw opinion should read this.
        reliable: False when the input quality gate declined to judge. The label
            is then "unknown" rather than a guess, because a confident verdict on
            audio the detector cannot handle is the failure mode this project
            spent the most effort removing.

    Returns:
        label ("genuine" | "synthetic" | "unknown"), both probabilities, risk
        band, and the numeric score.
    """
    genuine = max(0.0, min(1.0, float(genuine_probability)))
    if not reliable:
        label = "unknown"
    else:
        label = "synthetic" if alert else "genuine"

    return {
        "label": label,
        "genuine_probability": round(genuine, 4),
        "synthetic_probability": round(1.0 - genuine, 4),
        "risk": risk_band(risk_score) if reliable else "unknown",
        "risk_score": round(float(risk_score), 1),
    }
