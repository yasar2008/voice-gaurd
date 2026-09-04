"""
Check whether the spoof detector's answer depends on recording level.

It should not. AASIST-L was trained on ASVspoof 2019 LA, which sits around
RMS 0.1; a laptop microphone with AGC disabled is routinely 10-20x quieter. If
the score moves when only the gain changes, the model is being asked to judge
audio outside its training distribution and the verdict is about loudness rather
than authenticity.

    python scripts/level_sensitivity.py data/eval/spoof_tts/tts_zira_r+0_01.wav

Compare with normalisation off to see what it is protecting you from:

    set VCD_AUDIO__NORMALIZE_INPUT=false && python scripts/level_sensitivity.py <file>
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402
from backend.models.aasist import AASISTDetector  # noqa: E402

GAINS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="clip to sweep")
    args = parser.parse_args()

    detector = AASISTDetector(str(config.model.aasist_checkpoint), device=config.device)
    waveform, sr = load_audio(str(args.audio))
    mono = waveform.mean(dim=0)
    segment = mono[: config.model.aasist_input_length]

    peak = float(segment.abs().max())
    rms = float(segment.pow(2).mean().sqrt())
    print(f"\n{args.audio.name}: peak={peak:.3f} rms={rms:.4f}")
    print(f"normalisation: {'ON' if config.audio.normalize_input else 'OFF'}"
          f" (target rms {config.audio.target_rms})\n")

    print(f"{'gain':>7}{'peak':>9}{'rms':>9}{'bonafide':>12}")
    scores = []
    for gain in GAINS:
        scaled = segment * gain
        score = detector.predict(scaled.unsqueeze(0), sr)
        scores.append(score)
        print(
            f"{gain:7.2f}{float(scaled.abs().max()):9.3f}"
            f"{float(scaled.pow(2).mean().sqrt()):9.4f}{score:12.6f}"
        )

    spread = max(scores) - min(scores)
    verdicts = {"spoof" if s < 0.5 else "bonafide" for s in scores}
    print(f"\nscore spread across {len(GAINS)} gain settings: {spread:.4f}")
    if len(verdicts) > 1:
        print("VERDICT FLIPS with gain alone - the score is measuring loudness.")
        return 1
    print(f"verdict stable ({verdicts.pop()}) at every level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
