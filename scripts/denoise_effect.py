"""
Measure what noise reduction does to the spoof detector.

Short answer: it destroys it. Denoising genuine human speech flips the detector
from "certainly real" to "certainly fake".

    genuine clip      original   nr=12   nr=24   nr=40
    iemo_anger           1.000   1.000   0.000   0.000
    cl_it                1.000   0.995   0.000   0.000
    cv_en                0.940   0.000   0.000   0.000
    iemo_hap             1.000   1.000   0.000   0.000
    iemo_neutral         1.000   1.000   1.000   0.001

Four of five clips collapse at moderate strength, all five at high strength.
This is not a quirk of one file: spectral subtraction rewrites the fine spectral
structure the model uses to judge authenticity, and the model reads that
processing as synthesis. Audacity's Noise Reduction is the same family of
algorithm, so audio cleaned up in Audacity will read as fake.

    Never denoise, enhance, or "clean up" audio before analysing it.
    Feed the detector the rawest recording you have.

ffmpeg's afftdn is used here as the stand-in for Audacity's noise reduction —
both are FFT spectral subtraction.

    python scripts/denoise_effect.py
    python scripts/denoise_effect.py --clips data/eval/control_real
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.audio_io import load_audio  # noqa: E402

STRENGTHS = [12, 24, 40]


def denoise(src: Path, strength: int) -> str:
    """Apply FFT spectral-subtraction denoising, returning a temp wav path."""
    import imageio_ffmpeg

    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(src),
            "-af", f"afftdn=nr={strength}:nf=-25", "-ar", "16000", "-ac", "1", out,
        ],
        check=True,
        capture_output=True,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=PROJECT_ROOT / "data/eval/control_real")
    args = parser.parse_args()

    from backend.models.wav2vec_spoof import Wav2VecSpoofDetector

    detector = Wav2VecSpoofDetector()
    if not detector.is_calibrated:
        print("Spoof detector is not loaded. Run scripts/download_checkpoints.py")
        return 1

    clips = sorted(p for p in args.clips.glob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"})
    if not clips:
        raise SystemExit(f"No audio in {args.clips}")

    print("\nEffect of noise reduction on GENUINE speech (1.000 = model says real)\n")
    header = f"{'clip':<26}{'original':>10}" + "".join(f"{'nr=' + str(s):>9}" for s in STRENGTHS)
    print(header)
    print("-" * len(header))

    broke = 0
    for clip in clips:
        waveform, sr = load_audio(str(clip))
        scores = [detector.predict(waveform, sr)]
        for strength in STRENGTHS:
            tmp = denoise(clip.resolve(), strength)
            try:
                w2, sr2 = load_audio(tmp)
                scores.append(detector.predict(w2, sr2))
            finally:
                os.unlink(tmp)
        if scores[0] >= 0.5 and scores[-1] < 0.5:
            broke += 1
        print(f"{clip.stem[:25]:<26}" + "".join(f"{s:>9.3f}" for s in scores))

    print(
        f"\n{broke}/{len(clips)} genuine clips flipped from real to fake once denoised."
        "\nDo not pre-process audio before analysis."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
