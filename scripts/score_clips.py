"""
Score individual audio files exactly the way the running app scores them.

eval_finetuned.py takes one centre window per file, which is a lottery: adjacent
windows of the same recording have come back 0.003 and 0.999. The REST path and
the live stream both reduce over several windows instead, so a single-window
number here would not match what the UI shows. This walks the whole file and
reports the median, plus the spread that median is hiding.

    python scripts/score_clips.py data/eval/_personal_archive/*.mp3
    python scripts/score_clips.py --codecs data/eval/_personal_archive/kabab_real_raw.ogg
"""

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402

WINDOW = 64600
CODECS = {
    "opus-64k": ("libopus", "64k", ".ogg"),
    "aac-128k": ("aac", "128k", ".m4a"),
}


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def windows_of(path: Path, hop: int = WINDOW // 2):
    """Every window in the file, not just the middle one."""
    waveform, sr = load_audio(str(path))
    mono = waveform.mean(dim=0)
    if sr != 16000:
        mono = torchaudio.transforms.Resample(sr, 16000)(mono)
    if mono.shape[0] < WINDOW:
        return []
    return [mono[s : s + WINDOW] for s in range(0, mono.shape[0] - WINDOW + 1, hop)]


@torch.no_grad()
def genuine_probs(detector, wins) -> list[float]:
    out = []
    for w in wins:
        out.append(float(detector.predict(w.unsqueeze(0), 16000)))
    return out


def report(name: str, probs: list[float]) -> None:
    if not probs:
        print(f"  {name:14s} too short to score")
        return
    med = statistics.median(probs)
    verdict = "GENUINE" if med >= 0.5 else "SYNTHETIC"
    spread = f"{min(probs):.3f}..{max(probs):.3f}"
    print(f"  {name:14s} median P(genuine)={med:.3f}  {verdict:9s} n={len(probs):2d} range {spread}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--codecs", action="store_true",
                    help="also score after Opus/AAC round-trips, as the app receives them")
    args = ap.parse_args()

    from backend.models.finetuned_spoof import FineTunedSpoofDetector

    detector = FineTunedSpoofDetector()
    print(f"backend={detector.backend}  calibrated={detector.is_calibrated}  "
          f"device={config.device}\n")

    exe = ffmpeg_bin() if args.codecs else None
    for path in args.paths:
        if not path.exists():
            print(f"{path}: missing")
            continue
        print(f"{path.name}")
        report("clean", genuine_probs(detector, windows_of(path)))
        if args.codecs:
            with tempfile.TemporaryDirectory() as td:
                for label, (codec, bitrate, suffix) in CODECS.items():
                    out = Path(td) / f"{path.stem}{suffix}"
                    cmd = [exe, "-y", "-loglevel", "error", "-i", str(path),
                           "-ac", "1", "-ar", "48000", "-c:a", codec, "-b:a", bitrate, str(out)]
                    if subprocess.run(cmd, capture_output=True).returncode != 0:
                        print(f"  {label:14s} transcode failed")
                        continue
                    report(label, genuine_probs(detector, windows_of(out)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
