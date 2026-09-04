"""
Measure how far the detector drops when audio arrives the way the app sees it.

The app never gets clean WAV. Its only capture path is getDisplayMedia over a
browser tab, so a YouTube clip reaches the model having been Opus- or AAC-encoded
by the platform, decoded, mixed, resampled to 48 kHz by the audio graph and down
to 16 kHz by the recorder. Training data is clean WAV/FLAC. If accuracy holds on
clean audio but collapses after a codec round-trip, the model is right and the
deployment is wrong -- and that gap is invisible to eval_finetuned.py, which
scores the pristine files.

Each held-out clip is scored as-is, then re-scored after each codec round-trip,
so every condition sees the same clips and differences are attributable.

    python scripts/codec_robustness.py --per-generator 40
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from scripts.train_mlaad import centre_window, collect  # noqa: E402

# Bitrates chosen to bracket what streaming platforms actually serve for speech.
CONDITIONS = {
    "clean": None,
    "opus-64k": ("libopus", "64k", ".ogg"),
    "opus-96k": ("libopus", "96k", ".ogg"),
    "aac-128k": ("aac", "128k", ".m4a"),
}


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"no ffmpeg available: {exc}")


def transcode(src: Path, codec: str, bitrate: str, suffix: str, tmp: Path, exe: str):
    """Round-trip through a lossy codec at 48 kHz, the rate a tab actually plays at."""
    out = tmp / f"{src.stem}__{codec}{suffix}"
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src),
           "-ac", "1", "-ar", "48000", "-c:a", codec, "-b:a", bitrate, str(out)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return None
    return out


@torch.no_grad()
def predict(model, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return model(input_values=batch.to(device)).logits.argmax(dim=1).cpu()


def score(model, windows, want: int, device, batch_size: int = 16) -> tuple[float, int]:
    hits, total, buf = 0, 0, []
    for w in windows:
        buf.append(w)
        if len(buf) == batch_size:
            hits += int((predict(model, torch.stack(buf), device) == want).sum())
            total += len(buf)
            buf = []
    if buf:
        hits += int((predict(model, torch.stack(buf), device) == want).sum())
        total += len(buf)
    return (hits / total if total else float("nan")), total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "checkpoints/finetuned_encoder")
    ap.add_argument("--per-generator", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoModelForAudioClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForAudioClassification.from_pretrained(args.model).to(device).eval()
    exe = ffmpeg_bin()
    print(f"model: {args.model.name}   device: {device}\nffmpeg: {exe}\n")

    items = [it for it in collect(args.per_generator) if it[2] == "holdout"]
    # label 1 = genuine, matching the classifier's head.
    buckets = defaultdict(list)
    for path, label, _, source in items:
        buckets[("genuine" if label == 1 else "synthetic", source)].append((path, label))

    results: dict[str, dict[str, tuple[float, int]]] = defaultdict(dict)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for cond, spec in CONDITIONS.items():
            print(f"--- {cond} ---")
            for (kind, source), rows in sorted(buckets.items()):
                windows = []
                for path, _ in rows:
                    use = path
                    if spec is not None:
                        codec, bitrate, suffix = spec
                        use = transcode(path, codec, bitrate, suffix, tmp, exe)
                        if use is None:
                            continue
                    w = centre_window(use)
                    if w is not None:
                        windows.append(w)
                    if spec is not None and use.exists():
                        use.unlink()
                want = 1 if kind == "genuine" else 0
                acc, n = score(model, windows, want, device)
                results[f"{kind}/{source}"][cond] = (acc, n)
                print(f"  {kind:9s} {source:24s} {acc*100:5.1f}%  (n={n})")
            print()

    print("=" * 78)
    header = f"{'source':34s}" + "".join(f"{c:>11s}" for c in CONDITIONS)
    print(header)
    print("-" * len(header))
    for name in sorted(results):
        row = f"{name:34s}"
        for cond in CONDITIONS:
            acc, _ = results[name].get(cond, (float("nan"), 0))
            row += f"{acc*100:10.1f}%"
        print(row)

    print("\ndelta vs clean (negative = codec hurts)")
    print("-" * len(header))
    for name in sorted(results):
        base = results[name].get("clean", (float("nan"), 0))[0]
        row = f"{name:34s}{'':>11s}"
        for cond in list(CONDITIONS)[1:]:
            acc, _ = results[name].get(cond, (float("nan"), 0))
            row += f"{(acc - base)*100:+10.1f}pp"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
