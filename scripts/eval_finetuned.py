"""
Evaluate the fine-tuned encoder per source, so averages cannot hide a collapse.

Reports every held-out source separately: each unseen generator, each genuine
corpus, and the cross-corpus external sets. A single accuracy figure over a
synthetic-heavy holdout can look healthy while genuine speech sits at chance —
which is exactly what happened with the frozen-feature heads.

    python scripts/eval_finetuned.py
    python scripts/eval_finetuned.py --model checkpoints/finetuned_encoder
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from scripts.train_mlaad import centre_window, collect  # noqa: E402


def load(model_dir: Path, device: torch.device):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    fe = AutoFeatureExtractor.from_pretrained(config.model.wav2vec_model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_dir).to(device).eval()
    return fe, model


@torch.no_grad()
def predict(model, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return model(input_values=batch.to(device)).logits.argmax(dim=1).cpu()


def score_paths(model, paths, want: int, device, batch_size: int = 16) -> tuple[float, int]:
    hits, total, buf = 0, 0, []
    for p in paths:
        w = centre_window(p)
        if w is None:
            continue
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
    ap.add_argument("--per-generator", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model = load(args.model, device)
    print(f"model: {args.model.name}   device: {device}\n")

    items = [it for it in collect(args.per_generator) if it[2] == "holdout"]
    by_source = defaultdict(list)
    for path, label, _, source in items:
        key = "LibriSpeech (unseen speakers)" if source.startswith("librispeech:") else source
        by_source[key].append((path, label))

    print("HELD OUT — never trained on")
    print(f"{'source':<34}{'kind':<11}{'correct':>9}{'n':>7}")
    print("-" * 62)
    genuine_scores, synth_scores = [], []
    for source in sorted(by_source):
        entries = by_source[source]
        want = entries[0][1]
        acc, n = score_paths(model, [p for p, _ in entries], want, device)
        (genuine_scores if want == 1 else synth_scores).append(acc)
        print(f"{source:<34}{'genuine' if want else 'synthetic':<11}{acc:>9.1%}{n:>7}")
    print("-" * 62)
    if genuine_scores:
        print(f"{'mean genuine':<34}{'':<11}{statistics.mean(genuine_scores):>9.1%}")
    if synth_scores:
        print(f"{'mean synthetic':<34}{'':<11}{statistics.mean(synth_scores):>9.1%}")

    print("\nEXTERNAL — different corpora entirely")
    print(f"{'set':<34}{'kind':<11}{'correct':>9}{'n':>7}")
    print("-" * 62)
    for name, folder, want in [
        ("control_real (3 corpora)", PROJECT_ROOT / "data/eval/control_real", 1),
        ("SAPI TTS", PROJECT_ROOT / "data/eval/spoof_tts", 0),
    ]:
        paths = [p for p in folder.glob("*")
                 if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}]
        if not paths:
            continue
        acc, n = score_paths(model, paths, want, device)
        print(f"{name:<34}{'genuine' if want else 'synthetic':<11}{acc:>9.1%}{n:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
