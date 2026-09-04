"""
Train a classifier head on frozen wav2vec2 embeddings, using the UniData set.

Design decisions forced by the data, all of them defensive:

* **Frozen encoder + linear head, not full fine-tuning.** The corpus is 80 files.
  Fine-tuning 94M parameters on that memorises it outright. A linear probe on
  frozen features has a few thousand parameters and can be honestly evaluated.

* **Speaker-disjoint folds.** The set contains only FOUR speakers (UK/USA x
  female/male), each reading five passages. A random split would put the same
  voice in train and test, so "accuracy" would measure voice memorisation. Each
  fold here holds out one entire speaker.

* **External evaluation.** Held-out-speaker accuracy still only says "does this
  transfer within one corpus". The number that matters is performance on data
  from a different world entirely: the genuine control clips, the SAPI TTS
  renders, and the ElevenLabs clone — none of which are in training.

* **Codec confound is reported, not assumed away.** Every synthetic here is MP3
  and the originals are not, so a classifier can score well by detecting MP3.
  The external sets break that shortcut: if the head has learned codec rather
  than synthesis, it will fail on them while looking excellent in-corpus.

Licence: the corpus is CC BY-NC-ND. Keep any model trained here local; do not
redistribute the weights.

    python scripts/train_head.py --data <path to UK/USA folders>
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402

WINDOW = 64600
WINDOWS_PER_FILE = 4


def embedder():
    """Frozen wav2vec2 encoder; mean-pooled hidden states as features."""
    from transformers import AutoFeatureExtractor, AutoModel

    model_id = config.model.wav2vec_model_id
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).eval()
    for p in enc.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def embed(sig: torch.Tensor) -> torch.Tensor:
        inputs = fe(sig.numpy(), sampling_rate=16000, return_tensors="pt")
        out = enc(**inputs).last_hidden_state
        return out.mean(dim=1).squeeze(0)

    return embed


def windows_of(path: Path) -> list[torch.Tensor]:
    waveform, sr = load_audio(str(path))
    mono = waveform.mean(dim=0)
    if sr != 16000:
        mono = torchaudio.transforms.Resample(sr, 16000)(mono)
    if mono.shape[0] <= WINDOW:
        return [mono]
    step = (mono.shape[0] - WINDOW) / max(1, WINDOWS_PER_FILE - 1)
    return [mono[int(i * step) : int(i * step) + WINDOW] for i in range(WINDOWS_PER_FILE)]


def collect(root: Path):
    """Returns (path, label, speaker) — speaker is country/gender, not the folder number."""
    items = []
    for country in ("UK", "USA"):
        for gender in ("female", "male"):
            base = root / country / gender
            if not base.is_dir():
                continue
            for group in sorted(base.iterdir()):
                if not group.is_dir():
                    continue
                for audio in sorted(group.iterdir()):
                    if audio.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
                        continue
                    label = 1 if audio.stem == "original" else 0  # 1 = genuine
                    items.append((audio, label, f"{country}/{gender}"))
    return items


def train_head(x: torch.Tensor, y: torch.Tensor, epochs: int = 300) -> nn.Module:
    """Logistic probe with class weighting — the corpus is 1 genuine to 3 fake."""
    head = nn.Linear(x.shape[1], 2)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    weight = torch.tensor([len(y) / (2 * max(neg, 1)), len(y) / (2 * max(pos, 1))])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=weight)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(head(x), y)
        loss.backward()
        opt.step()
    return head.eval()


def accuracy(head: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    with torch.no_grad():
        pred = head(x).argmax(dim=1)
    genuine = (y == 1)
    fake = (y == 0)
    acc = float((pred == y).float().mean())
    tpr = float((pred[genuine] == 1).float().mean()) if genuine.any() else float("nan")
    tnr = float((pred[fake] == 0).float().mean()) if fake.any() else float("nan")
    return acc, tpr, tnr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder containing UK/ and USA/")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "probe.pt")
    args = parser.parse_args()

    items = collect(args.data)
    if not items:
        raise SystemExit(f"No audio found under {args.data}")
    speakers = sorted({s for _, _, s in items})
    print(f"\n{len(items)} files, {len(speakers)} speakers: {speakers}")
    n_gen = sum(1 for _, l, _ in items if l == 1)
    n_syn = sum(1 for _, l, _ in items if l == 0)
    print(f"genuine={n_gen}  synthetic={n_syn}")
    print(f"majority-class baseline (always say synthetic): {n_syn / len(items):.1%}")

    embed = embedder()
    print("\nExtracting frozen embeddings ...")
    feats, labels, spk = [], [], []
    for i, (path, label, speaker) in enumerate(items, 1):
        for w in windows_of(path):
            feats.append(embed(w))
            labels.append(label)
            spk.append(speaker)
        if i % 20 == 0:
            print(f"  {i}/{len(items)} files")
    X = torch.stack(feats)
    Y = torch.tensor(labels)
    print(f"  {X.shape[0]} windows, {X.shape[1]}-d features")

    # ---- speaker-disjoint cross-validation --------------------------------
    print("\nSPEAKER-DISJOINT FOLDS  (train on 3 voices, test on the 4th)")
    print(f"{'held-out speaker':<20}{'accuracy':>10}{'genuine ok':>12}{'fake ok':>10}")
    print("-" * 52)
    scores = []
    for held in speakers:
        mask = torch.tensor([s == held for s in spk])
        head = train_head(X[~mask], Y[~mask])
        acc, tpr, tnr = accuracy(head, X[mask], Y[mask])
        scores.append(acc)
        print(f"{held:<20}{acc:>10.1%}{tpr:>12.1%}{tnr:>10.1%}")
    print("-" * 52)
    print(f"{'mean':<20}{sum(scores) / len(scores):>10.1%}")

    # ---- final head on everything, for external evaluation ----------------
    head = train_head(X, Y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "dim": X.shape[1]}, args.out)
    print(f"\nSaved probe to {args.out}  (keep local - source data is CC BY-NC-ND)")

    # ---- the test that actually matters -----------------------------------
    print("\nEXTERNAL DATA  (nothing below appeared in training)")
    external = {
        "genuine control": (PROJECT_ROOT / "data/eval/control_real", 1),
        "SAPI TTS": (PROJECT_ROOT / "data/eval/spoof_tts", 0),
        "ElevenLabs clone": (PROJECT_ROOT / "data/eval/spoof_clone", 0),
    }
    print(f"{'set':<20}{'n':>4}{'correct':>10}")
    print("-" * 36)
    for name, (folder, want) in external.items():
        files = [p for p in folder.glob("*") if p.suffix.lower() in
                 {".wav", ".flac", ".mp3", ".m4a", ".ogg"}]
        if not files:
            continue
        ok = tot = 0
        for f in files:
            for w in windows_of(f):
                with torch.no_grad():
                    pred = int(head(embed(w).unsqueeze(0)).argmax())
                ok += int(pred == want)
                tot += 1
        print(f"{name:<20}{tot:>4}{ok / tot:>10.1%}")

    print(
        "\nRead the external block, not the folds. High fold accuracy with poor"
        "\nexternal accuracy means the probe learned these 4 voices or the MP3"
        "\ncodec, not synthesis."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
