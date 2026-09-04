"""
Train a spoof classifier on frozen wav2vec2 embeddings.

Guards against the ways this measurement can lie to itself:

* **Full-length clips only.** Anything shorter than the 4.04 s window gets
  tile-padded, and in this corpus the fake clips average 1.7 s longer than the
  real ones — so padding correlates with the label. Filtering to clips that fill
  the window removes that shortcut. It is the same class of trap as the MP3/M4A
  split that made the previous dataset unusable.

* **Balanced classes.** Equal counts, so accuracy cannot be won by guessing the
  majority. The previous attempt scored 57.5% where a constant answer scored 75%.

* **External evaluation is the headline, not the internal split.** This corpus
  ships no speaker or generator labels, so an internal split cannot be made
  speaker-disjoint and its accuracy is optimistic by an unknown margin. The sets
  that decide whether anything was learned are the ones from elsewhere entirely:
  genuine control recordings, SAPI renders, the ElevenLabs clone, and the
  UniData voice-conversion corpus. None appear in training.

* **Frozen encoder.** Only the head is trained, so the result is a fair test of
  what the pretrained representation already separates.

    python scripts/train_detector.py --limit 2500
    python scripts/train_detector.py --limit 2500 --epochs 400
"""

import argparse
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402

WINDOW = 64600
MIN_DURATION = WINDOW / 16000
CORPUS = Path(
    "D:/ml-cache/kaggle/datasets/jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
    "/versions/1/deepfake_audio_dataset_jay15k"
)
CACHE = PROJECT_ROOT / "checkpoints" / "embeddings.pt"


def build_embedder():
    from transformers import AutoFeatureExtractor, AutoModel

    model_id = config.model.wav2vec_model_id
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).eval()
    for p in enc.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def embed(sig: torch.Tensor) -> torch.Tensor:
        inputs = fe(sig.numpy(), sampling_rate=16000, return_tensors="pt")
        return enc(**inputs).last_hidden_state.mean(dim=1).squeeze(0)

    return embed


def centre_window(path: Path) -> torch.Tensor | None:
    """Middle 4.04 s at 16 kHz, or None if the clip is too short to fill it."""
    waveform, sr = load_audio(str(path))
    mono = waveform.mean(dim=0)
    if sr != 16000:
        mono = torchaudio.transforms.Resample(sr, 16000)(mono)
    if mono.shape[0] < WINDOW:
        return None
    start = (mono.shape[0] - WINDOW) // 2
    return mono[start : start + WINDOW]


def gather(limit: int, seed: int = 0):
    """Balanced sample of full-length clips from each class."""
    rng = random.Random(seed)
    picked = {}
    for label, name in ((1, "real"), (0, "fake")):
        files = sorted((CORPUS / name).glob("*.wav"))
        rng.shuffle(files)
        keep = []
        for f in files:
            try:
                if sf.info(str(f)).duration >= MIN_DURATION:
                    keep.append(f)
            except Exception:
                continue
            if len(keep) >= limit:
                break
        picked[label] = keep
    n = min(len(picked[0]), len(picked[1]))
    print(f"  usable full-length clips: real={len(picked[1])} fake={len(picked[0])} -> {n} each")
    return [(f, 1) for f in picked[1][:n]] + [(f, 0) for f in picked[0][:n]]


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, x):
        return self.net(x)


def evaluate(head, X, Y):
    with torch.no_grad():
        pred = head(X).argmax(dim=1)
    acc = float((pred == Y).float().mean())
    gen = Y == 1
    fake = Y == 0
    tpr = float((pred[gen] == 1).float().mean()) if gen.any() else float("nan")
    tnr = float((pred[fake] == 0).float().mean()) if fake.any() else float("nan")
    return acc, tpr, tnr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=2500, help="clips per class")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--refresh", action="store_true", help="ignore cached embeddings")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "detector_head.pt")
    args = ap.parse_args()

    embed = build_embedder()

    if CACHE.exists() and not args.refresh:
        blob = torch.load(CACHE)
        X, Y = blob["X"], blob["Y"]
        print(f"Loaded cached embeddings: {X.shape[0]} clips")
    else:
        print("Selecting clips ...")
        items = gather(args.limit)
        print(f"Embedding {len(items)} clips (frozen encoder, CPU) ...")
        feats, labels = [], []
        for i, (path, label) in enumerate(items, 1):
            w = centre_window(path)
            if w is None:
                continue
            feats.append(embed(w))
            labels.append(label)
            if i % 250 == 0:
                print(f"  {i}/{len(items)}")
        X = torch.stack(feats)
        Y = torch.tensor(labels)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "Y": Y}, CACHE)

    # held-out split of the corpus itself (optimistic - see module docstring)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=g)
    cut = int(0.8 * len(perm))
    tr, te = perm[:cut], perm[cut:]

    head = Head(X.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(X[tr]), Y[tr])
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            head.eval()
            acc, _, _ = evaluate(head, X[te], Y[te])
            print(f"  epoch {ep + 1:>4}  loss {float(loss):.4f}  held-out acc {acc:.1%}")
    head.eval()

    acc, tpr, tnr = evaluate(head, X[te], Y[te])
    print(f"\nIN-CORPUS HELD-OUT SPLIT  (optimistic: no speaker labels to separate on)")
    print(f"  accuracy {acc:.1%}   genuine correct {tpr:.1%}   synthetic correct {tnr:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "dim": X.shape[1]}, args.out)
    print(f"  saved head -> {args.out}")

    # ---- the numbers that decide it ---------------------------------------
    print("\nEXTERNAL SETS  (never seen in training)")
    uni = (
        Path.home() / ".cache/kagglehub/datasets/unidpro"
        / "real-vs-fake-human-voice-deepfake-audio/versions/1"
    )
    sets = [
        ("genuine control", sorted(Path(PROJECT_ROOT / "data/eval/control_real").glob("*.*")), 1),
        ("SAPI TTS", sorted(Path(PROJECT_ROOT / "data/eval/spoof_tts").glob("*.wav")), 0),
        ("ElevenLabs clone", sorted(Path(PROJECT_ROOT / "data/eval/spoof_clone").glob("*.mp3")), 0),
        ("UniData originals", sorted(uni.glob("*/*/*/original.*")), 1),
        ("UniData conversions", sorted(uni.glob("*/*/*/synthetic_1.mp3")), 0),
    ]
    print(f"{'set':<24}{'n':>5}{'correct':>10}")
    print("-" * 41)
    for name, files, want in sets:
        files = [f for f in files if f.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}]
        if not files:
            continue
        ok = tot = 0
        for f in files:
            w = centre_window(f)
            if w is None:
                continue
            with torch.no_grad():
                pred = int(head(embed(w).unsqueeze(0)).argmax())
            ok += int(pred == want)
            tot += 1
        if tot:
            print(f"{name:<24}{tot:>5}{ok / tot:>10.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
