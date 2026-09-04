"""
Train a spoof detector on MLAAD, with generator- and speaker-disjoint evaluation.

Replaces an earlier script of this name that, despite the name, versioned its
output "mlaad_v1" while training on the Kaggle corpus plus LibriSpeech and 28
clips lifted out of data/eval/. Zero MLAAD clips were involved and an evaluation
set was contaminated. This one trains on MLAAD and touches nothing in data/eval/.

The design exists to make one question answerable:

    Does training on many modern TTS systems catch a cloner it has never seen?

* **Generator-disjoint.** 19 systems train; ElevenLabs v2 / v2.5 / v3, Edge-TTS
  and ChatTTS are held out entirely. ElevenLabs is the generator that defeated
  every off-the-shelf detector tried on this project, so it is the test, not the
  training data.

* **Speaker-disjoint genuine.** LibriSpeech filenames encode the speaker
  ("1089-134686-0000.flac" -> speaker 1089). Test speakers never appear in
  training, so genuine accuracy is not voice memorisation.

* **A domain confound that cannot be designed away, only measured.** Every
  genuine clip here is LibriSpeech audiobook reading and every synthetic clip is
  MLAAD. A model can score well by recognising "LibriSpeech-ness" rather than
  authenticity. The external genuine sets — IEMOCAP, CommonVoice, CommonLanguage
  samples and the project's own recordings — are the check on that, and they are
  reported separately for exactly this reason.

* **Full-length clips only**, so tile-padding cannot correlate with the label.

    python scripts/train_mlaad.py
    python scripts/train_mlaad.py --per-generator 200 --epochs 600
"""

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402

DATA = Path("D:/ml-data")
MLAAD = DATA / "mlaad" / "en"
GENUINE = DATA / "genuine" / "librispeech"
# Second genuine source, deliberately different in recording conditions.
# Training genuine speech on LibriSpeech alone taught an earlier run to
# recognise "clean studio audiobook" rather than "human": it scored 68% on
# held-out genuine and 0% on the project's own phone recordings.
JAY15K = Path(
    "D:/ml-cache/kaggle/datasets/jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech"
    "/versions/1/deepfake_audio_dataset_jay15k"
)
WINDOW = 64600
MIN_DURATION = WINDOW / 16000
CACHE = PROJECT_ROOT / "checkpoints" / "mlaad_cache" / "mlaad_embeddings.pt"


class Head(nn.Module):
    """Must match backend/models/trained_spoof.py ImprovedHead."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def build_embedder(device: torch.device):
    from transformers import AutoFeatureExtractor, AutoModel

    model_id = config.model.wav2vec_model_id
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def embed(sig: torch.Tensor) -> torch.Tensor:
        inputs = fe(sig.numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Keep features on CPU: the whole set is held in memory at once and
        # 8 GB of VRAM is better spent on the encoder than on a feature cache.
        return enc(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu()

    return embed


def centre_window(path: Path):
    try:
        waveform, sr = load_audio(str(path))
    except Exception:
        return None
    mono = waveform.mean(dim=0)
    if sr != 16000:
        mono = torchaudio.transforms.Resample(sr, 16000)(mono)
    if mono.shape[0] < WINDOW:
        return None
    start = (mono.shape[0] - WINDOW) // 2
    return mono[start : start + WINDOW]


def long_enough(path: Path) -> bool:
    try:
        return sf.info(str(path)).duration >= MIN_DURATION
    except Exception:
        return False


def collect(per_generator: int, seed: int = 0):
    """(path, label, group, speaker) with label 1 = genuine."""
    rng = random.Random(seed)
    items = []

    for group in ("train", "holdout"):
        base = MLAAD / group
        if not base.is_dir():
            continue
        for gen_dir in sorted(base.iterdir()):
            if not gen_dir.is_dir():
                continue
            files = [f for f in sorted(gen_dir.glob("*.wav")) if long_enough(f)]
            rng.shuffle(files)
            for f in files[:per_generator]:
                items.append((f, 0, group, gen_dir.name))

    # --- second genuine corpus + a second synthesis family, for diversity ---
    for sub, label, tag in (("real", 1, "jay15k_genuine"), ("fake", 0, "jay15k_synth")):
        folder = JAY15K / sub
        if not folder.is_dir():
            continue
        files = sorted(folder.glob("*.wav"))
        rng.shuffle(files)
        keep, want = [], per_generator * (10 if label == 1 else 5)
        for f in files:
            if long_enough(f):
                keep.append(f)
            if len(keep) >= want:
                break
        # No speaker labels in this corpus, so the split is random rather than
        # speaker-disjoint; its holdout accuracy is optimistic by that margin.
        cut = int(0.8 * len(keep))
        for i, f in enumerate(keep):
            items.append((f, label, "train" if i < cut else "holdout", tag))

    genuine = [f for f in sorted(GENUINE.glob("*.flac")) if long_enough(f)]
    rng.shuffle(genuine)
    # LibriSpeech: "<speaker>-<chapter>-<utt>.flac"
    by_speaker = defaultdict(list)
    for f in genuine:
        by_speaker[f.stem.split("-")[0]].append(f)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    cut = int(0.8 * len(speakers))
    train_spk, test_spk = set(speakers[:cut]), set(speakers[cut:])
    for spk, files in by_speaker.items():
        grp = "train" if spk in train_spk else "holdout"
        for f in files:
            items.append((f, 1, grp, f"librispeech:{spk}"))

    print(f"  genuine speakers: {len(train_spk)} train / {len(test_spk)} held out")
    return items


def report(head, X, Y, tag):
    with torch.no_grad():
        pred = head(X).argmax(dim=1)
    gen, fake = Y == 1, Y == 0
    acc = float((pred == Y).float().mean())
    tpr = float((pred[gen] == 1).float().mean()) if gen.any() else float("nan")
    tnr = float((pred[fake] == 0).float().mean()) if fake.any() else float("nan")
    print(f"  {tag:<34} acc {acc:>6.1%}   genuine {tpr:>6.1%}   synthetic {tnr:>6.1%}")
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "detector_head.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"device: {device}{label}")
    embed = build_embedder(device)

    if CACHE.exists() and not args.refresh:
        blob = torch.load(CACHE, weights_only=False)
        X, Y, groups, sources = blob["X"], blob["Y"], blob["groups"], blob["sources"]
        print(f"Loaded cached embeddings: {X.shape[0]} clips")
    else:
        print("Selecting clips ...")
        items = collect(args.per_generator)
        print(f"Embedding {len(items)} clips (frozen encoder, CPU) ...")
        feats, labels, groups, sources = [], [], [], []
        for i, (path, label, group, source) in enumerate(items, 1):
            w = centre_window(path)
            if w is None:
                continue
            feats.append(embed(w))
            labels.append(label)
            groups.append(group)
            sources.append(source)
            if i % 500 == 0:
                print(f"  {i}/{len(items)}")
        X = torch.stack(feats)
        Y = torch.tensor(labels)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X, "Y": Y, "groups": groups, "sources": sources}, CACHE)

    is_train = torch.tensor([g == "train" for g in groups])
    Xtr, Ytr = X[is_train], Y[is_train]
    Xho, Yho = X[~is_train], Y[~is_train]
    print(
        f"\ntrain:   {Xtr.shape[0]:>5} clips "
        f"({int((Ytr == 1).sum())} genuine, {int((Ytr == 0).sum())} synthetic)"
    )
    print(
        f"holdout: {Xho.shape[0]:>5} clips "
        f"({int((Yho == 1).sum())} genuine, {int((Yho == 0).sum())} synthetic)"
    )

    # class weights, since genuine and synthetic counts differ
    n_gen, n_syn = float((Ytr == 1).sum()), float((Ytr == 0).sum())
    weight = torch.tensor([len(Ytr) / (2 * n_syn), len(Ytr) / (2 * n_gen)])

    head = Head(X.shape[1]).to(device)
    Xtr, Ytr, Xho, Yho = Xtr.to(device), Ytr.to(device), Xho.to(device), Yho.to(device)
    X = X.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss(weight=weight.to(device))
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xtr), Ytr)
        loss.backward()
        opt.step()
        if (ep + 1) % 200 == 0:
            head.eval()
            with torch.no_grad():
                acc = float((head(Xho).argmax(1) == Yho).float().mean())
            print(f"  epoch {ep + 1:>4}  loss {float(loss.detach()):.4f}  holdout acc {acc:.1%}")
    head.eval()

    print("\nHELD OUT: unseen generators + unseen speakers")
    report(head, Xho, Yho, "all holdout")

    print("\nPER HELD-OUT GENERATOR  (never trained on)")
    by_src = defaultdict(list)
    for i, s in enumerate(sources):
        if groups[i] == "holdout" and Y[i] == 0:
            by_src[s].append(i)
    for src in sorted(by_src):
        idx = torch.tensor(by_src[src])
        want = int(Y[idx][0])
        with torch.no_grad():
            correct = float((head(X[idx].to(device)).argmax(1) == want).float().mean())
        kind = "genuine" if want == 1 else "synthetic"
        print(f"  {src:<34} {kind:<10} correct {correct:>6.1%}  (n={len(idx)})")

    torch.save(
        {"state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
         "dim": X.shape[1], "version": "mlaad_v1",
         "n_train_samples": int(Xtr.shape[0])},
        args.out,
    )
    print(f"\nsaved -> {args.out}")

    print("\nEXTERNAL SETS  (different corpora entirely; none in training)")
    ext = [
        ("genuine control", PROJECT_ROOT / "data/eval/control_real", 1),
        ("your real recordings", PROJECT_ROOT / "data/eval/bonafide", 1),
        ("SAPI TTS", PROJECT_ROOT / "data/eval/spoof_tts", 0),
        ("your ElevenLabs clone", PROJECT_ROOT / "data/eval/spoof_clone", 0),
    ]
    for name, folder, want in ext:
        files = [p for p in folder.glob("*")
                 if p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}]
        scores = []
        for f in files:
            w = centre_window(f)
            if w is None:
                continue
            with torch.no_grad():
                # embed() returns CPU tensors by design; the head lives on `device`.
                feat = embed(w).unsqueeze(0).to(device)
                scores.append(int(head(feat).argmax()) == want)
        if scores:
            print(f"  {name:<26} {statistics.mean(scores):>6.1%}  (n={len(scores)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
