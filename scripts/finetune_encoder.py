"""
Fine-tune the wav2vec2 encoder itself, not just a head on frozen features.

Why this and not another head: two very different training mixes both plateaued
at ~69% on unseen ElevenLabs while genuine accuracy sat near chance. A frozen
encoder can only offer the features it already has — and those come from a model
fine-tuned for a different deepfake task. If modern cloning is not linearly
separable in that space, no classifier on top of it will find the boundary.
Unfreezing lets the representation adapt.

Setup notes, all of them forced by an 8 GB card:

* **Convolutional feature extractor stays frozen.** Standard practice for
  wav2vec2 fine-tuning: those layers learn low-level filters that transfer, and
  freezing them saves memory and stabilises early training.
* **Mixed precision**, so activations fit alongside 94M parameters.
* **Discriminative learning rates** — a small one for the pretrained encoder so
  it is nudged rather than overwritten, a larger one for the fresh head.

Splits are inherited from train_mlaad.py, unchanged: generator-disjoint (all
three ElevenLabs variants, Edge-TTS and ChatTTS held out), speaker-disjoint on
LibriSpeech, and nothing from data/eval/ in training.

    python scripts/finetune_encoder.py --epochs 3 --batch 8
"""

import argparse
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402
from scripts.train_mlaad import centre_window, collect  # noqa: E402

OUT = PROJECT_ROOT / "checkpoints" / "finetuned_encoder"


WINDOW = 64600


def full_mono(path):
    """Whole file as 16 kHz mono, or None if it cannot be read."""
    try:
        waveform, sr = load_audio(str(path))
    except Exception:
        return None
    mono = waveform.mean(dim=0)
    if sr != 16000:
        mono = torchaudio.transforms.Resample(sr, 16000)(mono)
    return mono if mono.shape[0] >= WINDOW else None


class ClipDataset(Dataset):
    """
    Training samples a fresh random window per epoch; holdout stays on the
    cached centre window so its numbers remain comparable across runs.

    The earlier version cached one centre window per clip and reused it every
    epoch. That is a train/inference mismatch: the app scores *every* window in
    a recording, but the model had only ever been shown the middle 4 seconds of
    each training file. Measured on held-out clips, adjacent windows of the same
    genuine recording scored 0.077 and 0.984 -- the model was never asked to be
    consistent across a file, so it isn't. Random cropping asks it directly, and
    multiplies effective data without downloading anything new.

    Random crops cannot be cached, so training re-decodes each epoch. That trades
    a few minutes of disk for memory staying flat, which matters on this box.
    """

    def __init__(self, items, random_crop: bool = False, gain_jitter: bool = False,
                 noise_aug: bool = False, seed: int = 0):
        self.items = items
        self.random_crop = random_crop
        self.gain_jitter = gain_jitter
        self.noise_aug = noise_aug
        self.rng = random.Random(seed)
        self.cache: dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.items)

    def _deterministic(self, i):
        if i not in self.cache:
            w = centre_window(self.items[i][0])
            if w is None:
                w = torch.zeros(WINDOW)
            self.cache[i] = w.half()
        return self.cache[i].float()

    def _speech_window(self, mono):
        """
        A random window that actually contains speech.

        The live path skips any window quieter than SILENCE_RMS_THRESHOLD, so
        the model is never asked to classify silence in production. Feeding it
        silent crops during training is label noise -- pauses sound the same
        whoever produced them -- and an unfiltered random crop regressed an
        independent ElevenLabs clone from 0.107 to 0.596. The floor is relative
        to the file so it holds across corpora recorded at different levels.
        """
        limit = mono.shape[0] - WINDOW
        file_rms = float(mono.pow(2).mean().sqrt())
        floor = max(0.004, 0.5 * file_rms)

        best, best_rms = None, -1.0
        for _ in range(10):
            start = self.rng.randint(0, limit)
            w = mono[start : start + WINDOW]
            rms = float(w.pow(2).mean().sqrt())
            if rms >= floor:
                return w.clone()
            if rms > best_rms:
                best, best_rms = w, rms
        # Nothing cleared the bar: keep the loudest of the tries rather than
        # dropping the clip, so quiet recordings still contribute.
        return best.clone()

    def __getitem__(self, i):
        label = self.items[i][1]
        if not self.random_crop:
            return self._deterministic(i), label

        mono = full_mono(self.items[i][0])
        if mono is None:
            return torch.zeros(WINDOW), label
        w = self._speech_window(mono)

        if self.noise_aug and self.rng.random() < 0.5:
            # Applied to both classes at the same rate, on purpose. jay15k
            # genuine is noisy consumer audio while LibriSpeech and MLAAD's TTS
            # are both clean, so "clean" correlates with synthetic in training --
            # and LibriSpeech genuine sits 22 points below jay15k genuine because
            # of it. Adding noise to both classes equally removes the shortcut
            # without teaching the inverse one.
            snr_db = self.rng.uniform(10.0, 40.0)
            sig_rms = float(w.pow(2).mean().sqrt())
            if sig_rms > 0:
                noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
                w = w + torch.randn_like(w) * noise_rms
                peak = w.abs().max()
                if peak > 1.0:
                    w = w / peak

        if self.gain_jitter:
            # +/-6 dB. An earlier detector here was measurably level-sensitive
            # (0.000 at one peak level, 0.830 at another on the same clip), which
            # was patched with RMS normalisation at inference. Training across
            # levels makes the invariance a property of the model instead.
            w = w * (10.0 ** (self.rng.uniform(-6.0, 6.0) / 20.0))
            peak = w.abs().max()
            if peak > 1.0:
                w = w / peak

        return w, label


def evaluate(model, loader, device):
    model.eval()
    correct_by_label = defaultdict(list)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(input_values=x).logits.argmax(dim=1)
            for label, ok in zip(y.tolist(), (pred == y).tolist()):
                correct_by_label[label].append(ok)
    genuine = statistics.mean(correct_by_label[1]) if correct_by_label[1] else float("nan")
    synth = statistics.mean(correct_by_label[0]) if correct_by_label[0] else float("nan")
    both = correct_by_label[0] + correct_by_label[1]
    return statistics.mean(both), genuine, synth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--no-random-crop", action="store_true",
                    help="reproduce the old centre-window-only training")
    ap.add_argument("--no-gain-jitter", action="store_true")
    ap.add_argument("--noise-aug", action="store_true",
                    help="add noise to BOTH classes so recording cleanliness "
                         "stops correlating with the label")
    ap.add_argument("--balance-genuine", action="store_true",
                    help="downsample jay15k genuine to match LibriSpeech, so raising "
                         "--per-generator cannot dilute the harder genuine corpus")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No GPU visible. Fine-tuning on CPU is impractical; aborting.")
        return 1
    print(f"device: {device} ({torch.cuda.get_device_name(0)})")

    from transformers import AutoModelForAudioClassification

    items = collect(args.per_generator)
    train_items = [it for it in items if it[2] == "train"]

    if args.balance_genuine:
        # LibriSpeech is capped by how many clips its 32 training speakers have,
        # so --per-generator only ever adds jay15k. At 200 LibriSpeech was 48% of
        # genuine; at 400 it fell to 37%, and its accuracy fell 83.2% -> 77.0%
        # with it. Trimming jay15k restores the ratio. Only train is touched:
        # holdout stays exactly as collect() built it, so per-source evals remain
        # comparable across runs.
        libri = [it for it in train_items if str(it[3]).startswith("librispeech")]
        jay = [it for it in train_items if it[3] == "jay15k_genuine"]
        rest = [
            it for it in train_items
            if not str(it[3]).startswith("librispeech") and it[3] != "jay15k_genuine"
        ]
        rng = random.Random(0)
        rng.shuffle(jay)
        jay = jay[: len(libri)]
        train_items = rest + libri + jay
        print(f"balanced genuine: librispeech {len(libri)}  jay15k {len(jay)}")
    hold_items = [it for it in items if it[2] == "holdout"]
    random.Random(0).shuffle(train_items)
    print(f"train {len(train_items)}   holdout {len(hold_items)}")

    train_ds = ClipDataset(
        train_items,
        random_crop=not args.no_random_crop,
        gain_jitter=not args.no_gain_jitter,
        noise_aug=args.noise_aug,
    )
    # Holdout stays deterministic: centre window, no jitter, so its accuracy is
    # measured the same way before and after this change.
    hold_ds = ClipDataset(hold_items)
    print(
        f"augmentation: random_crop={not args.no_random_crop} "
        f"gain_jitter={not args.no_gain_jitter}"
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    hold_dl = DataLoader(hold_ds, batch_size=args.batch, num_workers=0)

    model = AutoModelForAudioClassification.from_pretrained(
        config.model.wav2vec_model_id, num_labels=2, ignore_mismatched_sizes=True
    )
    # Low-level conv filters transfer; freezing them saves memory and stabilises.
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    model.to(device)

    head_names = ("classifier", "projector")
    head_params = [p for n, p in model.named_parameters() if n.startswith(head_names)]
    enc_params = [
        p for n, p in model.named_parameters()
        if not n.startswith(head_names) and p.requires_grad
    ]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.encoder_lr},
         {"params": head_params, "lr": args.head_lr}],
        weight_decay=0.01,
    )

    n_gen = sum(1 for it in train_items if it[1] == 1)
    n_syn = len(train_items) - n_gen
    weight = torch.tensor(
        [len(train_items) / (2 * max(n_syn, 1)), len(train_items) / (2 * max(n_gen, 1))],
        device=device,
    )
    lossf = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda")
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.encoder_lr, args.head_lr], total_steps=steps, pct_start=0.1
    )

    # Never let a run destroy the checkpoint the app is serving. This script
    # writes straight to the live path, so a short smoke run with tiny settings
    # will happily replace a good model with a bad one -- which is exactly how
    # the previous checkpoint was lost. Move the existing weights aside first.
    if (OUT / "model.safetensors").exists():
        backup = OUT.parent / f"{OUT.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(OUT), str(backup))
        print(f"existing checkpoint backed up -> {backup.name}")
    OUT.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(input_values=x).logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running.append(float(loss.detach()))
            if step % 100 == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  epoch {epoch} step {step}/{len(train_dl)}  "
                      f"loss {statistics.mean(running[-100:]):.4f}  vram {mem:.1f} GB")

        acc, gen, syn = evaluate(model, hold_dl, device)
        print(f"epoch {epoch}: holdout acc {acc:.1%}  genuine {gen:.1%}  synthetic {syn:.1%}")

        # Select on the weaker class, not overall accuracy — a model that calls
        # everything synthetic scores well on a synthetic-heavy holdout.
        score = min(gen, syn)
        if score > best:
            best = score
            model.save_pretrained(OUT)
            print(f"  saved (balanced score {score:.1%}) -> {OUT}")

    print(f"\nbest balanced holdout score: {best:.1%}")
    print("Evaluate with: python scripts/eval_finetuned.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
