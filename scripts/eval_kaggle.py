"""
Evaluate the detector on the UniData "Real vs Fake Human Voice" set from Kaggle.

    python scripts/eval_kaggle.py                    # downloads via kagglehub if needed
    python scripts/eval_kaggle.py --path <dir>       # or point at an existing copy
    python scripts/eval_kaggle.py --csv kaggle.csv

Layout: {UK,USA}/{female,male}/{1..5}/original.* + synthetic_{1,2,3}.mp3

The trailing 1..5 is the *utterance group*, not a speaker id — the metadata sheet
shows only FOUR speakers (UK female, UK male, USA female, USA male), each reading
five scripted passages. So it is 80 files but only 4 voices, and the 20 genuine
clips are not 20 independent samples. Treat error rates accordingly.

Within each group the original and its three synthetics share the same script, so
they are matched pairs — which is what makes the paired test below possible.

Three things this reports that a single accuracy number would hide:

* **Codec control.** Every synthetic is MP3, while the originals are a mix of
  M4A, MP3 and WAV. If the detector were keying on compression artefacts rather
  than synthesis, the MP3 originals would score like the fakes. They are the
  control group; watch them.
* **Paired comparison.** Within each utterance group, is the fake scored as more
  synthetic than the genuine reading of the same script? Pairing removes speaker
  and content variation, so it detects a real effect at far smaller n.
* **False alarms on genuine speech**, reported separately from misses. These are
  different failures with different costs.

Licence note: the dataset is CC BY-NC-ND 4.0 — fine for evaluation like this,
but it forbids derivative works, so do not train or fine-tune on it.
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate import (  # noqa: E402
    Pipeline,
    best_threshold,
    describe,
    equal_error_rate,
)

KAGGLE_SLUG = "unidpro/real-vs-fake-human-voice-deepfake-audio"


def resolve_dataset(path: Path | None) -> Path:
    if path is not None:
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        return path
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Either `pip install kagglehub` or pass --path."
        ) from None
    print(f"[kagglehub] resolving {KAGGLE_SLUG} ...")
    return Path(kagglehub.dataset_download(KAGGLE_SLUG))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, help="existing copy of the dataset")
    parser.add_argument("--csv", type=Path, help="write per-file rows here")
    args = parser.parse_args()

    root = resolve_dataset(args.path)
    speakers = sorted(p for p in root.glob("*/*/*") if p.is_dir())
    if not speakers:
        raise SystemExit(f"No speaker folders under {root}")

    pipeline = Pipeline()
    if not pipeline.spoof.is_calibrated:
        print("\n!! Spoof detector is UNCALIBRATED - run scripts/download_checkpoints.py\n")
        return 1

    rows = []
    print(f"\nScoring {len(speakers)} speakers from {root}\n")
    print(f"{'group':<18}{'file':<16}{'fmt':<6}{'label':<10}{'dur':>7}{'AASIST':>9}{'risk':>7}")
    print("-" * 73)

    for speaker_dir in speakers:
        speaker = "/".join(speaker_dir.parts[-3:])
        for audio in sorted(speaker_dir.iterdir()):
            if audio.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
                continue
            label = "bonafide" if audio.stem == "original" else "spoof"
            try:
                row = pipeline.score(audio, label)
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {speaker}/{audio.name}: {e}")
                continue
            rows.append((speaker, audio, label, row))
            print(
                f"{speaker:<18}{audio.name:<16}{audio.suffix[1:]:<6}{label:<10}"
                f"{row.duration_s:6.1f}s{row.bonafide:9.3f}{row.risk:7.1f}"
            )

    if not rows:
        return 1

    bona = [r.risk for _, _, lab, r in rows if lab == "bonafide"]
    spoof = [r.risk for _, _, lab, r in rows if lab == "spoof"]

    print("\n" + "=" * 73)
    print(f"\nBONAFIDE ({len(bona)} genuine recordings)")
    describe("risk", bona)
    describe("bonafide   ", [r.bonafide * 100 for _, _, lab, r in rows if lab == "bonafide"], "%")
    print(f"\nSPOOF ({len(spoof)} synthetic recordings)")
    describe("risk", spoof)
    describe("bonafide   ", [r.bonafide * 100 for _, _, lab, r in rows if lab == "spoof"], "%")

    # ---- codec control -----------------------------------------------------
    print("\nCODEC CONTROL (are we detecting synthesis, or MP3 compression?)")
    by_fmt = defaultdict(list)
    for _, audio, label, row in rows:
        by_fmt[(label, audio.suffix[1:].lower())].append(row.risk)
    for (label, fmt), risks in sorted(by_fmt.items()):
        print(
            f"  {label:<10} {fmt:<5} n={len(risks):<3} "
            f"mean risk {statistics.mean(risks):5.1f}   "
            f"range {min(risks):5.1f}-{max(risks):5.1f}"
        )
    mp3_bona = by_fmt.get(("bonafide", "mp3"), [])
    other_bona = [r for (lab, fmt), rs in by_fmt.items() if lab == "bonafide" and fmt != "mp3"
                  for r in rs]
    if mp3_bona and other_bona:
        delta = statistics.mean(mp3_bona) - statistics.mean(other_bona)
        print(f"\n  MP3 originals vs other originals: {delta:+.1f} risk")
        print("  A large positive gap would mean the score partly tracks the codec,")
        print("  not the synthesis. Small gap = the comparison is sound.")

    # ---- paired test -------------------------------------------------------
    print("\nPAIRED BY UTTERANCE (does each fake outscore the real clip of the same script?)")
    ordered = 0
    pairs = 0
    for speaker_dir in speakers:
        speaker = "/".join(speaker_dir.parts[-3:])
        mine = [(lab, r) for sp, _, lab, r in rows if sp == speaker]
        real = [r.risk for lab, r in mine if lab == "bonafide"]
        fake = [r.risk for lab, r in mine if lab == "spoof"]
        if not real or not fake:
            continue
        pairs += 1
        if min(fake) > real[0]:
            ordered += 1
    if pairs:
        print(f"  {ordered}/{pairs} groups: every fake scored higher than its genuine clip")
        print(f"  ({ordered / pairs * 100:.0f}% correctly ordered)")

    # ---- operating point ---------------------------------------------------
    current = pipeline.scorer.alert_threshold
    fa = sum(1 for b in bona if b >= current)
    miss = sum(1 for s in spoof if s < current)
    print(f"\nAT THE CONFIGURED THRESHOLD ({current:.0f})")
    print(f"  genuine flagged as synthetic : {fa}/{len(bona)}  ({fa / len(bona) * 100:.0f}%)")
    print(f"  synthetic missed             : {miss}/{len(spoof)}  ({miss / len(spoof) * 100:.0f}%)")

    eer, eer_t = equal_error_rate(bona, spoof)
    t, best_fa, best_miss = best_threshold(bona, spoof)
    gap = min(spoof) - max(bona)
    print("\nSEPARATION")
    print(f"  highest genuine risk : {max(bona):.1f}")
    print(f"  lowest spoof risk    : {min(spoof):.1f}")
    print(f"  margin               : {gap:+.1f}  ({'clean split' if gap > 0 else 'OVERLAP'})")
    print(f"  EER                  : {eer:.1f}% at threshold {eer_t:.1f}")
    print(f"  best threshold       : {t:.1f}  ({best_fa} false alarms, {best_miss} misses)")

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["speaker", "file", "format", "label", "duration_s",
                        "bonafide", "naturalness", "risk"])
            for speaker, audio, label, row in rows:
                w.writerow([speaker, audio.name, audio.suffix[1:], label,
                            f"{row.duration_s:.2f}", f"{row.bonafide:.4f}",
                            f"{row.naturalness:.4f}", f"{row.risk:.1f}"])
        print(f"\nWrote {len(rows)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
