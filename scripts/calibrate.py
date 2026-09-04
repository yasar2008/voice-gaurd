"""
Score a folder of audio through the detection pipeline and report the numbers you
need to set a threshold — instead of trusting the default of 65.

    # distribution of one unlabelled folder
    python scripts/calibrate.py --folder data/eval/spoof_tts

    # both classes: separation, EER, and a suggested threshold
    python scripts/calibrate.py --bonafide data/eval/bonafide --spoof data/eval/spoof_tts

    # several sets at once - each source folder is also reported separately
    python scripts/calibrate.py --spoof data/eval/spoof_tts data/eval/spoof_clone

    # write per-file rows for later analysis
    python scripts/calibrate.py --bonafide ... --spoof ... --csv results.csv

Runs the models in-process (no HTTP server needed). Speaker verification is
skipped unless --enroll is given, matching how the app behaves with no voiceprint.
"""

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config  # noqa: E402
from backend.data.audio_io import load_audio  # noqa: E402
from backend.features.prosody import ProsodyAnalyzer  # noqa: E402
from backend.fusion.risk_scorer import RiskScorer  # noqa: E402
from backend.models.aasist import AASISTDetector  # noqa: E402
from backend.models.finetuned_spoof import FineTunedSpoofDetector  # noqa: E402
from backend.models.speaker_verify import SpeakerVerifier  # noqa: E402
from backend.models.trained_spoof import TrainedSpoofDetector  # noqa: E402
from backend.models.wav2vec_spoof import Wav2VecSpoofDetector  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class Row:
    path: Path
    label: str
    duration_s: float
    bonafide: float          # AASIST: 1.0 = genuine
    speaker: float | None    # cosine similarity, None when not enrolled
    naturalness: float       # prosody
    risk: float              # fused 0-100
    anomalies: list[str] = field(default_factory=list)


class Pipeline:
    """The same four stages the API runs, wired up directly."""

    def __init__(self, enroll_path: Path | None = None):
        if config.model.spoof_backend == "finetuned":
            ft = FineTunedSpoofDetector(device=config.device)
            self.spoof = ft if ft.is_calibrated else Wav2VecSpoofDetector(device=config.device)
        elif config.model.spoof_backend == "aasist":
            self.spoof = AASISTDetector(
                checkpoint_path=str(config.model.aasist_checkpoint),
                device=config.device,
            )
        elif config.model.spoof_backend == "trained":
            trained = TrainedSpoofDetector(device=config.device)
            self.spoof = (
                trained
                if trained.is_calibrated
                else Wav2VecSpoofDetector(device=config.device)
            )
        else:
            self.spoof = Wav2VecSpoofDetector(device=config.device)
        self.speaker = SpeakerVerifier(device=config.device)
        self.prosody = ProsodyAnalyzer()
        self.scorer = RiskScorer()

        if enroll_path is not None:
            waveform, sr = load_audio(str(enroll_path))
            self.speaker.enroll(waveform, sr)
            print(f"[enrol] reference voice: {enroll_path.name}")

    def score(self, path: Path, label: str) -> Row:
        waveform, sr = load_audio(str(path))
        duration = waveform.shape[1] / sr

        bonafide = self.spoof.predict(waveform, sr)
        # None when nothing is enrolled, so fusion drops the signal — same as the API.
        speaker = self.speaker.verify(waveform, sr) if self.speaker.is_enrolled else None

        mono = waveform.mean(dim=0).numpy().astype(np.float64)
        prosody = self.prosody.extract(mono, sr)

        result = self.scorer.compute(
            spoof_score=bonafide,
            speaker_score=speaker,
            prosody_score=prosody["prosody_score"],
        )
        return Row(
            path=path,
            label=label,
            duration_s=duration,
            bonafide=bonafide,
            speaker=speaker,
            naturalness=prosody["prosody_score"],
            risk=result.risk_score,
            anomalies=prosody.get("anomalies", []),
        )


def collect(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"No audio files in {folder}")
    return files


def describe(name: str, values: list[float], unit: str = "") -> None:
    if not values:
        return
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"  {name:<14} n={len(values):<3} mean={mean:6.1f}{unit}  sd={sd:5.1f}  "
        f"min={min(values):6.1f}  med={statistics.median(values):6.1f}  max={max(values):6.1f}"
    )


def equal_error_rate(bona: list[float], spoof: list[float]) -> tuple[float, float]:
    """
    EER over the fused risk score, and the threshold where it occurs.

    Convention here: HIGH risk = predicted spoof. FAR = genuine clips wrongly
    flagged; FRR = spoofs that slipped through.
    """
    if not bona or not spoof:
        return float("nan"), float("nan")

    best_gap, best_eer, best_t = float("inf"), float("nan"), float("nan")
    for t in [i * 0.5 for i in range(0, 201)]:
        far = sum(1 for b in bona if b >= t) / len(bona)     # false alarm on genuine
        frr = sum(1 for s in spoof if s < t) / len(spoof)    # missed spoof
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer, best_t = gap, (far + frr) / 2, t
    return best_eer * 100.0, best_t


def best_threshold(bona: list[float], spoof: list[float]) -> tuple[float, int, int]:
    """Threshold minimising total misclassifications; ties break toward stricter."""
    best = (float("inf"), 65.0, 0, 0)
    for t in [i * 0.5 for i in range(0, 201)]:
        fa = sum(1 for b in bona if b >= t)
        miss = sum(1 for s in spoof if s < t)
        if fa + miss < best[0]:
            best = (fa + miss, t, fa, miss)
    return best[1], best[2], best[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, nargs="+", help="score unlabelled folder(s)")
    parser.add_argument("--bonafide", type=Path, nargs="+", help="folder(s) of genuine speech")
    parser.add_argument("--spoof", type=Path, nargs="+", help="folder(s) of synthetic speech")
    parser.add_argument("--enroll", type=Path, help="reference clip to enable speaker matching")
    parser.add_argument("--csv", type=Path, help="write per-file rows here")
    parser.add_argument("--quiet", action="store_true", help="summary only")
    args = parser.parse_args()

    jobs: list[tuple[Path, str]] = []
    for label, folders in (
        ("unlabelled", args.folder),
        ("bonafide", args.bonafide),
        ("spoof", args.spoof),
    ):
        for folder in folders or []:
            jobs += [(p, label) for p in collect(folder)]
    if not jobs:
        parser.error("give --folder, or --bonafide and/or --spoof")

    pipeline = Pipeline(enroll_path=args.enroll)
    if not pipeline.spoof.is_calibrated:
        print("\n!! Spoof detector is UNCALIBRATED - these numbers are meaningless.")
        print("!! Run: python scripts/download_checkpoints.py\n")

    rows: list[Row] = []
    print(f"\nScoring {len(jobs)} file(s)...\n")
    if not args.quiet:
        print(f"{'file':<34}{'label':<11}{'dur':>6}{'bonafide':>10}{'natural':>9}{'risk':>7}")
        print("-" * 77)

    for path, label in jobs:
        try:
            row = pipeline.score(path, label)
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't end the run
            print(f"  [skip] {path.name}: {e}")
            continue
        rows.append(row)
        if not args.quiet:
            print(
                f"{path.name[:33]:<34}{label:<11}{row.duration_s:5.1f}s"
                f"{row.bonafide:10.3f}{row.naturalness:9.3f}{row.risk:7.1f}"
            )

    if not rows:
        return 1

    print("\n" + "=" * 77)
    for label in ("bonafide", "spoof", "unlabelled"):
        subset = [r for r in rows if r.label == label]
        if not subset:
            continue
        print(f"\n{label.upper()}  ({len(subset)} files)")
        describe("risk", [r.risk for r in subset])
        describe("AASIST bona", [r.bonafide * 100 for r in subset], "%")
        describe("naturalness", [r.naturalness * 100 for r in subset], "%")

    # Break out by source folder: a SAPI render and a clone of your own voice are
    # different attack classes, and averaging them hides which one is failing.
    sources = {r.path.parent.name for r in rows}
    if len(sources) > 1:
        print("\nBY SOURCE")
        for src in sorted(sources):
            subset = [r for r in rows if r.path.parent.name == src]
            label = subset[0].label
            risks = [r.risk for r in subset]
            bonas = [r.bonafide for r in subset]
            print(
                f"  {src:<20} [{label:<10}] n={len(subset):<3} "
                f"risk {min(risks):5.1f}-{max(risks):5.1f} (mean {statistics.mean(risks):5.1f})   "
                f"AASIST {min(bonas):.3f}-{max(bonas):.3f}"
            )

    bona = [r.risk for r in rows if r.label == "bonafide"]
    spoof = [r.risk for r in rows if r.label == "spoof"]

    current = pipeline.scorer.alert_threshold
    if bona or spoof:
        print(f"\nAT THE CONFIGURED THRESHOLD ({current:.0f})")
        if bona:
            fa = sum(1 for b in bona if b >= current)
            pct = fa / len(bona) * 100
            print(f"  genuine flagged as synthetic : {fa}/{len(bona)}  ({pct:.0f}%)")
        if spoof:
            miss = sum(1 for s in spoof if s < current)
            pct = miss / len(spoof) * 100
            print(f"  synthetic missed             : {miss}/{len(spoof)}  ({pct:.0f}%)")

    if bona and spoof:
        eer, eer_t = equal_error_rate(bona, spoof)
        t, fa, miss = best_threshold(bona, spoof)
        gap = min(spoof) - max(bona)
        print("\nSEPARATION")
        print(f"  highest genuine risk : {max(bona):.1f}")
        print(f"  lowest spoof risk    : {min(spoof):.1f}")
        print(f"  margin               : {gap:+.1f}  ({'clean split' if gap > 0 else 'OVERLAP'})")
        print(f"  EER                  : {eer:.1f}% at threshold {eer_t:.1f}")
        print(f"  best threshold       : {t:.1f}  ({fa} false alarms, {miss} misses)")
        print("\n  Set it in the UI under Settings -> Detection sensitivity.")
    elif bona and not spoof:
        print("\n  Only genuine speech given: this measures false alarms only.")
        print("  Add --spoof to find a threshold.")
    elif spoof and not bona:
        print("\n  Only synthetic speech given: this measures misses only.")
        print("  Add --bonafide (real recordings) to find a threshold.")

    if args.csv:
        import csv

        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "label", "duration_s", "bonafide", "speaker",
                        "naturalness", "risk", "anomalies"])
            for r in rows:
                w.writerow([r.path.name, r.label, f"{r.duration_s:.2f}", f"{r.bonafide:.4f}",
                            "" if r.speaker is None else f"{r.speaker:.4f}",
                            f"{r.naturalness:.4f}", f"{r.risk:.1f}",
                            "; ".join(r.anomalies)])
        print(f"\nWrote {len(rows)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
