"""
Fetch a training corpus: synthetic speech from MLAAD, genuine speech from LibriSpeech.
Fast parallel direct download without hf_hub lock contention.
"""

import argparse
import json
import os
import sys
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("D:/ml-data")
REPO = "mueller91/MLAAD"
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"

HOLDOUT_GENERATORS = [
    "ElevenLabs-Turbo-v2.5",
    "ElevenLabs-v2-Multilingual",
    "ElevenLabs-v3",
    "Edge-TTS",
    "ChatTTS",
]


def _headers() -> dict:
    """
    MLAAD is a gated dataset (gated: auto). Without the token the resolve
    endpoint returns 401; with a token but no accepted licence it returns 403.
    Both look like generic failures if the header is omitted, which is how an
    earlier version of this script silently downloaded nothing.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def api(path: str):
    quoted = urllib.parse.quote(path, safe="/")
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{quoted}"
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def list_generators(lang: str = "en") -> list[str]:
    return sorted(x["path"].split("/")[-1] for x in api(f"fake/{lang}") if x["type"] == "directory")


def download_single_file(rel_path: str, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 1000:
        return True
    try:
        quoted = urllib.parse.quote(rel_path, safe="/")
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{quoted}"
        req = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return True
    except Exception as e:
        download_single_file.last_error = repr(e)[:120]
        return False


def fetch_generator(gen: str, lang: str, limit: int, dest: Path) -> tuple[str, int]:
    """Download up to `limit` clips for one generator using direct parallel HTTP."""
    try:
        entries = api(f"fake/{lang}/{gen}")
        files = [x["path"] for x in entries if x.get("type") == "file"]
    except Exception as e:
        print(f"  [error listing] {gen}: {e}")
        return gen, 0

    files = [f for f in files if f.lower().endswith((".wav", ".flac", ".mp3"))][:limit]
    out = dest / gen
    out.mkdir(parents=True, exist_ok=True)

    got = 0
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for rel in files:
            target = out / Path(rel).name
            futures.append(pool.submit(download_single_file, rel, target))
        for f in as_completed(futures):
            if f.result():
                got += 1

    return gen, got


def fetch_librispeech(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.glob("*.flac"))
    if existing:
        print(f"[skip] LibriSpeech already present ({len(existing)} files)")
        return len(existing)

    archive = DATA_ROOT / "test-clean.tar.gz"
    if not archive.exists():
        print(f"[get ] {LIBRISPEECH_URL} (~346 MB)")
        urllib.request.urlretrieve(LIBRISPEECH_URL, archive)

    print("[    ] extracting genuine speech ...")
    n = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            (dest / Path(member.name).name).write_bytes(src.read())
            n += 1
    archive.unlink(missing_ok=True)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-generator", type=int, default=200)
    parser.add_argument("--generators", type=int, default=24, help="how many systems to sample")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    gens = list_generators(args.lang)
    print(f"\n{len(gens)} generators available for '{args.lang}'")

    holdout = [g for g in HOLDOUT_GENERATORS if g in gens]
    train_pool = [g for g in gens if g not in holdout]
    chosen_train = train_pool[: max(0, args.generators - len(holdout))]

    print(f"\nTRAIN generators ({len(chosen_train)}):")
    print("   " + ", ".join(chosen_train))
    print(f"\nHOLDOUT generators ({len(holdout)}) -- never trained on:")
    print("   " + ", ".join(holdout))
    est = (len(chosen_train) + len(holdout)) * args.per_generator * 0.3
    print(f"\nEstimated synthetic download: ~{est:.0f} MB  (+346 MB genuine)")
    if args.list_only:
        return 0

    fake_root = DATA_ROOT / "mlaad" / args.lang
    for group, names in (("train", chosen_train), ("holdout", holdout)):
        dest = fake_root / group
        print(f"\nFetching {group} generators ({len(names)}) -> {dest}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_generator, g, args.lang, args.per_generator, dest): g
                for g in names
            }
            for fut in as_completed(futures):
                gen, n = fut.result()
                print(f"  {gen[:38]:<40} {n:>4} clips")

    print("\nFetching genuine speech (LibriSpeech test-clean)")
    n_real = fetch_librispeech(DATA_ROOT / "genuine" / "librispeech")
    print(f"  {n_real} genuine clips")

    n_fake = len(list(fake_root.rglob("*.wav"))) + len(list(fake_root.rglob("*.flac")))
    print(f"\nReady: {n_fake} synthetic, {n_real} genuine under {DATA_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
