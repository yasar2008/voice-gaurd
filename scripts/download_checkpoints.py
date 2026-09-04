"""
Download the pretrained AASIST-L checkpoint.

Without this checkpoint the spoof detector runs on randomly initialised weights
and its scores mean nothing, so this is a required setup step after cloning.

Usage:
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --force   # re-download

Source: https://github.com/clovaai/aasist (MIT, © NAVER Corp.)
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AASIST_L_URL = (
    "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth"
)
# SHA-256 of the released AASIST-L.pth, so a corrupted or substituted download
# fails loudly instead of silently degrading detection quality.
AASIST_L_SHA256 = "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"
AASIST_L_PATH = PROJECT_ROOT / "checkpoints" / "AASIST-L.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha256: str, force: bool = False) -> bool:
    """Download `url` to `dest`, verifying its checksum. Returns True on success."""
    if dest.exists() and not force:
        actual = sha256(dest)
        if actual == expected_sha256:
            print(f"[skip] {dest.name} already present and verified.")
            return True
        print(f"[warn] {dest.name} exists but checksum differs — re-downloading.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    print(f"[get ] {url}")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
    except Exception as e:  # noqa: BLE001 — surface any network/IO failure plainly
        tmp.unlink(missing_ok=True)
        print(f"[fail] Could not download {url}: {e}")
        return False

    actual = sha256(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        print("[fail] Checksum mismatch.")
        print(f"       expected {expected_sha256}")
        print(f"       got      {actual}")
        return False

    tmp.replace(dest)
    size_kb = dest.stat().st_size / 1024
    print(f"[ok  ] {dest} ({size_kb:.0f} KB, sha256 verified)")
    return True


def fetch_wav2vec() -> bool:
    """Warm the HuggingFace cache for the default spoof model."""
    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        from backend.config import config

        model_id = config.model.wav2vec_model_id
        print(f"[get ] {model_id}")
        AutoFeatureExtractor.from_pretrained(model_id)
        AutoModelForAudioClassification.from_pretrained(model_id)
        print(f"[ok  ] {model_id} cached")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[fail] Could not fetch the wav2vec2 model: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    ok = fetch_wav2vec()
    # AASIST-L is no longer the default detector, but keep it available for the
    # A/B comparison behind VCD_MODEL__SPOOF_BACKEND=aasist.
    ok = download(AASIST_L_URL, AASIST_L_PATH, AASIST_L_SHA256, force=args.force) and ok
    if ok:
        print("\nSpoof detector is ready. Start the backend with:")
        print("    python scripts/start_demo.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
