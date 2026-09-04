"""Voice Clone Detector — Backend Package."""

import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

if Path("D:/").exists():
    os.environ.setdefault("HF_HOME", "D:/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "D:/hf_cache/hub")
    os.environ.setdefault("HF_HUB_CACHE", "D:/hf_cache/hub")
    os.environ.setdefault("TORCH_HOME", "D:/torch_cache")
else:
    cache_dir = Path.home() / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


