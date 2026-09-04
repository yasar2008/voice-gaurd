import os, sys, time
sys.path.insert(0, ".")
from pathlib import Path
from scripts.fetch_training_data import fetch_generator
print("token visible to python:", bool(os.environ.get("HF_TOKEN")))
t0 = time.time()
gen, n = fetch_generator("Chatterbox", "en", 8, Path("D:/ml-data/authtest"))
print(f"{gen}: {n} clips in {time.time()-t0:.1f}s")
