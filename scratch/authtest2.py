import os, sys, urllib.request, urllib.parse
sys.path.insert(0, ".")
from scripts.fetch_training_data import api, REPO
tok = os.environ.get("HF_TOKEN")
files = [x["path"] for x in api("fake/en/Chatterbox") if x.get("type") == "file"][:2]
print("listed files:", len(files))
rel = files[0]
url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{urllib.parse.quote(rel, safe='/')}"
for label, hdrs in [("no auth", {"User-Agent": "Mozilla/5.0"}),
                    ("with token", {"User-Agent": "Mozilla/5.0", "Authorization": f"Bearer {tok}"})]:
    try:
        req = urllib.request.Request(url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"  {label:<12} HTTP {r.status}  {len(r.read())} bytes")
    except Exception as e:
        print(f"  {label:<12} {type(e).__name__}: {str(e)[:90]}")
