# Model checkpoints (gitignored)

## Required

`AASIST-L.pth` — AASIST-L trained on ASVspoof 2019 LA (~416 KB, ~85k params).

Fetch it with:

    python scripts/download_checkpoints.py

The script downloads from the official release
(https://github.com/clovaai/aasist, MIT © NAVER Corp.) and verifies its SHA-256.

## Why it matters

Without this file the spoof detector falls back to randomly initialised weights.
The API still runs, but every spoof score is noise. That state is reported as
`spoof_detector.calibrated = false` in `GET /health`, printed at startup, and
shown as a warning banner in the UI — it is never presented as a real verdict.
