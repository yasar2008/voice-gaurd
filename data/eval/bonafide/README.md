# Genuine speech goes here

Drop 5-10 recordings of a **real human voice** in this folder. Anything the
loader can read works: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`.

Guidelines that matter for the measurement:

- **10+ seconds each.** The spoof model needs ~4 s minimum; longer is better.
- **Use the same microphone you use with the app.** The whole point is to measure
  the detector on your actual recording setup, not on studio audio.
- **Speak normally.** Don't over-enunciate or perform — capture the everyday case.
- **A few different conditions helps**: quiet room, a bit of background noise,
  close to the mic, further away.

**Easiest route:** in the app, go to *Settings -> Build a calibration set* and hit
"Record my voice". It captures 16 kHz WAV through the identical pipeline the live
analyser uses, drops the file here, and scores it on the spot.

Note: Windows Voice Recorder writes **M4A**, which libsndfile cannot decode - those
files will be skipped. Convert to WAV first, or just use the in-app recorder.

Then, from the project root:

    python scripts/calibrate.py --bonafide data/eval/bonafide --spoof data/eval/spoof_tts

That reports how often genuine speech is wrongly flagged, the separation margin
against the synthetic set, the EER, and a threshold fitted to your voice and
microphone rather than the default 65 someone wrote in a config file.

This folder is gitignored — recordings of your voice stay local.
