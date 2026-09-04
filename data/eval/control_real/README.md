# Known-genuine speech (control set)

Real human recordings used to check that the detector does not flag authentic
speech. Pulled from public Apache-2.0 SpeechBrain model repos on HuggingFace
(LibriSpeech / CommonVoice / IEMOCAP / CommonLanguage examples).

This set exists because without it you cannot tell a broken model from a broken
pipeline. Adding it is what exposed the original failure: AASIST-L scored these
genuine clips at 0.152 mean P(real) — it was calling real voices fake.

Two known-awkward members, kept deliberately:

- `example2_spkrec-ecapa.flac` — only 2.07 s, so it gets tile-padded to fill the
  4.04 s window. It flags. Short clips are a genuine weak spot; the live path
  avoids it by refusing to score until 4 s of speech has accumulated.
- `vb_metric.wav` — the example from a speech-*enhancement* repo, i.e. probably
  GAN-processed audio. Flagging it may well be correct behaviour rather than a
  false alarm, which is why it is not counted as a clean miss.

Regenerate or extend with any permissively-licensed genuine speech. Duplicates
were removed: several repos ship the identical example file.
