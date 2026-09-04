# Cloned / synthetic versions of a real voice

Put AI-generated audio that imitates a specific person here — the output of a
voice-cloning service, or TTS trained on someone's voice.

Kept separate from `spoof_tts/` (generic Windows SAPI renders) on purpose: those
are two different attack classes, and `calibrate.py` reports each source folder
separately so a failure on one doesn't hide behind the other.

## Watch the channel confound

The most common way to get a misleading result: your genuine recordings come off
a laptop mic in a real room, while the clone arrives as clean studio-quality
audio from a cloud service. The detector can then separate the two on *recording
channel* rather than on real-vs-synthetic, and you would be measuring the wrong
thing — flattering numbers that collapse the moment a clone reaches you over a
phone call with matching noise.

Two ways to keep it honest:

- **Match the channel.** Play the clone through a speaker and re-record it on the
  same microphone you used for the genuine clips. That mirrors the real attack
  path (a clone arriving over a call) and puts both classes on equal footing.
  Note the browser's echo cancellation is already disabled for analysis, but for
  this you want a separate recording, not the live view.
- **Or keep both and label them.** `clone_clean/` and `clone_rerecorded/` as
  subfolders, then pass the parent to `--spoof`. Each is reported on its own row.

Level is no longer part of this confound: audio is RMS-normalised before spoof
inference, so loudness alone can't separate the classes.
