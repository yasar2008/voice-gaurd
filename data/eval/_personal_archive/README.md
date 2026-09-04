# Personal recordings — out of the evaluation path deliberately

Three clips supplied during development: one raw phone recording, the same
recording after Audacity noise reduction, and an ElevenLabs clone of that voice.

They are kept here rather than in the benchmark folders because n=1 per class
cannot support a decision about a model. Judging a detector on a single clone
means a coin flip at a 70% catch rate reads as total failure, and that was
actively distorting the comparisons.

Use the corpora instead: MLAAD held-out generators for synthesis, LibriSpeech
and jay15k for genuine speech, and data/eval/control_real for cross-corpus
genuine checks.
