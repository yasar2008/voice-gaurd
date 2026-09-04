"""
Audio decoding helper.

torchaudio 2.9+ delegates `torchaudio.load()` to TorchCodec, which needs FFmpeg
shared libraries installed system-wide — not something a Windows demo box
usually has, and its absence turns every file upload into a 500. libsndfile
(via `soundfile`, already a dependency and shipped as a self-contained wheel)
decodes WAV/FLAC/OGG without any of that, so it is tried first and torchaudio is
kept as the fallback for anything libsndfile refuses.
"""

from __future__ import annotations

from typing import BinaryIO

import torch

AudioSource = str | BinaryIO


def load_audio(source: AudioSource) -> tuple[torch.Tensor, int]:
    """
    Decode an audio file into a float32 tensor.

    Args:
        source: Filesystem path, or a binary file-like object (e.g. BytesIO of
            an uploaded file).

    Returns:
        (waveform, sample_rate) where waveform is [channels, samples] float32.

    Raises:
        RuntimeError: if neither backend can decode the input.
    """
    errors: list[str] = []

    try:
        import soundfile as sf

        if hasattr(source, "seek"):
            source.seek(0)
        data, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        # soundfile gives [samples, channels]; torchaudio convention is the transpose.
        waveform = torch.from_numpy(data).transpose(0, 1).contiguous()
        return waveform, int(sample_rate)
    except Exception as e:  # noqa: BLE001 — fall through to the next backend
        errors.append(f"soundfile: {e}")

    try:
        import torchaudio

        if hasattr(source, "seek"):
            source.seek(0)
        waveform, sample_rate = torchaudio.load(source)
        return waveform, int(sample_rate)
    except Exception as e:  # noqa: BLE001 — try the last backend before giving up
        errors.append(f"torchaudio: {e}")

    try:
        return _load_via_ffmpeg(source)
    except Exception as e:  # noqa: BLE001 — report every failure together
        errors.append(f"ffmpeg: {e}")

    raise RuntimeError("; ".join(errors))


def _load_via_ffmpeg(source: AudioSource) -> tuple[torch.Tensor, int]:
    """
    Decode with the ffmpeg binary bundled by imageio-ffmpeg.

    Covers the containers libsndfile refuses — chiefly M4A/AAC, which is what
    Windows Voice Recorder produces and what half the originals in the Kaggle
    deepfake set are stored as. Decodes to 32-bit float PCM on stdout so nothing
    is quantised on the way in.
    """
    import subprocess
    import tempfile

    import imageio_ffmpeg
    import numpy as np

    tmp_path = None
    try:
        if hasattr(source, "seek"):
            source.seek(0)
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(source.read())
                tmp_path = tmp.name
            uri = tmp_path
        else:
            uri = str(source)

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        probe = subprocess.run(
            [exe, "-i", uri, "-hide_banner"], capture_output=True, text=True, check=False
        )
        rate = 16000
        for line in probe.stderr.splitlines():
            if "Audio:" in line and " Hz" in line:
                for part in line.split(","):
                    if part.strip().endswith("Hz"):
                        rate = int(part.strip().split()[0])
                        break
                break

        result = subprocess.run(
            [exe, "-v", "error", "-i", uri, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-"],
            capture_output=True,
            check=True,
        )
        samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if samples.size == 0:
            raise RuntimeError("decoded zero samples")
        return torch.from_numpy(samples).unsqueeze(0), rate
    finally:
        if tmp_path:
            import os

            os.unlink(tmp_path)
