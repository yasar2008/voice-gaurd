"""
Input-quality gate for the spoof detector.

The detector has a narrow operating envelope, established by measurement rather
than assumption (see scripts/robustness.py):

* **Noise.** Synthetic speech evades detection at roughly 25 dB SNR and below —
  quiet-office conditions. At 15 dB SNR a TTS render scores P(real) = 0.997,
  i.e. confidently "genuine". Genuine speech is unaffected, so noise produces
  *misses*, never false alarms. That is the dangerous direction: without a gate
  the app silently reports "Real voice" for everything in any normal room.
* **Bandwidth.** Band-limited to 4 kHz (telephone), genuine speech scores 0.000
  — real callers get flagged as synthetic.
* **Duration.** Below ~2 s the window is padded and the score degrades.

So rather than emit a confident verdict on audio the model cannot judge, measure
the input first and say so. A detector that refuses to answer is far more useful
than one that is confidently wrong.
"""

import numpy as np

from backend.config import config


def estimate_snr_db(samples: np.ndarray, sample_rate: int) -> float:
    """
    Estimate SNR from the distribution of short-frame energies.

    Speech is intermittent: loud frames are speech-plus-noise, the quietest
    frames are noise alone. Comparing a high percentile against a low one gives
    a usable estimate without needing a separate noise recording. Validated
    against known added noise in scripts/robustness.py.
    """
    if samples.size == 0:
        return 0.0

    frame = max(1, int(0.02 * sample_rate))  # 20 ms
    n_frames = samples.size // frame
    if n_frames < 5:
        return 0.0

    frames = samples[: n_frames * frame].reshape(n_frames, frame)
    energies = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    noise = float(np.percentile(energies, 10))
    signal = float(np.percentile(energies, 90))
    if signal <= 0:
        return 0.0
    # Noise floor sits inside the signal estimate; subtract it before the ratio.
    speech = max(signal**2 - noise**2, 1e-20)
    noise_power = max(noise**2, 1e-20)
    return float(10.0 * np.log10(speech / noise_power))


def high_frequency_energy_db(
    samples: np.ndarray, sample_rate: int, split_hz: float = 4000.0
) -> float:
    """
    Energy above `split_hz` relative to total, in dB — a band-limit detector.

    A 99%-rolloff measure does NOT work here: natural speech genuinely puts 99%
    of its energy below ~2.4 kHz, so rolloff calls clean full-band speech
    "narrowband". Measured separation on real clips:

        full band            -18 to -32 dB   (detector fine)
        resampled via 8 kHz  -41 to -48 dB   (detector still fine)
        resampled via 4 kHz  around -80 dB   (detector breaks: genuine -> 0.000)

    So the meaningful cut is far lower than intuition suggests, and only the
    severe telephone-band case is worth refusing.
    """
    if samples.size < 256:
        return 0.0
    window = min(samples.size, sample_rate * 4)
    seg = samples[:window].astype(np.float64) * np.hanning(window)
    power = np.abs(np.fft.rfft(seg)) ** 2
    freqs = np.fft.rfftfreq(window, 1.0 / sample_rate)
    total = power.sum()
    high = power[freqs >= split_hz].sum()
    if total <= 0:
        return 0.0
    return float(10.0 * np.log10(max(high, 1e-30) / max(total, 1e-30)))


def assess(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Judge whether this audio is inside the detector's reliable range.

    Returns a dict with the measurements, a `reliable` flag, and plain-language
    `reasons` when it is not. Callers should surface the reasons instead of a
    verdict rather than discarding them.
    """
    duration_s = samples.size / sample_rate if sample_rate else 0.0
    snr_db = estimate_snr_db(samples, sample_rate)
    hf_db = high_frequency_energy_db(samples, sample_rate)

    limits = config.quality
    reasons: list[str] = []

    if duration_s < limits.min_duration_s:
        reasons.append(
            f"only {duration_s:.1f}s of audio (needs {limits.min_duration_s:.0f}s)"
        )
    if snr_db < limits.min_snr_db:
        reasons.append(
            f"too noisy - {snr_db:.0f} dB SNR, below the {limits.min_snr_db:.0f} dB "
            f"the detector needs; synthetic speech hides in this much noise"
        )
    if hf_db < limits.min_high_freq_db:
        reasons.append(
            f"telephone-band audio - almost no energy above 4 kHz ({hf_db:.0f} dB); "
            f"genuine voices read as synthetic at this bandwidth"
        )

    return {
        "reliable": not reasons,
        "reasons": reasons,
        "snr_db": round(snr_db, 1),
        "high_freq_db": round(hf_db, 1),
        "duration_s": round(duration_s, 2),
    }
