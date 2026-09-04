"""
Generate sample audio fixtures for testing and quick local evaluation.

Creates:
1. tests/fixtures/bonafide_human_sim.wav - Simulated voiced human audio with natural pitch variation, jitter, shimmer
2. tests/fixtures/tts_clone_sim.wav - Simulated TTS/vocoder clone with robotic pitch flatness and low jitter
3. tests/fixtures/different_speaker_sim.wav - Different speaker reference for verification tests
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DURATION_S = 4.5


def generate_bonafide_sample(filepath: Path):
    """Generate audio with natural human-like pitch glide, jitter, and shimmer."""
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    
    # Natural F0 contour varying between 140Hz and 190Hz (intonation phrase)
    f0 = 160.0 + 25.0 * np.sin(2 * np.pi * 0.8 * t) + 10.0 * np.sin(2 * np.pi * 2.2 * t)
    
    # Add micro-jitter (pitch perturbation ~1.5%)
    jitter_noise = 0.015 * np.random.randn(len(t))
    f0_jittered = f0 * (1.0 + jitter_noise)
    
    # Phase integration
    phase = 2 * np.pi * np.cumsum(f0_jittered) / SR
    
    # Glottal source pulse approximation (harmonics with roll-off)
    waveform = np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.35 * np.sin(3 * phase) + 0.2 * np.sin(4 * phase)
    
    # Micro-shimmer (amplitude perturbation ~5%)
    shimmer = 1.0 + 0.05 * np.random.randn(len(t))
    waveform *= shimmer
    
    # Natural amplitude envelope (vocal attack, body, decay with speech pauses)
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / DURATION_S))
    # Add short natural breath pause around 2.2s - 2.5s
    pause_mask = ~((t > 2.1) & (t < 2.5))
    env *= pause_mask.astype(float)
    
    waveform = waveform * env
    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6) * 0.85
    
    sf.write(str(filepath), waveform.astype(np.float32), SR)
    print(f"Generated bonafide speech sample: {filepath}")


def generate_tts_clone_sample(filepath: Path):
    """Generate audio with synthetic TTS characteristics (unnaturally flat F0, zero jitter, vocoder artifacts)."""
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    
    # Unnaturally flat F0 (typical of basic TTS without expressive prosody)
    f0 = 150.0 * np.ones_like(t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    
    # Perfect square-like harmonic content with high-frequency vocoder buzz
    waveform = np.sin(phase) + 0.7 * np.sin(2 * phase) + 0.5 * np.sin(3 * phase) + 0.4 * np.sin(4 * phase)
    
    # High frequency vocoder artifact (>6kHz metallic hiss)
    vocoder_artifact = 0.08 * np.sin(2 * np.pi * 7200 * t) * np.random.randn(len(t))
    waveform += vocoder_artifact
    
    # Mechanical rectangular gate envelope (no natural micro-shimmer)
    env = np.ones_like(t)
    env[:int(0.05 * SR)] = np.linspace(0, 1, int(0.05 * SR))
    env[-int(0.05 * SR):] = np.linspace(1, 0, int(0.05 * SR))
    
    waveform = waveform * env
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6) * 0.85
    
    sf.write(str(filepath), waveform.astype(np.float32), SR)
    print(f"Generated synthetic clone sample: {filepath}")


def generate_speaker_b_sample(filepath: Path):
    """Generate sample from a distinct speaker (higher pitch ~240Hz female/child voice range)."""
    t = np.linspace(0, DURATION_S, int(SR * DURATION_S), endpoint=False)
    f0 = 230.0 + 20.0 * np.sin(2 * np.pi * 1.2 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    
    waveform = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.3 * np.sin(3 * phase)
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / DURATION_S))
    waveform = waveform * env
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-6) * 0.85
    
    sf.write(str(filepath), waveform.astype(np.float32), SR)
    print(f"Generated speaker B reference sample: {filepath}")


if __name__ == "__main__":
    generate_bonafide_sample(FIXTURES_DIR / "bonafide_human_sim.wav")
    generate_tts_clone_sample(FIXTURES_DIR / "tts_clone_sim.wav")
    generate_speaker_b_sample(FIXTURES_DIR / "different_speaker_sim.wav")
    print("All sample audio fixtures successfully created!")
