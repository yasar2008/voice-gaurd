# Architecture

## System Overview

The Voice Clone Detector fuses three independent detection signals into a single, explainable risk score:

```
                                    ┌─────────────────────────┐
                                    │   AASIST-L (85k params) │
                                    │   Spoof Detection       │──── bonafide_score
                                    │   Raw waveform → logits │     [0, 1]
                                    └─────────────────────────┘
                                                │
Audio Input ──→ [Preprocessing] ──→─────────────┤
   (16kHz mono PCM)                             │
                                    ┌─────────────────────────┐
                                    │   ECAPA-TDNN (14M params)│
                                    │   Speaker Verification  │──── speaker_score
                                    │   Embedding → cosine sim│     [0, 1]
                                    └─────────────────────────┘
                                                │
                                    ┌─────────────────────────┐
                                    │   Prosody Analyzer      │
                                    │   Parselmouth + librosa │──── prosody_score
                                    │   Jitter/Shimmer/F0/HNR│     [0, 1]
                                    └─────────────────────────┘
                                                │
                                                ▼
                                    ┌─────────────────────────┐
                                    │   Fusion Layer          │
                                    │   Weighted combination  │──── risk_score
                                    │   Explainable breakdown │     [0, 100]
                                    └─────────────────────────┘
```

## Component Details

### 1. AASIST-L (Spoof Detection)

- **Architecture**: SincConv front-end → Convolutional encoder → Graph Attention Network → Readout
- **Input**: Raw 16kHz waveform, 64,600 samples (~4.04 seconds)
- **Output**: Bonafide confidence score [0, 1]
- **Training data**: ASVspoof 2019 LA (bonafide vs. TTS/VC attacks)
- **Why AASIST-L**: ~85k parameters — small enough for real-time CPU inference while maintaining competitive EER

### 2. ECAPA-TDNN (Speaker Verification)

- **Architecture**: ECAPA-TDNN pretrained on VoxCeleb 1+2
- **Embedding**: 192-dimensional speaker representation
- **Scoring**: Cosine similarity between enrolled reference and test audio
- **Enrollment**: Average embedding from one or more reference clips
- **Neutral mode**: Returns 0.5 when no speaker is enrolled (doesn't bias risk score)

### 3. Prosody Analyzer

- **Tool**: Parselmouth (Praat wrapper) for clinical-grade voice features
- **Features extracted**:
  - Pitch (F0) contour statistics — TTS often has unnaturally smooth F0
  - Local jitter — micro-pitch perturbations absent in synthetic speech
  - Local shimmer — micro-amplitude perturbations absent in TTS
  - Harmonics-to-Noise Ratio (HNR) — different noise profiles
  - Pause ratio — energy-based VAD for silence detection
- **Scoring**: Rule-based anomaly detection against reference ranges from bonafide speech

### 4. Fusion Layer

- **Method**: Weighted linear combination (not learned — by design)
- **Weights**: spoof_detection=0.50, speaker_similarity=0.30, prosody_naturalness=0.20
- **Output**: Risk score [0, 100] with per-component breakdown
- **Confidence**: Based on signal agreement (all signals agree = high confidence)
- **Why not learned fusion**: Explainability. Each component's contribution is visible.

## Real-Time Streaming Architecture

```
Browser                          Server
  │                                │
  │  getUserMedia (mic)            │
  │  → AudioContext                │
  │  → ScriptProcessor             │
  │  → PCM 16-bit chunks           │
  │                                │
  │  ════ WebSocket ═══════════►   │
  │  binary PCM (16kHz, mono)      │
  │                                │  → AudioRingBuffer (max 10s)
  │                                │  → Every 3s: run full pipeline
  │                                │     1. AASIST-L on last 4s
  │                                │     2. ECAPA-TDNN cosine sim
  │                                │     3. Prosody extraction
  │                                │     4. Fusion → risk score
  │                                │
  │  ◄═══════════════════════════  │
  │  JSON risk update              │
  │  {risk_score, breakdown, ...}  │
  │                                │
  │  → Update waveform viz         │
  │  → Update risk meter           │
  │  → Show/hide alert banner      │
```

## Latency Budget

| Stage | Target | Notes |
|---|---|---|
| Audio chunk accumulation | 3,000 ms | Configurable via `CHUNK_DURATION_S` |
| AASIST-L inference | ~50 ms (CPU) | 85k params, single forward pass |
| ECAPA-TDNN embedding | ~100 ms (CPU) | 14M params, but only extracts embedding |
| Prosody extraction | ~30 ms | Praat is highly optimized for speech |
| Fusion computation | <1 ms | Simple arithmetic |
| WebSocket round-trip | ~5 ms | Local network |
| **Total (per update)** | **~3,200 ms** | Dominated by chunk accumulation |

## Privacy

All inference runs locally — no audio data leaves the user's machine.
- Models are loaded from local checkpoints (no API calls)
- WebSocket runs on localhost in the demo configuration
- No audio is stored, logged, or transmitted externally
