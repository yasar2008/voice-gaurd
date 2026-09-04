"""
Speaker Verification using ECAPA-TDNN (SpeechBrain pretrained).

Provides enrollment (storing a reference speaker embedding) and verification
(comparing a new audio sample against the enrolled speaker via cosine similarity).

Model: ECAPA-TDNN trained on VoxCeleb1+2
    - Produces 192-dimensional speaker embeddings
    - Cosine similarity for verification scoring
    - Pretrained via SpeechBrain: speechbrain/spkrec-ecapa-voxceleb

Reference:
    Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation
    and Aggregation in TDNN Based Speaker Verification", Interspeech 2020.
"""

from typing import Optional

import numpy as np
import torch
import torchaudio

from backend.config import config


class SpeakerVerifier:
    """
    Speaker verification using pretrained ECAPA-TDNN.
    
    Workflow:
        1. Enroll a reference speaker by extracting their embedding from a voice sample
        2. Verify new audio samples by computing cosine similarity against enrollment
    
    If no speaker is enrolled, verification returns a neutral score (0.5).
    
    Usage:
        verifier = SpeakerVerifier()
        embedding = verifier.enroll(reference_audio, sr=16000)
        similarity = verifier.verify(test_audio, sr=16000, enrolled_embedding=embedding)
        # similarity ∈ [0.0, 1.0] — higher = more similar to enrolled speaker
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.device
        self.target_sr = config.audio.sample_rate
        self.embedding_dim = config.model.ecapa_embedding_dim
        self._model = None
        self._enrolled_embedding: Optional[np.ndarray] = None

    def _load_model(self):
        """Lazy-load the ECAPA-TDNN model from SpeechBrain."""
        if self._model is not None:
            return

        try:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy

            self._model = EncoderClassifier.from_hparams(
                source=config.model.ecapa_source,
                savedir=str(config.model.ecapa_savedir),
                run_opts={"device": self.device},
                local_strategy=LocalStrategy.COPY,
            )
            print(
                f"[SpeakerVerifier] Loaded ECAPA-TDNN from {config.model.ecapa_source}"
            )
        except Exception as e:
            print(
                f"[SpeakerVerifier] Failed to load ECAPA-TDNN: {e}. "
                f"Speaker verification will return neutral scores."
            )
            self._model = None

    def _preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Prepare audio for ECAPA-TDNN embedding extraction.
        
        Args:
            waveform: Audio tensor [channels, samples] or [samples]
            sample_rate: Original sample rate
            
        Returns:
            Preprocessed tensor [1, samples] at 16kHz mono
        """
        # Ensure 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16kHz
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)

        return waveform

    def _extract_embedding(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Extract a 192-dim speaker embedding from audio.
        
        Args:
            waveform: Audio tensor
            sample_rate: Sample rate
            
        Returns:
            Speaker embedding as numpy array [192,]
        """
        self._load_model()

        if self._model is None:
            # Return zero embedding if model failed to load
            return np.zeros(self.embedding_dim, dtype=np.float32)

        audio = self._preprocess(waveform, sample_rate)

        # SpeechBrain expects [batch, samples]
        with torch.no_grad():
            embedding = self._model.encode_batch(audio)

        # embedding shape: [1, 1, 192] → flatten to [192,]
        return embedding.squeeze().cpu().numpy()

    def enroll(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Enroll a reference speaker by extracting and storing their embedding.
        
        For best results, provide 5–15 seconds of clean speech from the target speaker.
        
        Args:
            waveform: Reference audio tensor
            sample_rate: Sample rate of the reference audio
            
        Returns:
            The enrolled speaker embedding [192,]
        """
        embedding = self._extract_embedding(waveform, sample_rate)
        self._enrolled_embedding = embedding
        print(
            f"[SpeakerVerifier] Enrolled speaker "
            f"(embedding norm: {np.linalg.norm(embedding):.4f})"
        )
        return embedding

    def enroll_from_multiple(
        self, waveforms: list[torch.Tensor], sample_rate: int
    ) -> np.ndarray:
        """
        Enroll from multiple audio clips (averages embeddings for robustness).
        
        Args:
            waveforms: List of audio tensors from the same speaker
            sample_rate: Common sample rate
            
        Returns:
            Averaged speaker embedding [192,]
        """
        embeddings = [self._extract_embedding(w, sample_rate) for w in waveforms]
        avg_embedding = np.mean(embeddings, axis=0)
        # L2-normalize the averaged embedding
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        self._enrolled_embedding = avg_embedding
        print(
            f"[SpeakerVerifier] Enrolled speaker from {len(waveforms)} clips "
            f"(averaged embedding norm: {np.linalg.norm(avg_embedding):.4f})"
        )
        return avg_embedding

    def verify(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        enrolled_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Verify whether audio matches the enrolled speaker.
        
        Args:
            waveform: Test audio tensor
            sample_rate: Sample rate
            enrolled_embedding: Optional override for stored enrollment.
                                If None, uses self._enrolled_embedding.
            
        Returns:
            Similarity score ∈ [0.0, 1.0]
            - 1.0 = very high confidence same speaker
            - 0.0 = definitely different speaker
            - 0.5 = neutral (no enrolled speaker or model unavailable)
        """
        ref_embedding = enrolled_embedding if enrolled_embedding is not None else self._enrolled_embedding

        if ref_embedding is None:
            # No enrolled speaker — return neutral score
            return 0.5

        test_embedding = self._extract_embedding(waveform, sample_rate)

        # Cosine similarity
        dot = np.dot(ref_embedding, test_embedding)
        norm_ref = np.linalg.norm(ref_embedding)
        norm_test = np.linalg.norm(test_embedding)

        if norm_ref == 0 or norm_test == 0:
            return 0.5

        cosine_sim = dot / (norm_ref * norm_test)

        # Map from [-1, 1] to [0, 1]
        similarity = (cosine_sim + 1.0) / 2.0
        return float(np.clip(similarity, 0.0, 1.0))

    def clear_enrollment(self) -> None:
        """Clear the enrolled speaker embedding."""
        self._enrolled_embedding = None
        print("[SpeakerVerifier] Enrollment cleared")

    @property
    def is_enrolled(self) -> bool:
        """Whether a speaker has been enrolled."""
        return self._enrolled_embedding is not None

    @property
    def enrolled_embedding(self) -> Optional[np.ndarray]:
        """Get the currently enrolled embedding (or None)."""
        return self._enrolled_embedding
