"""
Tests for the speaker verification module.

Note: These tests use mock embeddings to avoid downloading the full
ECAPA-TDNN model during CI. The _extract_embedding method is tested
indirectly through the verify() interface.
"""

import numpy as np
import torch
import pytest

from backend.models.speaker_verify import SpeakerVerifier


class TestSpeakerVerifier:
    """Test speaker verification logic (with mocked embeddings)."""

    def setup_method(self):
        self.verifier = SpeakerVerifier()

    def test_no_enrollment_returns_neutral(self):
        """Without enrollment, verification should return 0.5 (neutral)."""
        waveform = torch.randn(1, 16000 * 3)
        score = self.verifier.verify(waveform, sample_rate=16000)
        assert score == 0.5

    def test_enrollment_stores_embedding(self):
        """After enrollment, is_enrolled should be True."""
        # Manually set an embedding (skip model loading)
        self.verifier._enrolled_embedding = np.random.randn(192).astype(np.float32)
        assert self.verifier.is_enrolled

    def test_clear_enrollment(self):
        """Clearing enrollment should reset state."""
        self.verifier._enrolled_embedding = np.random.randn(192).astype(np.float32)
        assert self.verifier.is_enrolled
        self.verifier.clear_enrollment()
        assert not self.verifier.is_enrolled

    def test_cosine_similarity_same_embedding(self):
        """Verifying with the same embedding should give high similarity."""
        embedding = np.random.randn(192).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        self.verifier._enrolled_embedding = embedding

        # Mock _extract_embedding to return the same embedding
        original_extract = self.verifier._extract_embedding
        self.verifier._extract_embedding = lambda w, sr: embedding

        waveform = torch.randn(1, 16000 * 3)
        score = self.verifier.verify(waveform, sample_rate=16000)

        assert score > 0.9, f"Same embedding should give high score, got {score}"

        # Restore
        self.verifier._extract_embedding = original_extract

    def test_cosine_similarity_orthogonal_embeddings(self):
        """Orthogonal embeddings should give neutral similarity (~0.5)."""
        emb1 = np.zeros(192, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(192, dtype=np.float32)
        emb2[1] = 1.0

        self.verifier._enrolled_embedding = emb1
        self.verifier._extract_embedding = lambda w, sr: emb2

        score = self.verifier.verify(torch.randn(1, 16000), sample_rate=16000)
        assert abs(score - 0.5) < 0.01, f"Orthogonal embeddings should give ~0.5, got {score}"

    def test_cosine_similarity_opposite_embeddings(self):
        """Opposite embeddings should give low similarity (~0.0)."""
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        self.verifier._enrolled_embedding = emb
        self.verifier._extract_embedding = lambda w, sr: -emb

        score = self.verifier.verify(torch.randn(1, 16000), sample_rate=16000)
        assert score < 0.1, f"Opposite embeddings should give low score, got {score}"

    def test_verify_with_explicit_enrollment(self):
        """verify() with explicit enrolled_embedding should override stored one."""
        stored = np.ones(192, dtype=np.float32)
        explicit = np.zeros(192, dtype=np.float32)
        explicit[0] = 1.0

        self.verifier._enrolled_embedding = stored
        test_emb = np.zeros(192, dtype=np.float32)
        test_emb[0] = 1.0
        self.verifier._extract_embedding = lambda w, sr: test_emb

        # Using explicit embedding (matches test)
        score_explicit = self.verifier.verify(
            torch.randn(1, 16000), 16000, enrolled_embedding=explicit
        )
        # Using stored embedding (all ones, different from test)
        score_stored = self.verifier.verify(torch.randn(1, 16000), 16000)

        assert score_explicit > score_stored

    def test_embedding_dim(self):
        """Embedding dimensionality should match config (192)."""
        assert self.verifier.embedding_dim == 192
