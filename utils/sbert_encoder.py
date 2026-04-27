"""
SBERT Encoder — PubMedBERT-based semantic embeddings
=====================================================
Model priority:
  1. pritamdeka/S-PubMedBert-MS-MARCO         (PubMed SBERT, best for clinical text)
  2. microsoft/BiomedNLP-PubMedBERT-base-...  (raw BiomedBERT, mean-pool fallback)
  3. all-MiniLM-L6-v2                         (general SBERT, last resort)

Lazy-loaded on first call. If sentence-transformers / torch is unavailable in
the current runtime (e.g. CUDA bus-error in a constrained sandbox), the module
falls back to a TF-IDF + cosine similarity encoder so the rest of the pipeline
never breaks.

All embeddings are L2-normalised before returning, making dot-product ==
cosine similarity — compatible with faiss.IndexFlatIP.
"""

from __future__ import annotations
import re
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# ── Model preference order ────────────────────────────────────────────────────
_MODEL_CANDIDATES = [
    "pritamdeka/S-PubMedBert-MS-MARCO",            # PubMed SBERT (preferred)
    "NthIterations/mini-biobert",                   # lighter biomedical SBERT
    "all-MiniLM-L6-v2",                             # general fallback
]

_encoder = None          # SentenceTransformer instance or TfidfFallback
_dim: int | None = None  # embedding dimension


# ── TF-IDF fallback (no torch required) ──────────────────────────────────────
class _TfidfFallback:
    """
    Pure-numpy drop-in for SentenceTransformer when torch/sklearn are unavailable.
    Tokenises text into character n-gram hashes, applies TF weighting, then
    projects to DIM dimensions via a random Gaussian projection (stable across
    calls because the RNG is seeded from the vocabulary size).
    Produces DIM-d L2-normalised float32 vectors.
    """
    DIM = 512

    def __init__(self):
        self._vocab: dict[str, int] = {}   # token -> column index
        self._fitted = False
        self._proj: np.ndarray | None = None  # (vocab_size, DIM) projection matrix

    # ── tokeniser ─────────────────────────────────────────────────────────
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    # ── sparse TF vector (as dense, capped at vocab size) ─────────────────
    def _tf_vector(self, text: str) -> np.ndarray:
        tokens = self._tokenise(text)
        vec = np.zeros(len(self._vocab), dtype="float32")
        for tok in tokens:
            if tok in self._vocab:
                vec[self._vocab[tok]] += 1.0
        # Sublinear TF
        nz = vec > 0
        vec[nz] = 1.0 + np.log(vec[nz])
        return vec

    def fit(self, texts: list[str]):
        for text in texts:
            for tok in self._tokenise(text):
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
        v = len(self._vocab)
        rng = np.random.RandomState(42 + v)
        # Random Gaussian projection (Johnson-Lindenstrauss)
        self._proj = (rng.randn(v, self.DIM) / np.sqrt(self.DIM)).astype("float32")
        self._fitted = True

    def encode(self, texts: list[str] | str,
               convert_to_numpy: bool = True,
               normalize_embeddings: bool = True,
               show_progress_bar: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not self._fitted:
            self.fit(texts)
        vecs = []
        for text in texts:
            tf = self._tf_vector(text)
            # Project to DIM; pad/trim if vocab grew after projection built
            v = min(len(tf), self._proj.shape[0])
            projected = tf[:v] @ self._proj[:v]
            vecs.append(projected)
        out = np.stack(vecs).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            out = out / norms
        return out


def _try_load_sbert() -> tuple[object, int] | None:
    """Attempt to import sentence_transformers and load the best available model."""
    try:
        from sentence_transformers import SentenceTransformer
        for model_name in _MODEL_CANDIDATES:
            try:
                logger.info(f"[SBERT] Loading model: {model_name}")
                model = SentenceTransformer(model_name)
                # Quick smoke-test (catches bus-errors / CUDA issues early)
                _ = model.encode(["test"], convert_to_numpy=True,
                                  normalize_embeddings=True)
                dim = model.get_sentence_embedding_dimension()
                logger.info(f"[SBERT] Loaded '{model_name}', dim={dim}")
                return model, dim
            except Exception as e:
                logger.warning(f"[SBERT] Failed to load '{model_name}': {e}")
    except ImportError:
        logger.warning("[SBERT] sentence-transformers not installed.")
    return None


def _get_encoder() -> tuple[object, int]:
    """Return the cached (encoder, dim) pair, initialising on first call."""
    global _encoder, _dim
    if _encoder is not None:
        return _encoder, _dim
    # Allow forcing the lightweight TF-IDF fallback to avoid large model downloads
    if os.getenv("SBERT_FORCE_FALLBACK", "0") == "1":
        logger.warning("[SBERT] SBERT_FORCE_FALLBACK=1 -> using TF-IDF fallback (fast, less accurate)")
        _encoder = _TfidfFallback()
        _dim = _TfidfFallback.DIM
        return _encoder, _dim

    result = _try_load_sbert()
    if result is not None:
        _encoder, _dim = result
    else:
        logger.warning("[SBERT] Falling back to TF-IDF encoder (no torch/SBERT available).")
        _encoder = _TfidfFallback()
        _dim = _TfidfFallback.DIM

    return _encoder, _dim


# ── Public API ────────────────────────────────────────────────────────────────

def get_dim() -> int:
    """Return the embedding dimensionality (needed when building FAISS indices)."""
    _, dim = _get_encoder()
    return dim


def fit_fallback(corpus: list[str]):
    """
    Pre-fit the TF-IDF fallback encoder on a known corpus (e.g. all CPT/ICD
    descriptions). Call this once at index-build time so the vocabulary is rich.
    Has no effect when the real SBERT model is loaded.
    """
    enc, _ = _get_encoder()
    if isinstance(enc, _TfidfFallback):
        enc.fit(corpus)


def encode(texts: list[str] | str, normalize: bool = True) -> np.ndarray:
    """
    Encode one or more strings into L2-normalised float32 embeddings.

    Parameters
    ----------
    texts     : str or list[str]
    normalize : L2-normalise output (default True — keeps cosine == dot-product)

    Returns
    -------
    np.ndarray  shape (n, dim), dtype float32
    """
    if isinstance(texts, str):
        texts = [texts]

    enc, _ = _get_encoder()

    vecs = enc.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

    # Ensure float32 (some models return float16 or float64)
    vecs = np.array(vecs, dtype="float32")

    # Explicit normalisation safety pass
    if normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

    return vecs


def encode_single(text: str, normalize: bool = True) -> np.ndarray:
    """Convenience wrapper — returns shape (dim,) for a single string."""
    return encode([text], normalize=normalize)[0]


def prewarm_encoder():
    """Public helper to pre-initialise the encoder (call in background to avoid blocking)."""
    try:
        _get_encoder()
        logger.info("[SBERT] Encoder prewarmed")
    except Exception as e:
        logger.warning(f"[SBERT] Prewarm failed: {e}")
