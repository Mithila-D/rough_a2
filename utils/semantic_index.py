"""
Semantic FAISS Index for CPT / ICD codes
=========================================
Builds a FAISS IndexFlatIP (cosine similarity via L2-normalised dot product)
over all CPT or ICD descriptions + aliases using the SBERT encoder.

On first call the index is built and cached in memory.  Subsequent calls reuse
the cached index (no re-encoding).

Usage
-----
    from utils.semantic_index import SemanticIndex
    idx = SemanticIndex("data/cpt_procedures.json", id_field="code")
    results = idx.search("genetic sequencing panel for lung cancer", top_k=3)
    # → [{"code": "81455", "score": 0.91, ...}, ...]
"""

from __future__ import annotations
import json
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class SemanticIndex:
    """
    FAISS semantic index over a JSON catalogue (CPT or ICD data).

    Each item is represented by a single embedding built from:
        "<description>. <alias1>. <alias2>. ..."
    This gives the model maximum context per code.
    """

    def __init__(self, json_path: str, id_field: str = "code"):
        """
        Parameters
        ----------
        json_path : path to the JSON file (list of dicts)
        id_field  : the field used as the primary key in results ("code")
        """
        self._json_path = json_path
        self._id_field  = id_field
        self._index     = None   # faiss index
        self._items: list[dict] = []
        self._texts: list[str]  = []
        self._built = False

    # ── Index construction ────────────────────────────────────────────────────

    def _build_text(self, item: dict) -> str:
        """Concatenate description + aliases into one rich text string."""
        parts = []
        desc = item.get("description", "")
        if desc:
            parts.append(desc)
        for alias in item.get("aliases", []):
            if alias and alias not in parts:
                parts.append(alias)
        return ". ".join(parts)

    def _ensure_built(self):
        if self._built:
            return

        import faiss
        from utils.sbert_encoder import encode, fit_fallback, get_dim

        with open(self._json_path, encoding="utf-8") as f:
            self._items = json.load(f)

        self._texts = [self._build_text(item) for item in self._items]

        # Pre-fit TF-IDF fallback (no-op when real SBERT is available)
        fit_fallback(self._texts)

        logger.info(f"[SemanticIndex] Encoding {len(self._texts)} items from {self._json_path}…")
        vecs = encode(self._texts, normalize=True)  # (n, dim) float32

        dim = get_dim()
        self._index = faiss.IndexFlatIP(dim)   # Inner-product == cosine (L2-norm'd)
        self._index.add(vecs)
        self._built = True
        logger.info(f"[SemanticIndex] Index built: {self._index.ntotal} vectors, dim={dim}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, min_score: float = 0.30) -> list[dict]:
        """
        Semantic search over the index.

        Parameters
        ----------
        query     : free-text clinical query (e.g. fragment from a clinical note)
        top_k     : number of candidates to return
        min_score : minimum cosine similarity threshold (0-1)

        Returns
        -------
        List of result dicts, each is the original catalogue item augmented with:
            "semantic_score" : float  (cosine similarity 0-1)
            "semantic_rank"  : int    (1 = best)
        """
        self._ensure_built()

        from utils.sbert_encoder import encode

        q_vec = encode([query], normalize=True)  # (1, dim)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)  # (1, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or float(score) < min_score:
                continue
            item = dict(self._items[idx])
            item["semantic_score"] = round(float(score), 4)
            item["semantic_rank"]  = rank
            results.append(item)

        return results

    def search_batch(self, queries: list[str], top_k: int = 3,
                     min_score: float = 0.30) -> list[list[dict]]:
        """
        Batch semantic search — more efficient than calling search() in a loop.
        Returns a list (one per query) of result lists.
        """
        self._ensure_built()

        from utils.sbert_encoder import encode

        q_vecs = encode(queries, normalize=True)  # (n_queries, dim)
        k = min(top_k, self._index.ntotal)
        scores_all, indices_all = self._index.search(q_vecs, k)

        all_results = []
        for scores, indices in zip(scores_all, indices_all):
            results = []
            for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
                if idx < 0 or float(score) < min_score:
                    continue
                item = dict(self._items[idx])
                item["semantic_score"] = round(float(score), 4)
                item["semantic_rank"]  = rank
                results.append(item)
            all_results.append(results)

        return all_results
