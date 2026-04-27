"""
Policy Evaluator  (v2 — RAG + FAISS + structured rule evaluation)
=================================================================
Two-stage evaluation:

STAGE 1 — RAG retrieval (new)
  Build a FAISS index over policy "body" text using SBERT.
  For the current clinical note + ICD + drug context, retrieve the
  top-K most semantically relevant policy chunks.
  This replaces the pure drug_id → policy lookup when exact drug IDs
  are not found, and also surfaces relevant policy clauses for the LLM.

STAGE 2 — Structured rule evaluation (original, preserved)
  For every drug extracted from the note:
    a. Resolve drug_id via name/brand lookup
    b. Walk all policies that cover that drug_id (exact match — reliable)
    c. Check approved_icd10_codes
    d. Evaluate exclusion_rules (clinical_keywords, negation_patterns,
       absence_keywords)

RAG is used to AUGMENT the structured evaluation, not replace it:
  - If exact drug match found → structured evaluation is authoritative.
  - RAG result is added to audit_log for transparency.
  - If NO exact drug match → RAG result drives the decision (PENDING or DENY).

FAISS index
-----------
  Index: IndexFlatIP (cosine via L2-norm'd dot product)
  Each policy is indexed by its full body text.
  At query time the note text + ICD + drug names are concatenated to form
  a rich query vector.
"""

from __future__ import annotations
import json
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Data loading ──────────────────────────────────────────────────────────────
with open("data/policy_documents.json", encoding="utf-8") as f:
    POLICIES: list[dict] = json.load(f)

with open("data/drugs.json", encoding="utf-8") as f:
    DRUGS: list[dict] = json.load(f)

_drug_by_name  = {d["name"].lower(): d["id"] for d in DRUGS}
_drug_by_brand = {d.get("brand", "").lower(): d["id"] for d in DRUGS if d.get("brand")}
_policy_by_id  = {p["id"]: p for p in POLICIES}


# ── FAISS Policy RAG index (lazy-built) ───────────────────────────────────────

class _PolicyRAG:
    """
    Builds and queries a FAISS semantic index over policy body text.
    Each policy is a single document; scores reflect relevance of the
    clinical context to the policy's coverage language.
    """
    def __init__(self):
        self._index   = None
        self._items   = []    # list of policy dicts in index order
        self._texts   = []    # the text used to build each embedding
        self._built   = False

    def _build(self):
        if self._built:
            return
        import faiss
        from utils.sbert_encoder import encode, fit_fallback, get_dim

        self._items = list(POLICIES)
        # Index text = title + insurer + body — gives the model rich context
        self._texts = [
            f"{p.get('title', '')}. {p.get('insurer', '')}. {p.get('body', '')}"
            for p in self._items
        ]

        fit_fallback(self._texts)
        logger.info(f"[PolicyRAG] Encoding {len(self._texts)} policies…")

        vecs = encode(self._texts, normalize=True)
        dim  = get_dim()

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        self._built = True
        logger.info(f"[PolicyRAG] FAISS index built: {self._index.ntotal} policies, dim={dim}")

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.25) -> list[dict]:
        """
        Retrieve top-K policies semantically similar to the query.
        Each result has "rag_score" added.
        """
        self._build()
        from utils.sbert_encoder import encode

        q_vec = encode([query], normalize=True)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_score:
                continue
            item = dict(self._items[idx])
            item["rag_score"] = round(float(score), 4)
            results.append(item)
        return results


_rag = _PolicyRAG()   # singleton


# ── Drug ID resolution ────────────────────────────────────────────────────────

def _find_drug_id(name: str) -> str | None:
    k = name.lower().strip()
    did = _drug_by_name.get(k) or _drug_by_brand.get(k)
    if did:
        return did
    for n2, did2 in _drug_by_name.items():
        if k in n2 or n2 in k:
            return did2
    for b2, did2 in _drug_by_brand.items():
        if b2 and (k in b2 or b2 in k):
            return did2
    return None


# ── Exclusion rule evaluation (unchanged from v1) ─────────────────────────────

def _check_exclusion_rules(policy: dict, drug_id: str,
                            clinical_text: str) -> list[dict]:
    """
    Evaluate all exclusion_rules in a policy for a given drug.
    Returns list of triggered exclusion dicts (empty = no violations).

    Rule types:
      clinical_keywords   : ANY keyword present -> DENY
      negation_patterns   : explicit negation OR missing positive evidence -> DENY
      absence_keywords    : NONE of the keywords found -> DENY (legacy)
    """
    triggered = []
    text_lower = (clinical_text or "").lower()

    for rule in policy.get("exclusion_rules", []):
        if drug_id not in rule.get("drug_ids", []):
            continue

        # Type 1: contraindicated keyword present
        if "clinical_keywords" in rule and "negation_patterns" not in rule:
            for kw in rule["clinical_keywords"]:
                if kw.lower() in text_lower:
                    triggered.append(rule)
                    break

        # Type 2: step-therapy / negation-aware check
        elif "negation_patterns" in rule:
            explicit_denial = any(
                re.search(pat, text_lower)
                for pat in rule["negation_patterns"]
            )
            positive_kws        = rule.get("positive_prior_keywords", [])
            has_positive_evidence = any(kw.lower() in text_lower for kw in positive_kws)

            if explicit_denial or not has_positive_evidence:
                triggered.append(rule)

        # Type 3: absence check (legacy)
        elif "absence_keywords" in rule:
            if not any(kw.lower() in text_lower for kw in rule["absence_keywords"]):
                triggered.append(rule)

    return triggered


# ── Main agent entry point ────────────────────────────────────────────────────

def policy_evaluator(state):
    """
    Evaluate prior-auth policy using:
      Stage 1 — FAISS RAG retrieval (semantic relevance, any drug/ICD)
      Stage 2 — Structured drug_id + ICD exact matching + exclusion rules
    """
    icd_code      = (state.icd or {}).get("code", "")
    entities      = state.entities or {}
    drugs_list    = entities.get("drugs", [])
    clinical_text = state.sanitized_text or state.raw_text or ""

    # ── Stage 1: RAG retrieval ─────────────────────────────────────────────────
    # Build a rich query from all available clinical signals
    rag_query = " ".join(filter(None, [
        clinical_text[:500],          # first 500 chars of note (most relevant)
        icd_code,
        " ".join(drugs_list[:5]),
    ]))
    rag_hits = _rag.search(rag_query, top_k=3, min_score=0.25)
    rag_policy_ids = [h["id"] for h in rag_hits]
    logger.info(f"[PolicyRAG] Top hits: {[(h['id'], h['rag_score']) for h in rag_hits]}")

    # ── Stage 2: Structured evaluation ────────────────────────────────────────
    decision        = "DENY"
    policy_hits     = []
    matched_pol     = None
    exclusion_hits  = []

    for drug_name in drugs_list:
        if not drug_name:
            continue
        drug_id = _find_drug_id(drug_name)
        if not drug_id:
            continue

        for pol in POLICIES:
            if drug_id not in pol.get("drug_ids", []):
                continue

            approved_icds = pol.get("approved_icd10_codes", [])
            icd_ok        = (icd_code in approved_icds) if icd_code else False

            triggered = _check_exclusion_rules(pol, drug_id, clinical_text)

            # RAG corroboration flag — did semantic retrieval also surface this policy?
            rag_corroborated = pol["id"] in rag_policy_ids
            rag_score = next(
                (h["rag_score"] for h in rag_hits if h["id"] == pol["id"]), 0.0
            )

            policy_hits.append({
                "drug":                  drug_name,
                "drug_id":               drug_id,
                "policy_id":             pol["id"],
                "policy_title":          pol.get("title", ""),
                "approved_icds":         approved_icds,
                "icd_matched":           icd_ok,
                "step_therapy":          pol.get("step_therapy_required", False),
                "exclusions_triggered":  [r["rule_id"] for r in triggered],
                "exclusion_reasons":     [r["reason"] for r in triggered],
                "rag_corroborated":      rag_corroborated,
                "rag_score":             rag_score,
            })

            if triggered:
                decision = "DENY"
                exclusion_hits.extend(triggered)
            elif icd_ok:
                decision    = "APPROVED"
                matched_pol = pol

            break  # one policy per drug

    # ── No exact drug match — use RAG result ──────────────────────────────────
    if not policy_hits:
        if rag_hits and icd_code:
            # Check if the RAG-retrieved policy covers this ICD
            for rag_pol in rag_hits:
                pol_data = _policy_by_id.get(rag_pol["id"])
                if pol_data and icd_code in pol_data.get("approved_icd10_codes", []):
                    decision = "PENDING"
                    policy_hits.append({
                        "drug":            "(RAG-retrieved, no exact drug match)",
                        "drug_id":         None,
                        "policy_id":       rag_pol["id"],
                        "policy_title":    rag_pol.get("title", ""),
                        "approved_icds":   pol_data.get("approved_icd10_codes", []),
                        "icd_matched":     True,
                        "rag_corroborated": True,
                        "rag_score":       rag_pol["rag_score"],
                    })
                    break
        elif icd_code:
            # Fallback: scan all policies for ICD match
            for pol in POLICIES:
                if icd_code in pol.get("approved_icd10_codes", []):
                    decision = "PENDING"
                    break

    # ── Deduplicate exclusion hits ─────────────────────────────────────────────
    seen_rules: set[str] = set()
    deduped_exclusions = []
    for r in exclusion_hits:
        if r["rule_id"] not in seen_rules:
            deduped_exclusions.append(r)
            seen_rules.add(r["rule_id"])

    state.policy = {
        "decision":          decision,
        "policy_hits":       policy_hits,
        "matched_policy":    matched_pol.get("id") if matched_pol else None,
        "exclusion_hits":    [r["rule_id"] for r in deduped_exclusions],
        "exclusion_reasons": [r["reason"] for r in deduped_exclusions],
        "rag_hits":          [{"id": h["id"], "title": h.get("title",""), "rag_score": h["rag_score"]} for h in rag_hits],
    }
    state.audit_log.append({
        "step":       "POLICY_EVALUATED",
        "decision":   decision,
        "drug_hits":  len(policy_hits),
        "exclusions": [r["rule_id"] for r in exclusion_hits],
        "rag_hits":   rag_policy_ids,
    })
    return state
