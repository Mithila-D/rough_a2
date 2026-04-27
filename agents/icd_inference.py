"""
ICD Inference Agent  (v2 — SBERT + Regex + LLM)
================================================
Identical three-matcher architecture as CPT inference, adapted for ICD-10:

  1. SBERT semantic search  — cosine similarity over ICD descriptions + aliases
  2. Regex / alias lookup   — word-boundary match + drug-context bonus
  3. LLM final selection    — sees the fused candidate list as context

Fusion
------
    fused_score = 0.65 * sbert_score + 0.35 * alias_score

alias_score already incorporates the drug-context bonus (same as original
_alias_infer logic), so that signal is preserved.

Confidence returned is the fused score, replacing the static confidence_base.
"""

from __future__ import annotations
import json
import re
import logging
from utils.confidence import combine_confidences
from utils.semantic_index import SemanticIndex
from utils.llm import llm

logger = logging.getLogger(__name__)

# ── Data loading ──────────────────────────────────────────────────────────────
with open("data/icd10_knowledge_graph.json", encoding="utf-8") as f:
    ICD_DATA: list[dict] = json.load(f)

with open("data/drugs.json", encoding="utf-8") as f:
    DRUGS: list[dict] = json.load(f)

_icd_by_code   = {item["code"]: item for item in ICD_DATA}
_drug_by_name  = {d["name"].lower(): d["id"] for d in DRUGS}
_drug_by_brand = {d.get("brand", "").lower(): d["id"] for d in DRUGS if d.get("brand")}

# ── Index singleton ───────────────────────────────────────────────────────────
_icd_semantic_index = SemanticIndex("data/icd10_knowledge_graph.json", id_field="code")

# ── Fusion weights ────────────────────────────────────────────────────────────
_W_SBERT = 0.65
_W_ALIAS = 0.35
_TOP_K   = 5

# ── Candidate list shown to LLM ───────────────────────────────────────────────
_FULL_CANDIDATE_LIST = "\n".join(
    f"  {item['code']} — {item.get('description', '')}"
    for item in ICD_DATA
)


# ── Drug ID resolution (shared helper) ───────────────────────────────────────

def _resolve_drug_ids(drug_names: list[str]) -> set[str]:
    """Map extracted drug name strings to their canonical drug IDs."""
    ids: set[str] = set()
    for name in drug_names:
        k = name.lower().strip()
        did = _drug_by_name.get(k) or _drug_by_brand.get(k)
        if did:
            ids.add(did)
        else:
            for n2, did2 in _drug_by_name.items():
                if k in n2 or n2 in k:
                    ids.add(did2)
                    break
    return ids


# ── Matcher 1: SBERT semantic search ─────────────────────────────────────────

def _sbert_match(text: str) -> dict[str, float]:
    """Returns {icd_code: sbert_score} for top-K semantic hits."""
    try:
        results = _icd_semantic_index.search(text, top_k=_TOP_K, min_score=0.20)
        return {r["code"]: r["semantic_score"] for r in results}
    except Exception as e:
        logger.warning(f"[ICD] SBERT search failed: {e}")
        return {}


# ── Matcher 2: Regex / alias lookup with drug-context bonus ──────────────────

def _alias_match(text: str, drug_ids: set[str]) -> dict[str, float]:
    """
    Returns {icd_code: alias_score} using:
      - word-boundary regex match on all aliases
      - drug context bonus: +0.15 per matching related drug (capped at 0.30)

    Final scores are normalised to [0, 1].
    """
    text_lower = text.lower()
    raw_scores: dict[str, float] = {}

    for item in ICD_DATA:
        hit_count = 0
        hit_len   = 0
        for alias in item.get("aliases", []):
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, text_lower):
                hit_count += 1
                hit_len   += len(alias)

        if hit_count == 0:
            continue

        base_score = float(hit_count * hit_len)

        # Drug-context bonus
        related_drugs = set(item.get("related_drugs", []))
        drug_bonus    = min(len(related_drugs & drug_ids) * 0.15, 0.30)
        raw_scores[item["code"]] = base_score + drug_bonus

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    return {code: round(s / max_score, 4) for code, s in raw_scores.items()}


# ── Score fusion ──────────────────────────────────────────────────────────────

def _fuse_scores(sbert: dict[str, float],
                 alias: dict[str, float]) -> list[tuple[str, float]]:
    all_codes = set(sbert) | set(alias)
    fused = {
        code: round(_W_SBERT * sbert.get(code, 0.0) + _W_ALIAS * alias.get(code, 0.0), 4)
        for code in all_codes
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ── Matcher 3: LLM final selection ───────────────────────────────────────────

def _llm_infer(text: str, fused_candidates: list[tuple[str, float]]) -> str | None:
    if fused_candidates:
        candidate_block = "TOP CANDIDATES (ranked by semantic + alias score):\n"
        for code, score in fused_candidates[:_TOP_K]:
            desc = _icd_by_code.get(code, {}).get("description", "")
            candidate_block += f"  {code} — {desc}  [score: {score:.3f}]\n"
        candidate_block += "\nFULL CANDIDATE LIST (if none of the above fit):\n" + _FULL_CANDIDATE_LIST
    else:
        candidate_block = "CANDIDATE ICD-10 CODES:\n" + _FULL_CANDIDATE_LIST

    prompt = f"""You are a medical coding assistant for a prior authorization system.

Read the clinical note below and select the SINGLE best ICD-10 code for the
PRIMARY diagnosis or condition being treated in this authorization request.

Rules:
- Choose ONLY from the candidate list. Do NOT invent codes.
- If no listed condition is clearly present, output: NONE
- Output ONLY the ICD-10 code (e.g. C34.10) and nothing else. No explanation.

{candidate_block}

CLINICAL NOTE:
{text}

Your answer (code only, or NONE):"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().upper()
        raw = re.sub(r"[^A-Z0-9.\-]", "", raw)
        if raw == "NONE" or not raw:
            return None
        return raw if raw in _icd_by_code else None
    except Exception as e:
        logger.warning(f"[ICD] LLM inference failed: {e}")
        return None


# ── Main agent entry point ────────────────────────────────────────────────────

def icd_inference(state):
    """
    ICD Inference Agent.

    Always runs SBERT + regex/alias (with drug-context bonus), fuses scores,
    then asks the LLM to make the final selection.
    """
    text     = state.sanitized_text or ""
    entities = state.entities or {}

    # Resolve drug IDs for alias drug-context bonus
    drug_ids = _resolve_drug_ids(entities.get("drugs", []))

    # Step 1 — SBERT
    sbert_scores = _sbert_match(text)
    logger.info(f"[ICD] SBERT top hits: {list(sbert_scores.items())[:3]}")

    # Step 2 — Regex / alias + drug bonus (always runs)
    alias_scores = _alias_match(text, drug_ids)
    logger.info(f"[ICD] Alias top hits: {list(alias_scores.items())[:3]}")

    # Step 3 — Fuse
    fused = _fuse_scores(sbert_scores, alias_scores)

    # Step 4 — LLM final decision
    llm_code = _llm_infer(text, fused)

    # ── Resolve best code ─────────────────────────────────────────────────────
    best_code  = None
    confidence = 0.0
    methods    = []
    fused_dict = dict(fused)

    if llm_code:
        best_code = llm_code
        methods.append("llm")
        if best_code in fused_dict:
            confidence = fused_dict[best_code]
            methods.append("sbert+alias")
        else:
            confidence = combine_confidences(
                _icd_by_code.get(best_code, {}).get("confidence_base", 0.80)
            )
    elif fused:
        best_code, confidence = fused[0]
        methods.append("sbert+alias_fallback")

    if not best_code or best_code not in _icd_by_code:
        state.audit_log.append({
            "step":        "ICD_NOT_FOUND",
            "sbert_hits":  len(sbert_scores),
            "alias_hits":  len(alias_scores),
        })
        return state

    best_item = _icd_by_code[best_code]
    state.icd = {
        "code":             best_code,
        "description":      best_item.get("description", ""),
        "confidence":       round(confidence, 4),
        "inference_method": "+".join(methods),
        "sbert_score":      sbert_scores.get(best_code, 0.0),
        "alias_score":      alias_scores.get(best_code, 0.0),
        "fused_score":      round(fused_dict.get(best_code, confidence), 4),
        "top_candidates":   [
            {
                "code":        c,
                "fused_score": s,
                "description": _icd_by_code.get(c, {}).get("description", ""),
            }
            for c, s in fused[:_TOP_K]
        ],
    }
    state.audit_log.append({
        "step":         "ICD_INFERRED",
        "code":         best_code,
        "method":       state.icd["inference_method"],
        "sbert_score":  state.icd["sbert_score"],
        "alias_score":  state.icd["alias_score"],
        "fused_score":  state.icd["fused_score"],
        "confidence":   state.icd["confidence"],
    })
    return state
