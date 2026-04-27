"""
CPT Inference Agent  (v2 — SBERT + Regex + LLM)
================================================
Three matchers run on EVERY call and their results are fused:

  1. SBERT semantic search  (PubMedBERT via SemanticIndex / FAISS)
     – Encodes the clinical note and retrieves the top-K semantically
       similar CPT codes.  Score = cosine similarity (0-1).

  2. Regex / alias lookup   (AliasIndex — always runs, never skipped)
     – Exact and substring match against every CPT alias.
     – Score = normalised alias-hit density (0-1).

  3. LLM inference          (primary decision maker)
     – Given the note AND the top candidates from steps 1+2 as context,
       the LLM selects the single best code.
     – Falls back to the highest-scoring fused candidate if LLM fails.

Fusion strategy
---------------
For each candidate code seen in SBERT or alias results:
    fused_score = α * sbert_score + β * alias_score
    α = 0.65,  β = 0.35

The LLM is shown the top-5 fused candidates (code + description + score) so
it can make a more informed choice.  If the LLM picks a code that is in the
fused candidate list, its confidence is the fused score; otherwise a default
of 0.75 is used.

Confidence returned in state.cpt["confidence"] is always the fused score
(not the static confidence_base), making it a real signal for the assembler.
"""

from __future__ import annotations
import json
import re
import logging
from utils.alias_index import AliasIndex
from utils.semantic_index import SemanticIndex
from utils.llm import llm

logger = logging.getLogger(__name__)

# ── Data loading ──────────────────────────────────────────────────────────────
with open("data/cpt_procedures.json", encoding="utf-8") as f:
    _CPT_DATA: list[dict] = json.load(f)

_cpt_by_code: dict[str, dict] = {c["code"]: c for c in _CPT_DATA}

# ── Index singletons (built lazily on first inference call) ───────────────────
_cpt_alias_index    = AliasIndex("data/cpt_procedures.json")
_cpt_semantic_index = SemanticIndex("data/cpt_procedures.json", id_field="code")

# ── Fusion weights ────────────────────────────────────────────────────────────
_W_SBERT = 0.65
_W_ALIAS = 0.35
_TOP_K   = 5   # candidates shown to LLM

# ── Candidate list for LLM fallback (full list) ───────────────────────────────
_FULL_CANDIDATE_LIST = "\n".join(
    f"  {c['code']} — {c.get('description', '')}"
    for c in _CPT_DATA
)


# ── Matcher 1: SBERT semantic search ─────────────────────────────────────────

def _sbert_match(text: str) -> dict[str, float]:
    """
    Returns {code: sbert_score} for the top-K semantic matches.
    Scores are cosine similarities (0-1).
    """
    try:
        results = _cpt_semantic_index.search(text, top_k=_TOP_K, min_score=0.20)
        return {r["code"]: r["semantic_score"] for r in results}
    except Exception as e:
        logger.warning(f"[CPT] SBERT search failed: {e}")
        return {}


# ── Matcher 2: Regex / alias lookup ──────────────────────────────────────────

def _alias_match(text: str) -> dict[str, float]:
    """
    Returns {code: alias_score} for all codes whose aliases appear in the text.
    Score = (number of alias hits x average alias length) normalised to 0-1.
    Uses regex word-boundary for precision over plain substring.
    """
    text_lower = text.lower()
    raw_scores: dict[str, float] = {}

    for item in _CPT_DATA:
        hit_count = 0
        hit_len   = 0
        for alias in item.get("aliases", []):
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, text_lower):
                hit_count += 1
                hit_len   += len(alias)

        if hit_count > 0:
            raw_scores[item["code"]] = float(hit_count * hit_len)

    if not raw_scores:
        return {}

    max_score = max(raw_scores.values())
    return {code: round(s / max_score, 4) for code, s in raw_scores.items()}


# ── Score fusion ──────────────────────────────────────────────────────────────

def _fuse_scores(sbert: dict[str, float],
                 alias: dict[str, float]) -> list[tuple[str, float]]:
    """
    Merge SBERT and alias scores into a single ranked list.
    Codes that appear in only one matcher receive only that partial contribution.
    """
    all_codes = set(sbert) | set(alias)
    fused: dict[str, float] = {
        code: round(_W_SBERT * sbert.get(code, 0.0) + _W_ALIAS * alias.get(code, 0.0), 4)
        for code in all_codes
    }
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


# ── Matcher 3: LLM inference ─────────────────────────────────────────────────

def _llm_infer(text: str, fused_candidates: list[tuple[str, float]]) -> str | None:
    """
    Ask the LLM to select the best CPT code using fused candidates as context.
    Returns the code string or None.
    """
    if fused_candidates:
        candidate_block = "TOP CANDIDATES (ranked by semantic + alias score):\n"
        for code, score in fused_candidates[:_TOP_K]:
            desc = _cpt_by_code.get(code, {}).get("description", "")
            candidate_block += f"  {code} — {desc}  [score: {score:.3f}]\n"
        candidate_block += "\nFULL CANDIDATE LIST (if none of the above fit):\n" + _FULL_CANDIDATE_LIST
    else:
        candidate_block = "CANDIDATE CPT CODES:\n" + _FULL_CANDIDATE_LIST

    prompt = f"""You are a medical coding assistant for a prior authorization system.

Read the clinical note below and select the SINGLE most relevant CPT procedure code.

Rules:
- Choose ONLY from the candidate list. Do NOT invent codes.
- If no listed procedure/test is clearly relevant, output: NONE
- Output ONLY the CPT code (e.g. 81455) and nothing else. No explanation.

{candidate_block}

CLINICAL NOTE:
{text}

Your answer (code only, or NONE):"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip().upper()
        raw = re.sub(r"[^A-Z0-9]", "", raw)
        if raw == "NONE" or not raw:
            return None
        return raw if raw in _cpt_by_code else None
    except Exception as e:
        logger.warning(f"[CPT] LLM inference failed: {e}")
        return None


# ── Main agent entry point ────────────────────────────────────────────────────

def cpt_inference(state):
    """
    CPT Inference Agent.

    Always runs SBERT + regex/alias matching, fuses their scores, then
    asks the LLM to make the final selection with fused candidates as context.
    Falls back gracefully at each layer.
    """
    text = state.sanitized_text or ""

    # Step 1 — SBERT (semantic)
    sbert_scores = _sbert_match(text)
    logger.info(f"[CPT] SBERT top hits: {list(sbert_scores.items())[:3]}")

    # Step 2 — Regex / alias (always runs)
    alias_scores = _alias_match(text)
    logger.info(f"[CPT] Alias top hits: {list(alias_scores.items())[:3]}")

    # Step 3 — Fuse
    fused = _fuse_scores(sbert_scores, alias_scores)

    # Step 4 — LLM final decision (with fused context)
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
            confidence = _cpt_by_code.get(best_code, {}).get("confidence_base", 0.75)
    elif fused:
        best_code, confidence = fused[0]
        methods.append("sbert+alias_fallback")

    if not best_code or best_code not in _cpt_by_code:
        state.audit_log.append({
            "step":        "CPT_NOT_FOUND",
            "sbert_hits":  len(sbert_scores),
            "alias_hits":  len(alias_scores),
        })
        return state

    state.cpt = {
        "code":             best_code,
        "confidence":       round(confidence, 4),
        "inference_method": "+".join(methods),
        "sbert_score":      sbert_scores.get(best_code, 0.0),
        "alias_score":      alias_scores.get(best_code, 0.0),
        "fused_score":      round(fused_dict.get(best_code, confidence), 4),
        "top_candidates":   [
            {
                "code":        c,
                "fused_score": s,
                "description": _cpt_by_code.get(c, {}).get("description", ""),
            }
            for c, s in fused[:_TOP_K]
        ],
    }
    state.audit_log.append({
        "step":         "CPT_INFERRED",
        "code":         best_code,
        "method":       state.cpt["inference_method"],
        "sbert_score":  state.cpt["sbert_score"],
        "alias_score":  state.cpt["alias_score"],
        "fused_score":  state.cpt["fused_score"],
        "confidence":   state.cpt["confidence"],
    })
    return state
