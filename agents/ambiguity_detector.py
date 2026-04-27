"""
Ambiguity Detection Agent — Agent 7
=====================================
Uses LLM to detect missing or unclear information in the PA request.
This is the only place in the pipeline where LLM is used for semantic
reasoning about gaps (rules-only systems miss nuance here).

Examples of what it detects:
  - Missing severity / staging
  - Missing lab values referenced in policy
  - Unclear diagnosis (e.g. "possible NSCLC" vs confirmed)
  - Missing duration of previous therapy
  - Conflicting drug routes

Output: state.ambiguities (list of strings describing each gap)
IMPORTANT: Ambiguities are passed to HITL in decision_assembler (Rule 5).

LLM role here is justified: ambiguity is semantic — rule engines miss it.
"""

from __future__ import annotations
import json
import re
import logging

from utils.llm import llm

logger = logging.getLogger(__name__)


_SYSTEM_CONTEXT = """You are a prior authorization clinical reviewer.
Your ONLY job is to identify missing or unclear information in the note
that would block a prior authorization decision.

Focus on:
1. Missing lab values required by the policy (e.g., HbA1c, LVEF, eGFR)
2. Missing documentation (e.g., step therapy evidence, prior drug failures)
3. Unclear or ambiguous diagnosis (e.g., "possible", "rule-out", unconfirmed)
4. Missing severity / staging / duration information
5. Conflicting information (e.g., drug route contradicts drug type)

DO NOT flag issues that are clearly present in the note.
DO NOT flag normal clinical documentation like physician name or visit date.
Return a JSON array of short strings. Each string is ONE specific gap.
Return [] if nothing is missing or unclear.
Return ONLY valid JSON — no markdown, no explanation."""


def ambiguity_detector(state):
    """
    Detect ambiguities / missing information using LLM.
    Writes list of strings to state.ambiguities.
    """
    text      = state.sanitized_text or state.raw_text or ""
    icd_code  = (state.icd  or {}).get("code", "")
    cpt_code  = (state.cpt  or {}).get("code", "")
    drugs     = (state.entities or {}).get("drugs", [])
    policy_id = (state.policy or {}).get("matched_policy", "")

    # Build a compact clinical summary to send to LLM (no PHI — already sanitized)
    summary = f"""CLINICAL NOTE (PHI-masked):
{text[:600]}

INFERRED:
  ICD-10 code : {icd_code or 'not matched'}
  CPT code    : {cpt_code  or 'not matched'}
  Drugs       : {', '.join(drugs) if drugs else 'none extracted'}
  Policy      : {policy_id or 'not matched'}
"""

    prompt = f"""{_SYSTEM_CONTEXT}

{summary}

List each specific missing or unclear piece of information as a short string.
Return JSON array only. Example: ["HbA1c value not documented", "LVEF not specified"]
"""

    ambiguities: list[str] = []

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            ambiguities = [str(a) for a in parsed if a]
        logger.info(f"[AmbiguityDetector] Found {len(ambiguities)} ambiguities")
    except Exception as e:
        logger.warning(f"[AmbiguityDetector] LLM call failed: {e} — defaulting to no ambiguities")
        ambiguities = []

    state.ambiguities = ambiguities
    state.audit_log.append({
        "step":              "AMBIGUITY_DETECTED",
        "count":             len(ambiguities),
        "ambiguities":       ambiguities,
        "hitl_triggered":    len(ambiguities) > 0,
    })
    return state
