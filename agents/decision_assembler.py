"""
Decision Assembler — Agent 8 (Confidence Scoring) + Final Decision Logic
=========================================================================

Implements the exact scoring from the design doc:

  confidence =
      avg(icd_conf)  * 0.30
    + avg(cpt_conf)  * 0.25
    + avg(drug_conf) * 0.20
    + policy_score   * 0.15
    - ambiguity_penalty        # = 0.30 * raw_score  IF ambiguities > 0

Final decision rules (in order):
  1. No policy match found                      -> DENY  (hard, no score)
  2. Signal in input but no code matched         -> HUMAN_REVIEW
  3. Policy EXCLUSION rule triggered             -> DENY  (hard)
  4. Critique flags HITL / STEP_THERAPY          -> HUMAN_REVIEW
  5. Ambiguities detected (Agent 7)              -> HITL_NEEDED
  6. Any component below min threshold (0.60)    -> DENY
  7. Total weighted score < 0.60                 -> DENY
  8. Otherwise                                   -> policy_decision
     (APPROVED / PENDING / DENY from Agent 6)
"""

from utils.confidence import (
    compute_weighted_score,
    MIN_ICD_CONFIDENCE,
    MIN_CPT_CONFIDENCE,
    MIN_DRUG_CONFIDENCE,
    MIN_TOTAL_SCORE,
)


# ── Helpers to detect what the original note contains ────────────────────────

def _procedure_mentioned(state) -> bool:
    """True if any CPT alias appears in the (sanitized) note."""
    try:
        from utils.alias_index import AliasIndex
        idx  = AliasIndex("data/cpt_procedures.json")
        text = (state.sanitized_text or state.raw_text or "").lower()
        for item in idx.data:
            for alias in item.get("aliases", []):
                if alias.lower() in text:
                    return True
    except Exception:
        pass
    return False


def _drug_mentioned(state) -> bool:
    """True if any drug name / brand or ICD alias appears in the note."""
    import json
    text = (state.sanitized_text or state.raw_text or "").lower()
    try:
        with open("data/drugs.json") as f:
            drugs = json.load(f)
        for d in drugs:
            if d.get("name", "").lower() in text:
                return True
            if d.get("brand", "").lower() in text:
                return True
    except Exception:
        pass
    try:
        with open("data/icd10_knowledge_graph.json") as f:
            icd_data = json.load(f)
        for item in icd_data:
            for alias in item.get("aliases", []):
                if alias.lower() in text:
                    return True
    except Exception:
        pass
    return False


def _get_drug_confidence(state) -> float | None:
    """
    Return a drug certainty score (0-1) from the drug_checks list.
    Returns None if no drug validation has been run.
    Aggregation: average of all checked drugs (design doc: avg(drug_conf)).
    """
    checks = getattr(state, "drug_checks", None)
    if not checks:
        return None
    scores = [c.get("confidence", 0.0) for c in checks if "confidence" in c]
    return round(sum(scores) / len(scores), 4) if scores else None


def _get_ambiguity_count(state) -> int:
    """
    Return the number of ambiguities detected by Agent 7.
    Reads state.ambiguities (list of strings set by ambiguity_detector).
    Falls back to 0 if agent hasn't run.
    """
    ambiguities = getattr(state, "ambiguities", None)
    if isinstance(ambiguities, list):
        return len(ambiguities)
    return 0


# ── Main agent ────────────────────────────────────────────────────────────────

def decision_assembler(state):
    policy_decision  = state.policy.get("decision", "DENY")
    critique_flags   = state.critique.get("flags", [])
    policy_hits      = state.policy.get("policy_hits", [])
    ambiguity_count  = _get_ambiguity_count(state)

    cpt_conf  = state.cpt.get("confidence")  if state.cpt.get("code")  else None
    icd_conf  = state.icd.get("confidence")  if state.icd.get("code")  else None
    drug_conf = _get_drug_confidence(state)

    cpt_in_input = _procedure_mentioned(state)
    icd_in_input = _drug_mentioned(state)

    # ── Rule 1: No policy match at all -> hard DENY ───────────────────────────
    if not policy_hits and policy_decision == "DENY":
        state.final_decision = {
            "decision":        "DENY",
            "reason":          "No matching policy found for this drug/ICD combination.",
            "confidence":      0.0,
            "scoring_logic":   "Policy match required — none found → hard DENY",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "DENY",
                                 "reason": "no_policy_match"})
        return state

    # ── Rule 2: Signal present in note but no code matched -> HUMAN_REVIEW ───
    if cpt_in_input and cpt_conf is None:
        state.final_decision = {
            "decision":        "HUMAN_REVIEW",
            "reason":          "Procedure mentioned in note but no CPT code matched — human review required.",
            "confidence":      None,
            "scoring_logic":   "CPT keyword found in note but unmatched → auto-decision blocked",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "HUMAN_REVIEW",
                                 "reason": "cpt_unmatched"})
        return state

    if icd_in_input and icd_conf is None:
        state.final_decision = {
            "decision":        "HUMAN_REVIEW",
            "reason":          "Drug/diagnosis in note but no ICD-10 code matched — human review required.",
            "confidence":      None,
            "scoring_logic":   "ICD keyword found in note but unmatched → auto-decision blocked",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "HUMAN_REVIEW",
                                 "reason": "icd_unmatched"})
        return state

    # ── Rule 3: Policy EXCLUSION triggered -> hard DENY ──────────────────────
    exclusion_flags = [f for f in critique_flags if "POLICY_EXCLUSION" in f]
    if exclusion_flags:
        exclusion_reasons = state.critique.get("exclusion_reasons", [])
        deny_reason = ("; ".join(exclusion_reasons)
                       if exclusion_reasons
                       else "A policy exclusion rule was triggered for this drug/diagnosis combination.")
        state.final_decision = {
            "decision":        "DENY",
            "reason":          deny_reason,
            "confidence":      0.0,
            "scoring_logic":   f"Policy exclusion rule(s) triggered: {exclusion_flags} → hard DENY",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": "DENY",
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "DENY",
                                 "reason": "policy_exclusion", "flags": exclusion_flags})
        return state

    # ── Rule 4: Critique flags HITL / STEP_THERAPY -> HUMAN_REVIEW ───────────
    if any("HITL" in f or "STEP_THERAPY" in f for f in critique_flags):
        step_drugs = [f.split(":", 1)[1] for f in critique_flags if "STEP_THERAPY" in f]
        review_reason = (
            f"Step therapy not satisfied for: {', '.join(step_drugs)}. "
            f"Policy requires prior treatment failure before this drug. "
            f"Escalate to clinical review queue."
            if step_drugs
            else "Critique agent flagged issues requiring human review."
        )
        state.final_decision = {
            "decision":        "HUMAN_REVIEW",
            "reason":          review_reason,
            "confidence":      None,
            "scoring_logic":   f"Critique flags {critique_flags} → HUMAN_REVIEW",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "HUMAN_REVIEW",
                                 "reason": "critique_flags", "flags": critique_flags})
        return state

    # ── Rule 5: Ambiguities detected (Agent 7) -> HITL_NEEDED ──────────
    if ambiguity_count > 0:
        ambiguity_list = getattr(state, "ambiguities", [])
        state.final_decision = {
            "decision":        "HITL_NEEDED",
            "reason":          f"{ambiguity_count} ambiguity/ambiguities detected: {ambiguity_list}",
            "confidence":      None,
            "scoring_logic":   f"Ambiguities present ({ambiguity_count}) → HITL_NEEDED per design doc rule",
            "weights_used":    {},
            "component_scores": {},
            "critique_flags":  critique_flags,
            "ambiguity_count": ambiguity_count,
            "policy_decision": policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "HITL_NEEDED",
                                 "reason": "ambiguities_detected", "count": ambiguity_count})
        return state

    # ── Rules 6 & 7: Score + threshold check ─────────────────────────────────
    policy_score = (
        1.0 if policy_decision == "APPROVED" else
        0.5 if policy_decision == "PENDING"  else
        0.0
    )

    score_result = compute_weighted_score(
        icd_conf        = icd_conf,
        cpt_conf        = cpt_conf,
        drug_conf       = drug_conf,
        policy_score    = policy_score,
        ambiguity_count = ambiguity_count,   # 0 here (caught above), penalty = 0
    )
    total_score            = score_result["total"]
    threshold_violations   = score_result["threshold_violations"]

    # Rule 6: Component below minimum threshold
    if threshold_violations:
        state.final_decision = {
            "decision":         "DENY",
            "reason":           f"Component confidence below minimum: {'; '.join(threshold_violations)}",
            "confidence":       total_score,
            "scoring_logic":    score_result["scoring_logic"],
            "weights_used":     score_result["weights_used"],
            "component_scores": score_result["component_scores"],
            "critique_flags":   critique_flags,
            "ambiguity_count":  ambiguity_count,
            "policy_decision":  policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "DENY",
                                 "reason": "threshold_violation",
                                 "violations": threshold_violations})
        return state

    # Rule 7: Total weighted score below 0.60
    if total_score < MIN_TOTAL_SCORE:
        state.final_decision = {
            "decision":         "DENY",
            "reason":           f"Total weighted confidence {total_score:.3f} below minimum threshold {MIN_TOTAL_SCORE}.",
            "confidence":       total_score,
            "scoring_logic":    score_result["scoring_logic"],
            "weights_used":     score_result["weights_used"],
            "component_scores": score_result["component_scores"],
            "critique_flags":   critique_flags,
            "ambiguity_count":  ambiguity_count,
            "policy_decision":  policy_decision,
        }
        state.audit_log.append({"step": "FINAL_DECISION", "decision": "DENY",
                                 "reason": "low_total_score", "score": total_score})
        return state

    # ── Rule 8: Accept policy decision ───────────────────────────────────────
    final = policy_decision
    state.final_decision = {
        "decision":         final,
        "reason":           f"All checks passed. Policy decision: {final}.",
        "confidence":       total_score,
        "scoring_logic":    score_result["scoring_logic"],
        "weights_used":     score_result["weights_used"],
        "component_scores": score_result["component_scores"],
        "critique_flags":   critique_flags,
        "ambiguity_count":  ambiguity_count,
        "policy_decision":  policy_decision,
    }
    state.audit_log.append({"step": "FINAL_DECISION", "decision": final,
                             "score": total_score})
    return state
