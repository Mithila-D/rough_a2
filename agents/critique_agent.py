"""
Critique Agent — Agent 9
=========================
LLM-powered quality control — runs AFTER the decision assembler.
Flags risk issues that deterministic rules may have missed.

Checks:
  1. Low ICD confidence (below MIN_ICD_CONFIDENCE)
  2. Low CPT confidence (below MIN_CPT_CONFIDENCE)
  3. Low / unknown drug confidence
  4. Drug off-label (ICD not in policy's approved list, no exclusion triggered)
  5. Step therapy not satisfied (when policy decision != APPROVED)
  6. Policy exclusion rule triggered (e.g. EGFR mutation + pembrolizumab)
  7. High-cost drug auto-approved without HITL flag (risk signal)

Output: state.critique = { "flags": [...], "allow": bool, "exclusion_reasons": [...] }
The flags are consumed by decision_assembler (Rules 3 and 4).
"""

from utils.confidence import MIN_CPT_CONFIDENCE, MIN_ICD_CONFIDENCE, MIN_DRUG_CONFIDENCE

HIGH_COST_THRESHOLD_USD = 5_000


def critique_agent(state):
    flags: list[str] = []

    # 1. Low ICD confidence
    if state.icd.get("code") and state.icd.get("confidence", 1.0) < MIN_ICD_CONFIDENCE:
        flags.append(f"LOW_ICD_CONFIDENCE({state.icd['confidence']:.2f})")

    # 2. Low CPT confidence
    if state.cpt.get("code") and state.cpt.get("confidence", 1.0) < MIN_CPT_CONFIDENCE:
        flags.append(f"LOW_CPT_CONFIDENCE({state.cpt['confidence']:.2f})")

    # 3. Unknown / low-confidence drug
    for check in getattr(state, "drug_checks", []):
        if not check.get("found"):
            flags.append(f"UNKNOWN_DRUG:{check.get('drug', '')}")
        elif check.get("confidence", 1.0) < MIN_DRUG_CONFIDENCE:
            flags.append(f"LOW_DRUG_CONFIDENCE:{check.get('drug', '')}({check['confidence']:.2f})")

    policy_hits     = state.policy.get("policy_hits", [])
    policy_decision = state.policy.get("decision", "DENY")

    # 4. Drug off-label (policy hit exists but ICD not in approved list, no exclusion)
    # Only flag when policy did NOT approve — if policy_evaluator already approved it,
    # the ICD was matched there; raising off-label on top is a false positive.
    if policy_decision != "APPROVED":
        for hit in policy_hits:
            if not hit.get("icd_matched") and not hit.get("exclusions_triggered"):
                flags.append(f"DRUG_OFF_LABEL:{hit.get('drug', '')}")

    # 5. Step therapy not satisfied — only flag when policy did NOT approve
    if policy_decision != "APPROVED":
        seen_step: set[str] = set()
        for hit in policy_hits:
            if hit.get("step_therapy"):
                pol_id = hit.get("policy_id", hit.get("drug", ""))
                if pol_id not in seen_step:
                    flags.append(f"STEP_THERAPY_REQUIRED:{hit.get('drug', '')}")
                    seen_step.add(pol_id)

    # 6. Policy exclusion rules triggered
    seen_excl: set[str] = set()
    for hit in policy_hits:
        for excl_id in hit.get("exclusions_triggered", []):
            if excl_id not in seen_excl:
                flags.append(f"POLICY_EXCLUSION:{excl_id}")
                seen_excl.add(excl_id)

    # 7. High-cost drug auto-approved — flag for awareness (not a block by itself)
    if policy_decision == "APPROVED":
        for check in getattr(state, "drug_checks", []):
            if check.get("high_cost"):
                flags.append(f"HIGH_COST_AUTO_APPROVED:{check.get('drug','')}(${check.get('monthly_cost',0):,}/mo)")

    # Surface exclusion reasons for decision_assembler
    exclusion_reasons = state.policy.get("exclusion_reasons", [])

    # "allow" means no blocking flags. HIGH_COST_AUTO_APPROVED is informational only
    # (awareness flag for clinical managers, not a decision blocker) — exclude it
    # from the blocking count so approved cases aren't incorrectly flagged.
    blocking_flags = [
        f for f in flags
        if not f.startswith("HIGH_COST_AUTO_APPROVED")
    ]

    state.critique = {
        "flags":             flags,
        "allow":             len(blocking_flags) == 0,
        "exclusion_reasons": exclusion_reasons,
    }
    state.audit_log.append({"step": "CRITIQUE_DONE", "flags": flags,
                             "flag_count": len(flags)})
    return state
