"""
Scoring & Confidence Utilities — Prior Authorization Decision Engine
====================================================================

Weights (your design doc, Agent 8):
  ICD confidence   : 30%
  CPT confidence   : 25%
  Drug certainty   : 20%
  Policy clarity   : 15%
  Ambiguity penalty: -30% of raw score (applied last, only when ambiguities > 0)

Formula:
  raw_score = (
      avg(icd_conf)  * 0.30 +
      avg(cpt_conf)  * 0.25 +
      avg(drug_conf) * 0.20 +
      policy_score   * 0.15
  )
  confidence = raw_score - (0.30 * raw_score)  IF ambiguities > 0
             = raw_score                        IF no ambiguities

Weight redistribution when a signal is absent from the input note:
  If a component has no match its weight is redistributed proportionally
  among the remaining present signals so weights always sum to 1.0.

Thresholds:
  ICD  >= 0.60   CPT  >= 0.60   Drug >= 0.60   Total >= 0.60
  Below any component threshold -> DENY (regardless of total score)
  Policy absence is a hard DENY handled in decision_assembler.
"""

# ── Baseline weights (from design doc Agent 8) ────────────────────────────────
W_ICD    = 0.30
W_CPT    = 0.25
W_DRUG   = 0.20
W_POLICY = 0.15

# Ambiguity reduces raw score by this fraction when ambiguities detected
AMBIGUITY_PENALTY_FRACTION = 0.30

# ── Per-component minimum thresholds ─────────────────────────────────────────
MIN_ICD_CONFIDENCE  = 0.60
MIN_CPT_CONFIDENCE  = 0.60
MIN_DRUG_CONFIDENCE = 0.60
MIN_TOTAL_SCORE     = 0.60


def compute_weighted_score(
    icd_conf: float | None,
    cpt_conf: float | None,
    drug_conf: float | None,
    policy_score: float,
    ambiguity_count: int = 0,
) -> dict:
    """
    Compute the final weighted confidence score for a PA case.

    Parameters
    ----------
    icd_conf        Confidence from ICD inference (0-1), None if no ICD found.
    cpt_conf        Confidence from CPT inference (0-1), None if no CPT found.
    drug_conf       Confidence from drug validation (0-1), None if no drug found.
    policy_score    1.0=APPROVED, 0.5=PENDING, 0.0=DENY/no match.
    ambiguity_count Number of ambiguities detected (triggers -30% penalty if >0).

    Returns
    -------
    dict with:
        total               - final confidence score (0 to 0.95 cap)
        raw_score           - score before ambiguity penalty
        ambiguity_penalty   - amount deducted (0.0 if no ambiguities)
        weights_used        - effective {icd, cpt, drug, policy} after redistribution
        component_scores    - {icd, cpt, drug, policy} values used
        scoring_logic       - human-readable formula for audit trail
        threshold_violations- list of components below their minimum threshold
    """

    icd_present  = icd_conf  is not None
    cpt_present  = cpt_conf  is not None
    drug_present = drug_conf is not None

    # Assign raw weights; absent signals get 0
    w_icd    = W_ICD    if icd_present  else 0.0
    w_cpt    = W_CPT    if cpt_present  else 0.0
    w_drug   = W_DRUG   if drug_present else 0.0
    w_policy = W_POLICY

    # Normalise so weights always sum to 1.0 regardless of missing signals
    total_w = w_icd + w_cpt + w_drug + w_policy
    if total_w > 0:
        w_icd    = w_icd    / total_w
        w_cpt    = w_cpt    / total_w
        w_drug   = w_drug   / total_w
        w_policy = w_policy / total_w

    # Effective scores for present signals
    eff_icd    = icd_conf  if icd_present  else 0.0
    eff_cpt    = cpt_conf  if cpt_present  else 0.0
    eff_drug   = drug_conf if drug_present else 0.0
    eff_policy = policy_score

    # Raw weighted score
    raw_score = round(
        w_icd * eff_icd + w_cpt * eff_cpt + w_drug * eff_drug + w_policy * eff_policy,
        4
    )

    # Ambiguity penalty: -30% of raw score when ambiguities exist
    ambiguity_penalty = 0.0
    if ambiguity_count > 0:
        ambiguity_penalty = round(AMBIGUITY_PENALTY_FRACTION * raw_score, 4)

    # Final score capped at 0.95 (regulators dislike absolute certainty)
    total_score = round(max(0.0, min(0.95, raw_score - ambiguity_penalty)), 4)

    # Threshold violation check
    threshold_violations = []
    if icd_present  and eff_icd  < MIN_ICD_CONFIDENCE:
        threshold_violations.append(f"ICD_BELOW_MIN({eff_icd:.2f}<{MIN_ICD_CONFIDENCE})")
    if cpt_present  and eff_cpt  < MIN_CPT_CONFIDENCE:
        threshold_violations.append(f"CPT_BELOW_MIN({eff_cpt:.2f}<{MIN_CPT_CONFIDENCE})")
    if drug_present and eff_drug < MIN_DRUG_CONFIDENCE:
        threshold_violations.append(f"DRUG_BELOW_MIN({eff_drug:.2f}<{MIN_DRUG_CONFIDENCE})")

    # Human-readable scoring logic string for audit trail
    parts = []
    if icd_present:
        parts.append(f"ICD({eff_icd:.2f}x{w_icd:.2f}={w_icd*eff_icd:.3f})")
    else:
        parts.append("ICD[absent->weight redistributed]")
    if cpt_present:
        parts.append(f"CPT({eff_cpt:.2f}x{w_cpt:.2f}={w_cpt*eff_cpt:.3f})")
    else:
        parts.append("CPT[absent->weight redistributed]")
    if drug_present:
        parts.append(f"Drug({eff_drug:.2f}x{w_drug:.2f}={w_drug*eff_drug:.3f})")
    else:
        parts.append("Drug[absent->weight redistributed]")
    parts.append(f"Policy({eff_policy:.2f}x{w_policy:.2f}={w_policy*eff_policy:.3f})")

    scoring_logic = " + ".join(parts) + f" = raw {raw_score:.4f}"
    if ambiguity_count > 0:
        scoring_logic += (
            f" - ambiguity_penalty({ambiguity_count} issues, 30% of raw = {ambiguity_penalty:.4f})"
            f" = final {total_score:.4f}"
        )
    else:
        scoring_logic += f" = final {total_score:.4f} (no ambiguity penalty)"

    return {
        "total":               total_score,
        "raw_score":           raw_score,
        "ambiguity_penalty":   ambiguity_penalty,
        "weights_used":        {
            "icd":    round(w_icd, 4),
            "cpt":    round(w_cpt, 4),
            "drug":   round(w_drug, 4),
            "policy": round(w_policy, 4),
        },
        "component_scores":    {
            "icd":    eff_icd,
            "cpt":    eff_cpt,
            "drug":   eff_drug,
            "policy": eff_policy,
        },
        "scoring_logic":        scoring_logic,
        "threshold_violations": threshold_violations,
    }


# ── Legacy helper kept for backward compatibility ─────────────────────────────
def combine_confidences(*scores) -> float:
    """Simple mean — kept for callers not yet on compute_weighted_score()."""
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 2) if valid else 0.0
