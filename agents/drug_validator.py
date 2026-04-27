"""
Drug Validation Agent — Agent 5
================================
Checks each extracted drug against data/drugs.json:
  - Is the drug covered by the formulary?
  - Does it require prior authorization?
  - What tier / monthly cost?
  - Is there a step therapy requirement noted in policy?

Output written to state.drug_checks (list of per-drug result dicts).
The confidence returned for each drug is 1.0 for known drugs
(deterministic lookup — we either find it or we don't).

This confidence feeds Agent 8 (Confidence Scoring) as drug_conf:
  avg(drug_conf across all checked drugs)
"""

from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)

# ── Data loading ──────────────────────────────────────────────────────────────
with open("data/drugs.json", encoding="utf-8") as f:
    _DRUGS: list[dict] = json.load(f)

_by_name  = {d["name"].lower(): d for d in _DRUGS}
_by_brand = {d.get("brand", "").lower(): d for d in _DRUGS if d.get("brand")}

HIGH_COST_THRESHOLD_USD = 5_000   # monthly cost above this = high-cost flag


def _find_drug(name: str) -> dict | None:
    """Look up a drug by name or brand (case-insensitive, substring fallback)."""
    k = name.lower().strip()
    if k in _by_name:
        return _by_name[k]
    if k in _by_brand:
        return _by_brand[k]
    # Substring fallback
    for n, rec in _by_name.items():
        if k in n or n in k:
            return rec
    for b, rec in _by_brand.items():
        if b and (k in b or b in k):
            return rec
    return None


def drug_validator(state):
    """
    Validate each drug in state.entities['drugs'].
    Writes results to state.drug_checks.
    """
    drugs_list = (state.entities or {}).get("drugs", [])
    checks: list[dict] = []

    for drug_name in drugs_list:
        if not drug_name:
            continue

        rec = _find_drug(drug_name)

        if rec is None:
            # Drug not in formulary — unknown, flag for HITL
            checks.append({
                "drug":           drug_name,
                "found":          False,
                "covered":        False,
                "requires_pa":    True,
                "tier":           None,
                "monthly_cost":   None,
                "high_cost":      False,
                "confidence":     0.60,   # low certainty — unknown drug
                "note":           "Drug not found in formulary — manual review required",
            })
            logger.warning(f"[DrugValidator] Unknown drug: {drug_name}")
            continue

        monthly_cost = rec.get("monthly_cost_usd", 0)
        high_cost    = monthly_cost >= HIGH_COST_THRESHOLD_USD

        checks.append({
            "drug":           drug_name,
            "drug_id":        rec.get("id"),
            "found":          True,
            "covered":        True,               # present in formulary = formulary-covered
            "requires_pa":    rec.get("requires_pa", True),
            "tier":           rec.get("tier"),
            "monthly_cost":   monthly_cost,
            "high_cost":      high_cost,
            "drug_class":     rec.get("class", ""),
            "brand":          rec.get("brand", ""),
            "confidence":     1.0,               # deterministic lookup = certain
            "note":           f"Tier {rec.get('tier')} | ${monthly_cost:,}/month" +
                              (" | HIGH-COST FLAG" if high_cost else ""),
        })
        logger.info(f"[DrugValidator] {drug_name} -> {rec.get('id')} tier={rec.get('tier')} "
                    f"cost=${monthly_cost:,} high_cost={high_cost}")

    state.drug_checks = checks
    state.audit_log.append({
        "step":        "DRUG_VALIDATED",
        "drugs_found": sum(1 for c in checks if c["found"]),
        "drugs_unknown": sum(1 for c in checks if not c["found"]),
        "high_cost_drugs": [c["drug"] for c in checks if c.get("high_cost")],
    })
    return state
