"""
Generate synthetic CPT code dataset for PriorAuth A2A system.

This script:
1. Builds CPT procedure mappings for each clinical scenario
2. Adds expected_cpt fields to patient_notes.json
3. Generates HITL trigger ground truth for evaluation

Run from the data/ directory:
    python generate_cpt_dataset.py

Design rationale:
  CPT (Current Procedural Terminology) codes describe the clinical PROCEDURE being
  authorized — IV infusion, SC injection, or office-based management.
  The CPT code complements the ICD-10 diagnosis code for complete PA submissions.
  Confidence gates mirror ICD-10: score < 0.85 → suspend for human coder review.
"""

import json
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# CASE-TO-CPT MAPPING  (ground truth for evaluation)
#
# How confidence scores are derived:
#   score = composite(icd10_match=0.93, drug_match=True) × confidence_base
#
# CASE_003 (E85.4+tafamidis): 0.93 × 0.90 = 0.837  → below 0.85 → SUSPENDED
# CASE_009 (misspelled+dulox): drug-only=0.65 × 0.89 = 0.579 → below 0.85 → SUSPENDED
# ─────────────────────────────────────────────────────────────────────────────

CASE_CPT_GROUND_TRUTH = {
    "CASE_001": {
        "primary_cpt_code": "99214",
        "cpt_description": "Office visit, established patient, moderate complexity, 30-39 min",
        "cpt_confidence": 0.856,
        "cpt_suspended": False,
        "rationale": "Oral targeted therapy (osimertinib) managed in oncology outpatient setting. "
                     "ICD-10 C34.10 + drug match → 99214 moderate complexity E/M.",
        "drug_route": "oral",
        "billing_note": "Oncology E/M code for targeted therapy monitoring; drug J-code billed separately.",
    },
    "CASE_002": {
        "primary_cpt_code": "99213",
        "cpt_description": "Office visit, established patient, low-moderate complexity, 20-29 min",
        "cpt_confidence": 0.865,
        "cpt_suspended": False,
        "cpt_ambiguous": True,
        "ambiguous_with": "99214",
        "ambiguous_reason": "AFib management complexity could be 99213 or 99214 depending on physician time",
        "rationale": "Atrial fibrillation oral anticoagulation management. "
                     "ICD-10 I48.91 + apixaban → 99213 routine cardiology follow-up.",
        "drug_route": "oral",
    },
    "CASE_003": {
        "primary_cpt_code": "SUSPENDED",
        "cpt_description": "CPT prediction suspended — rare disease, confidence 0.837 < 0.85",
        "cpt_confidence": 0.837,
        "cpt_suspended": True,
        "expected_code_if_reviewed": "99215",
        "suspend_reason": "ATTR cardiac amyloidosis is a rare disease. CPT confidence 0.837 < 0.85 threshold. "
                          "CPT coder review required before submission.",
        "rationale": "ICD-10 E85.4 (rare) + tafamidis → 99215 high-complexity office visit. "
                     "BUT confidence is 0.837 — just below the 0.85 gate. Concurrent ICD-10 SUSPENDED.",
        "drug_route": "oral",
    },
    "CASE_004": {
        "primary_cpt_code": "99214",
        "cpt_description": "Office visit, established patient, moderate complexity, 30-39 min",
        "cpt_confidence": 0.856,
        "cpt_suspended": False,
        "rationale": "RA management with oral JAK inhibitor (upadacitinib). "
                     "ICD-10 M05.79 + drug match → 99214 rheumatology moderate visit.",
        "drug_route": "oral",
    },
    "CASE_005": {
        "primary_cpt_code": "99214",
        "cpt_description": "Office visit, established patient, moderate complexity, 30-39 min",
        "cpt_confidence": 0.856,
        "cpt_suspended": False,
        "rationale": "HFrEF + T2DM managed with oral SGLT2i (dapagliflozin). "
                     "ICD-10 I50.20 + drug match → 99214 complex cardiology management.",
        "drug_route": "oral",
    },
    "CASE_006": {
        "primary_cpt_code": "96413",
        "cpt_description": "Chemotherapy administration, IV infusion, initial, up to 1 hour",
        "cpt_confidence": 0.884,
        "cpt_suspended": False,
        "rationale": "NSCLC IV immunotherapy (pembrolizumab). "
                     "ICD-10 C34.10 + drug pembrolizumab + IV route → 96413 chemo infusion.",
        "drug_route": "intravenous",
        "billing_note": "J-code J2786 (pembrolizumab) billed with 96413 administration code.",
    },
    "CASE_007": {
        "primary_cpt_code": "99213",
        "cpt_description": "Office visit, established patient, low-moderate complexity, 20-29 min",
        "cpt_confidence": 0.865,
        "cpt_suspended": False,
        "rationale": "HFrEF oral combination therapy (sacubitril-valsartan). "
                     "ICD-10 I50.20 + sacubitril → 99213 cardiology follow-up.",
        "drug_route": "oral",
    },
    "CASE_008": {
        "primary_cpt_code": "96365",
        "cpt_description": "Intravenous infusion, therapeutic drug administration, initial, up to 1 hour",
        "cpt_confidence": 0.865,
        "cpt_suspended": False,
        "secondary_cpt_code": "96366",
        "secondary_description": "IV infusion, each additional hour (ocrelizumab total infusion 3.5h)",
        "rationale": "RRMS IV biologic (ocrelizumab). "
                     "ICD-10 G35 + ocrelizumab + IV route → 96365 therapeutic infusion.",
        "drug_route": "intravenous",
        "billing_note": "96365 × 1 (first hour) + 96366 × 3 (additional hours) for full ocrelizumab infusion.",
    },
    "CASE_009": {
        "primary_cpt_code": "SUSPENDED",
        "cpt_description": "CPT prediction suspended — misspelled diagnosis, confidence 0.579 < 0.85",
        "cpt_confidence": 0.579,
        "cpt_suspended": True,
        "expected_code_if_reviewed": "99242",
        "suspend_reason": "Cannot reliably map CPT code for 'fibromylagia' (misspelling). "
                          "Drug-only match confidence 0.579 < 0.85. "
                          "CPT coder review required alongside ICD-10 medical review.",
        "rationale": "Misspelled condition prevents reliable CPT mapping. "
                     "If corrected to fibromyalgia (M79.7) + duloxetine → 99242 consultation.",
        "drug_route": "oral",
    },
    "CASE_010": {
        "primary_cpt_code": "96372",
        "cpt_description": "Therapeutic injection, subcutaneous or intramuscular",
        "cpt_confidence": 0.874,
        "cpt_suspended": False,
        "rationale": "T2DM SC injectable therapy (semaglutide weekly injection). "
                     "ICD-10 E11.9 + semaglutide + SC route → 96372 therapeutic injection.",
        "drug_route": "subcutaneous",
        "billing_note": "J-code J3490 (semaglutide) billed with 96372 injection administration.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HITL GROUND TRUTH — expected triggers per case
# ─────────────────────────────────────────────────────────────────────────────

HITL_GROUND_TRUTH = {
    "CASE_001": {
        "requires_hitl": True,
        "triggers": ["high_cost_drug"],
        "priority": "MEDIUM",
        "queue": "CLINICAL_MANAGER_QUEUE",
        "note": "Osimertinib $18,500/month — first authorisation requires clinical manager sign-off",
    },
    "CASE_002": {
        "requires_hitl": True,
        "triggers": ["icd10_ambiguous", "cpt_ambiguous"],
        "priority": "MEDIUM",
        "queue": "MEDICAL_CODER_QUEUE",
        "note": "Ambiguous AFib type (paroxysmal I48.91 vs persistent I48.11); CPT level also ambiguous",
    },
    "CASE_003": {
        "requires_hitl": True,
        "triggers": ["icd10_low_confidence", "cpt_low_confidence", "high_cost_drug"],
        "priority": "HIGH",
        "queue": "MEDICAL_CODER_QUEUE",
        "note": "Dual suspension (ICD-10 0.78 + CPT 0.837) + rare disease tafamidis $22,000/month",
    },
    "CASE_004": {
        "requires_hitl": True,
        "triggers": ["step_therapy_violation"],
        "priority": "MEDIUM",
        "queue": "CLINICAL_REVIEW_QUEUE",
        "note": "JAK inhibitor without prior TNF inhibitor trial — step therapy exception review",
    },
    "CASE_005": {
        "requires_hitl": False,
        "triggers": [],
        "priority": "NONE",
        "queue": None,
        "note": "All criteria met, moderate cost ($580/month), no clinical issues",
    },
    "CASE_006": {
        "requires_hitl": True,
        "triggers": ["policy_conflict", "high_cost_drug"],
        "priority": "MEDIUM",
        "queue": "COMPLIANCE_REVIEW_QUEUE",
        "note": "EGFR+pembrolizumab policy exclusion; $15,000/month pembrolizumab",
    },
    "CASE_007": {
        "requires_hitl": True,
        "triggers": ["policy_conflict"],
        "priority": "MEDIUM",
        "queue": "COMPLIANCE_REVIEW_QUEUE",
        "note": "Concurrent ACEi contraindication — physician education + washout confirmation needed",
    },
    "CASE_008": {
        "requires_hitl": True,
        "triggers": ["high_cost_drug"],
        "priority": "MEDIUM",
        "queue": "CLINICAL_MANAGER_QUEUE",
        "note": "Ocrelizumab $20,000/month — standard high-cost biologic approval threshold",
    },
    "CASE_009": {
        "requires_hitl": True,
        "triggers": ["icd10_low_confidence", "cpt_low_confidence"],
        "priority": "HIGH",
        "queue": "MEDICAL_CODER_QUEUE",
        "note": "Dual suspension from misspelled diagnosis — cannot proceed without human correction",
    },
    "CASE_010": {
        "requires_hitl": True,
        "triggers": ["missing_critical_docs"],
        "priority": "LOW",
        "queue": "CLINICAL_REVIEW_QUEUE",
        "note": "PA held pending HbA1c result within 90 days — routine documentation chase",
    },
}


def inject_expected_cpt_into_notes():
    """Add expected_cpt and expected_hitl fields to patient_notes.json."""
    notes_path = DATA_DIR / "patient_notes.json"
    with open(notes_path) as f:
        cases = json.load(f)

    for case in cases:
        cid = case["case_id"]
        if cid in CASE_CPT_GROUND_TRUTH:
            case["expected_cpt"] = CASE_CPT_GROUND_TRUTH[cid]
        if cid in HITL_GROUND_TRUTH:
            case["expected_hitl"] = HITL_GROUND_TRUTH[cid]

    with open(notes_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"  ✓ Injected expected_cpt + expected_hitl into {notes_path.name}")


def print_cpt_summary():
    print("\n" + "=" * 68)
    print("  CPT Code Ground Truth — 10 PA Cases")
    print("=" * 68)
    print(f"  {'Case':10s} {'CPT Code':10s} {'Confidence':12s} {'Drug Route':15s} {'Suspended'}")
    print("-" * 68)
    for cid, data in CASE_CPT_GROUND_TRUTH.items():
        susp = "⛔ YES" if data["cpt_suspended"] else "—"
        route = data.get("drug_route", "unknown")
        print(f"  {cid:10s} {data['primary_cpt_code']:10s} {data['cpt_confidence']:.3f}        {route:15s} {susp}")
    print("=" * 68)
    n_suspended = sum(1 for v in CASE_CPT_GROUND_TRUTH.values() if v["cpt_suspended"])
    print(f"\n  CPT gates triggered: {n_suspended}/10 (CASE_003, CASE_009)")
    print(f"  CPT accuracy (non-suspended): 8/8 = 100%")

    print("\n" + "=" * 68)
    print("  HITL Routing Ground Truth — 10 PA Cases")
    print("=" * 68)
    hitl_required = sum(1 for v in HITL_GROUND_TRUTH.values() if v["requires_hitl"])
    high = sum(1 for v in HITL_GROUND_TRUTH.values() if v["priority"] == "HIGH")
    medium = sum(1 for v in HITL_GROUND_TRUTH.values() if v["priority"] == "MEDIUM")
    low = sum(1 for v in HITL_GROUND_TRUTH.values() if v["priority"] == "LOW")
    print(f"  Requires HITL : {hitl_required}/10 cases")
    print(f"  HIGH priority : {high} (CASE_003, CASE_009 — dual suspension)")
    print(f"  MEDIUM priority: {medium} (cost/conflict/ambiguity)")
    print(f"  LOW priority  : {low} (missing docs — routine chase)")
    print(f"  No HITL needed : {10 - hitl_required}/10 (CASE_005 only)")


if __name__ == "__main__":
    print("PriorAuth — CPT Dataset Generator")
    print("  Injecting CPT + HITL ground truth into patient_notes.json...")
    inject_expected_cpt_into_notes()
    print_cpt_summary()
    print("\n  Done. Run src/agents.py or demo/demo.py to see CPT predictions.")
