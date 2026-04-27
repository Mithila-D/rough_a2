"""
PriorAuth — Synthetic Dataset Generator
=========================================
Generates all datasets for the multi-agent prior authorization workflow.

Datasets:
  patient_notes.json       ← Messy unstructured doctor notes with injected issues
  icd10_knowledge_graph.json ← Simplified KG: condition → ICD-10 code mappings
  policy_documents.json    ← Insurer policy rules per condition/drug
  pa_cases.json            ← 10 labeled PA cases with ground truth outcomes
  schema_definition.json   ← The JSON schema the final output must satisfy

INJECTED PATTERNS (motivate A2A over monolithic LLM):
  [AMBIGUOUS_CODE]    Condition maps to multiple ICD-10 codes — Agent 2 must clarify
  [LOW_CONFIDENCE]    Rare/misspelled condition — Agent 2 confidence <0.90 → SUSPEND
  [PHI_LEAK]         Patient note contains raw PII — guardrail must mask before LLM
  [POLICY_CONFLICT]  Drug approved for one code but not another — Agent 3 catches it
  [MULTI_AGENT_HANDOFF] Data flows across 3 agents — test inter-agent traceability
  [FORM_SCHEMA_FAIL]  Missing required field — JSON validator catches it
"""

import json
import random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# ICD-10 KNOWLEDGE GRAPH  (simplified — condition → code mappings with metadata)
# ─────────────────────────────────────────────────────────────────────────────

ICD10_KG = [
    # Cardiology
    {"id": "ICD_001", "code": "I50.20", "description": "Unspecified systolic heart failure",
     "aliases": ["HFrEF", "heart failure reduced EF", "systolic HF", "congestive heart failure"],
     "category": "cardiology", "billable": True, "confidence_base": 0.95,
     "related_drugs": ["DR_DAPA", "DR_EMPA", "DR_SACVAL", "DR_BISOPROLOL"],
     "challenge_tags": []},
    {"id": "ICD_002", "code": "I10", "description": "Essential hypertension",
     "aliases": ["hypertension", "HTN", "high blood pressure", "elevated BP"],
     "category": "cardiology", "billable": True, "confidence_base": 0.98,
     "related_drugs": ["DR_RAMIPRIL", "DR_AMLODIPINE", "DR_LOSARTAN"],
     "challenge_tags": []},
    {"id": "ICD_003", "code": "I21.3", "description": "ST elevation myocardial infarction of unspecified site",
     "aliases": ["STEMI", "ST elevation MI", "heart attack"],
     "category": "cardiology", "billable": True, "confidence_base": 0.97,
     "related_drugs": ["DR_ASPIRIN", "DR_CLOPIDOGREL", "DR_TICAGRELOR"],
     "challenge_tags": []},
    {"id": "ICD_004", "code": "I48.91", "description": "Unspecified atrial fibrillation",
     "aliases": ["AFib", "A-fib", "atrial fibrillation", "AF"],
     "category": "cardiology", "billable": True, "confidence_base": 0.96,
     "related_drugs": ["DR_APIXABAN", "DR_RIVAROXABAN", "DR_WARFARIN"],
     "challenge_tags": ["AMBIGUOUS_CODE"]},
    {"id": "ICD_004B", "code": "I48.11", "description": "Longstanding persistent atrial fibrillation",
     "aliases": ["persistent AFib", "longstanding AF", "chronic atrial fibrillation"],
     "category": "cardiology", "billable": True, "confidence_base": 0.88,
     "related_drugs": ["DR_APIXABAN", "DR_RIVAROXABAN"],
     "challenge_tags": ["AMBIGUOUS_CODE"],
     "note": "Requires clinical clarification — persistent vs paroxysmal changes code"},

    # Oncology
    {"id": "ICD_005", "code": "C34.10", "description": "Malignant neoplasm of upper lobe bronchus and lung",
     "aliases": ["NSCLC", "non-small cell lung cancer", "lung adenocarcinoma", "lung cancer"],
     "category": "oncology", "billable": True, "confidence_base": 0.93,
     "related_drugs": ["DR_OSIMERTINIB", "DR_PEMBROLIZUMAB", "DR_CARBOPLATIN"],
     "challenge_tags": []},
    {"id": "ICD_006", "code": "C50.919", "description": "Malignant neoplasm of unspecified site of unspecified female breast",
     "aliases": ["breast cancer", "HER2+ breast cancer", "TNBC", "triple negative breast cancer"],
     "category": "oncology", "billable": True, "confidence_base": 0.91,
     "related_drugs": ["DR_TRASTUZUMAB", "DR_PALBOCICLIB", "DR_OLAPARIB"],
     "challenge_tags": ["AMBIGUOUS_CODE"],
     "note": "HER2+ vs TNBC maps to same base code but different drug approvals"},

    # Endocrinology
    {"id": "ICD_007", "code": "E11.65", "description": "Type 2 diabetes mellitus with hyperglycemia and chronic kidney disease",
     "aliases": ["T2DM with CKD", "diabetic nephropathy", "type 2 diabetes CKD"],
     "category": "endocrinology", "billable": True, "confidence_base": 0.94,
     "related_drugs": ["DR_FINERENONE", "DR_DAPAGLIFLOZIN", "DR_SEMAGLUTIDE"],
     "challenge_tags": []},
    {"id": "ICD_008", "code": "E11.9", "description": "Type 2 diabetes mellitus without complications",
     "aliases": ["T2DM", "type 2 diabetes", "diabetes mellitus type 2", "DM2"],
     "category": "endocrinology", "billable": True, "confidence_base": 0.97,
     "related_drugs": ["DR_METFORMIN", "DR_SEMAGLUTIDE", "DR_SITAGLIPTIN"],
     "challenge_tags": []},

    # Rheumatology
    {"id": "ICD_009", "code": "M05.79", "description": "Rheumatoid arthritis with rheumatoid factor of multiple sites",
     "aliases": ["RA", "rheumatoid arthritis", "seropositive RA", "inflammatory arthritis"],
     "category": "rheumatology", "billable": True, "confidence_base": 0.95,
     "related_drugs": ["DR_ADALIMUMAB", "DR_METHOTREXATE", "DR_UPADACITINIB"],
     "challenge_tags": []},
    {"id": "ICD_010", "code": "M32.9", "description": "Systemic lupus erythematosus, unspecified",
     "aliases": ["SLE", "lupus", "systemic lupus"],
     "category": "rheumatology", "billable": True, "confidence_base": 0.92,
     "related_drugs": ["DR_HYDROXYCHLOROQUINE", "DR_BELIMUMAB"],
     "challenge_tags": []},

    # Rare / Low Confidence (triggers SUSPEND workflow)
    {"id": "ICD_011", "code": "E85.4", "description": "Organ-limited amyloidosis",
     "aliases": ["ATTR amyloidosis", "transthyretin amyloidosis", "cardiac amyloidosis"],
     "category": "rare_disease", "billable": True, "confidence_base": 0.78,
     "related_drugs": ["DR_TAFAMIDIS"],
     "challenge_tags": ["LOW_CONFIDENCE"],
     "note": "Rare — high risk of miscoding. Agent 2 confidence <0.90 → SUSPEND"},
    {"id": "ICD_012", "code": "G35", "description": "Multiple sclerosis",
     "aliases": ["MS", "multiple sclerosis", "RRMS", "relapsing remitting MS"],
     "category": "neurology", "billable": True, "confidence_base": 0.96,
     "related_drugs": ["DR_OCRELIZUMAB", "DR_NATALIZUMAB"],
     "challenge_tags": []},
    {"id": "ICD_013", "code": "J44.1", "description": "Chronic obstructive pulmonary disease with acute exacerbation",
     "aliases": ["COPD exacerbation", "COPD", "chronic obstructive pulmonary disease"],
     "category": "pulmonology", "billable": True, "confidence_base": 0.95,
     "related_drugs": ["DR_TIOTROPIUM", "DR_BUDESONIDE_FORM"],
     "challenge_tags": []},

    # Ambiguous / Misspelled (tests fuzzy matching)
    {"id": "ICD_014", "code": "UNKNOWN", "description": "Unknown condition",
     "aliases": ["fibromylagia", "fibromyalga", "fybromyalgia"],  # deliberate misspellings
     "category": "rheumatology", "billable": False, "confidence_base": 0.42,
     "related_drugs": [],
     "challenge_tags": ["LOW_CONFIDENCE"],
     "note": "Misspelled — confidence will be very low — triggers SUSPEND + manual review"},
    {"id": "ICD_015", "code": "M79.3", "description": "Panniculitis, unspecified",
     "aliases": ["fibromyalgia", "FM", "widespread pain syndrome"],
     "category": "rheumatology", "billable": True, "confidence_base": 0.62,
     "related_drugs": ["DR_DULOXETINE", "DR_PREGABALIN"],
     "challenge_tags": ["LOW_CONFIDENCE"],
     "note": "Fibromyalgia ICD code is actually M79.7 — common billing error, confidence reduced"},
]

# ─────────────────────────────────────────────────────────────────────────────
# DRUG FORMULARY (referenced by policy docs)
# ─────────────────────────────────────────────────────────────────────────────

DRUGS = [
    {"id": "DR_DAPA", "name": "Dapagliflozin", "brand": "Farxiga", "class": "SGLT2i",
     "tier": 3, "monthly_cost_usd": 580, "requires_pa": True},
    {"id": "DR_EMPA", "name": "Empagliflozin", "brand": "Jardiance", "class": "SGLT2i",
     "tier": 3, "monthly_cost_usd": 590, "requires_pa": True},
    {"id": "DR_OSIMERTINIB", "name": "Osimertinib", "brand": "Tagrisso", "class": "EGFR TKI",
     "tier": 5, "monthly_cost_usd": 18500, "requires_pa": True},
    {"id": "DR_PEMBROLIZUMAB", "name": "Pembrolizumab", "brand": "Keytruda", "class": "PD-1 inhibitor",
     "tier": 5, "monthly_cost_usd": 15000, "requires_pa": True},
    {"id": "DR_TRASTUZUMAB", "name": "Trastuzumab", "brand": "Herceptin", "class": "HER2 mAb",
     "tier": 5, "monthly_cost_usd": 8000, "requires_pa": True},
    {"id": "DR_ADALIMUMAB", "name": "Adalimumab", "brand": "Humira", "class": "TNF inhibitor",
     "tier": 4, "monthly_cost_usd": 6200, "requires_pa": True},
    {"id": "DR_OCRELIZUMAB", "name": "Ocrelizumab", "brand": "Ocrevus", "class": "Anti-CD20",
     "tier": 5, "monthly_cost_usd": 22000, "requires_pa": True},
    {"id": "DR_TAFAMIDIS", "name": "Tafamidis", "brand": "Vyndamax", "class": "TTR stabilizer",
     "tier": 5, "monthly_cost_usd": 21000, "requires_pa": True},
    {"id": "DR_APIXABAN", "name": "Apixaban", "brand": "Eliquis", "class": "DOAC",
     "tier": 3, "monthly_cost_usd": 480, "requires_pa": True},
    {"id": "DR_SACVAL", "name": "Sacubitril-valsartan", "brand": "Entresto", "class": "ARNI",
     "tier": 4, "monthly_cost_usd": 680, "requires_pa": True},
    {"id": "DR_FINERENONE", "name": "Finerenone", "brand": "Kerendia", "class": "MRA",
     "tier": 4, "monthly_cost_usd": 540, "requires_pa": True},
    {"id": "DR_SEMAGLUTIDE", "name": "Semaglutide", "brand": "Ozempic", "class": "GLP-1 RA",
     "tier": 3, "monthly_cost_usd": 890, "requires_pa": True},
    {"id": "DR_UPADACITINIB", "name": "Upadacitinib", "brand": "Rinvoq", "class": "JAK1i",
     "tier": 4, "monthly_cost_usd": 5400, "requires_pa": True},
    {"id": "DR_METFORMIN", "name": "Metformin", "brand": "Glucophage", "class": "Biguanide",
     "tier": 1, "monthly_cost_usd": 12, "requires_pa": False},
    {"id": "DR_RAMIPRIL", "name": "Ramipril", "brand": "Altace", "class": "ACEi",
     "tier": 1, "monthly_cost_usd": 18, "requires_pa": False},
]

# ─────────────────────────────────────────────────────────────────────────────
# INSURER POLICY DOCUMENTS (Agent 3 - Agentic RAG source)
# ─────────────────────────────────────────────────────────────────────────────

POLICY_DOCS = [
    {
        "id": "POL_001",
        "title": "BlueStar Health — SGLT2 Inhibitor Coverage Policy",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_DAPA", "DR_EMPA"],
        "approved_icd10_codes": ["E11.65", "I50.20", "E11.9"],
        "body": (
            "Dapagliflozin and empagliflozin are covered under the BlueStar formulary for the following indications: "
            "(1) Type 2 diabetes mellitus (E11.9, E11.65) when HbA1c ≥7.5% documented AND ≥1 trial of metformin "
            "for ≥3 months with inadequate glycaemic control. "
            "(2) Heart failure with reduced ejection fraction (I50.20) with LVEF ≤40% documented by echocardiogram, "
            "regardless of diabetes status. "
            "(3) CKD stage 3b or higher (eGFR 25–44) with albuminuria. "
            "STEP THERAPY REQUIRED for diabetes indication: must have failed metformin (tier 1) first. "
            "EXCLUSIONS: eGFR <20 mL/min/1.73m² (absolute contraindication). "
            "DOCUMENTATION REQUIRED: recent HbA1c, echo report for HF indication, eGFR within 90 days."
        ),
        "step_therapy_required": True,
        "documentation_required": ["HbA1c_recent", "echo_if_HF", "eGFR_90days"],
        "challenge_tags": ["POLICY_CONFLICT"],
        "injected_issue": "Dapagliflozin for NSCLC (C34.10) — NOT an approved indication → DENY"
    },
    {
        "id": "POL_002",
        "title": "BlueStar Health — Oncology Biologics Policy: EGFR TKI and PD-1 Inhibitors",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_OSIMERTINIB", "DR_PEMBROLIZUMAB"],
        "approved_icd10_codes": ["C34.10", "C34.11", "C34.12"],
        "body": (
            "Osimertinib (Tagrisso) is approved for NSCLC (ICD-10 C34.xx) when: "
            "(1) EGFR exon 19 deletion or exon 21 L858R mutation confirmed by validated molecular testing. "
            "(2) Stage IIIB–IV or recurrent disease. "
            "(3) Pathology report and molecular testing report must be submitted. "
            "Pembrolizumab (Keytruda) for NSCLC: requires PD-L1 TPS ≥50% OR MSI-H confirmed by IHC 22C3. "
            "Pembrolizumab is NOT approved for NSCLC if concurrent EGFR sensitizing mutation is present. "
            "STEP THERAPY: osimertinib does NOT require prior chemotherapy step for EGFR-mutant NSCLC. "
            "DOCUMENTATION: molecular testing report, pathology confirming stage, oncologist attestation."
        ),
        "step_therapy_required": False,
        "documentation_required": ["molecular_testing_report", "pathology_stage", "oncologist_attestation"],
        "challenge_tags": ["POLICY_CONFLICT"],
        "injected_issue": "Pembrolizumab + concurrent EGFR mutation → policy conflict → DENY/QUERY"
    },
    {
        "id": "POL_003",
        "title": "BlueStar Health — TNF Inhibitor and JAK Inhibitor Coverage: RA",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_ADALIMUMAB", "DR_UPADACITINIB"],
        "approved_icd10_codes": ["M05.79", "M05.9", "M06.9"],
        "body": (
            "Adalimumab (Humira) is approved for moderate-to-severe rheumatoid arthritis (M05.xx, M06.xx): "
            "DAS28 score ≥3.2 documented, AND failure of at least one conventional DMARD "
            "(methotrexate ≥15mg/week × 3 months OR sulfasalazine or hydroxychloroquine). "
            "Upadacitinib (Rinvoq) is approved as SECOND-LINE biologic: must have failed ≥1 prior TNF inhibitor. "
            "CONTRAINDICATION: upadacitinib NOT approved if active hepatitis B or lymphoma history. "
            "Biosimilar step required: adalimumab biosimilar must be tried before branded Humira. "
            "DOCUMENTATION REQUIRED: DAS28 score, prior DMARD trial documentation, TB screen negative."
        ),
        "step_therapy_required": True,
        "documentation_required": ["DAS28_score", "prior_DMARD_trial", "TB_screen_negative"],
        "challenge_tags": [],
        "injected_issue": "Upadacitinib requested without prior TNF inhibitor trial → DENY step therapy"
    },
    {
        "id": "POL_004",
        "title": "BlueStar Health — DOAC Coverage Policy: Atrial Fibrillation",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_APIXABAN"],
        "approved_icd10_codes": ["I48.91", "I48.11", "I48.19", "I48.0"],
        "body": (
            "Apixaban (Eliquis) is approved for stroke prevention in non-valvular atrial fibrillation (I48.xx). "
            "Approval criteria: CHA₂DS₂-VASc score ≥2 (males) or ≥3 (females) documented. "
            "eGFR must be ≥25 mL/min; if eGFR 25–49, dose reduction to 2.5mg BD applies. "
            "NOT approved for: mechanical heart valves (warfarin only), antiphospholipid syndrome. "
            "STEP THERAPY: warfarin trial NOT required; DOAC is first-line for non-valvular AF. "
            "DOCUMENTATION: ECG or Holter confirming AF, CHA₂DS₂-VASc calculation, eGFR within 90 days."
        ),
        "step_therapy_required": False,
        "documentation_required": ["ECG_confirming_AF", "CHA2DS2_VASc", "eGFR_90days"],
        "challenge_tags": ["AMBIGUOUS_CODE"],
        "injected_issue": "I48.91 (unspecified AFib) triggers code clarification request before approval"
    },
    {
        "id": "POL_005",
        "title": "BlueStar Health — ATTR Amyloidosis: Tafamidis Coverage",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_TAFAMIDIS"],
        "approved_icd10_codes": ["E85.4", "I43"],
        "body": (
            "Tafamidis (Vyndamax/Vyndaqel) is approved for transthyretin (ATTR) cardiac amyloidosis (E85.4). "
            "Approval criteria: (1) Confirmed diagnosis: Technetium-99m PYP scan grade 2 or 3, OR "
            "endomyocardial biopsy with Congo red positive and TTR immunohistochemistry. "
            "(2) Genetic testing to determine hereditary (hATTR) vs wild-type (wtATTR) subtype. "
            "(3) NYHA class I–III heart failure. "
            "(4) Cardiology specialist (heart failure) attestation required. "
            "EXCLUSIONS: NYHA class IV, concurrent use of diflunisal or other TTR stabilisers. "
            "Annual re-authorisation required with 6MWT and echocardiographic data."
        ),
        "step_therapy_required": False,
        "documentation_required": ["PYP_scan_or_biopsy", "genetic_testing_TTR", "NYHA_class_documented", "cardiology_attestation"],
        "challenge_tags": ["LOW_CONFIDENCE"],
        "injected_issue": "Rare disease — ICD-10 confidence <0.90 for ATTR amyloidosis → SUSPEND"
    },
    {
        "id": "POL_006",
        "title": "BlueStar Health — Anti-CD20 Coverage: Multiple Sclerosis",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_OCRELIZUMAB"],
        "approved_icd10_codes": ["G35"],
        "body": (
            "Ocrelizumab (Ocrevus) is approved for relapsing or primary progressive MS (G35). "
            "Criteria: MRI-confirmed diagnosis per McDonald 2017 criteria. "
            "Relapsing MS: ≥1 relapse in prior 12 months OR ≥1 new T2/Gd+ lesion on MRI. "
            "Primary progressive MS: EDSS 3.0–6.5, age ≤55 at treatment initiation. "
            "STEP THERAPY: no prior DMT step required for PPMS; "
            "for RRMS: interferon beta or glatiramer acetate failure preferred but not mandatory. "
            "DOCUMENTATION: MRI confirming diagnosis, neurologist diagnosis letter, "
            "hepatitis B serology (CD20 depletion risk), JC antibody index if natalizumab history."
        ),
        "step_therapy_required": False,
        "documentation_required": ["MRI_MS_diagnosis", "neurologist_letter", "hepatitis_B_serology"],
        "challenge_tags": []
    },
    {
        "id": "POL_007",
        "title": "BlueStar Health — HF Advanced Therapy: Sacubitril-Valsartan",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_SACVAL"],
        "approved_icd10_codes": ["I50.20", "I50.22", "I50.32"],
        "body": (
            "Sacubitril-valsartan (Entresto) is approved for HFrEF (LVEF ≤40%) or HFmrEF (LVEF 41–49%). "
            "Criteria: (1) LVEF ≤40% confirmed by echo, (2) NYHA class II–IV, "
            "(3) currently on stable ACEI or ARB for ≥4 weeks. "
            "WASHOUT PERIOD: 36-hour washout from ACEi required before initiation (angioedema risk). "
            "EXCLUSIONS: history of angioedema with ACEi/ARB, bilateral renal artery stenosis, "
            "concomitant ACEi use (absolutely contraindicated). "
            "DOCUMENTATION: echo LVEF, current medication list, NYHA class."
        ),
        "step_therapy_required": True,
        "documentation_required": ["echo_LVEF", "current_medication_list", "NYHA_class"],
        "challenge_tags": ["POLICY_CONFLICT"],
        "injected_issue": "Patient on concurrent ACEi + Entresto → policy conflict → DENY"
    },
    {
        "id": "POL_008",
        "title": "BlueStar Health — GLP-1 RA Coverage: Semaglutide",
        "insurer": "BlueStar Health",
        "effective_date": "2024-01-01",
        "drug_ids": ["DR_SEMAGLUTIDE"],
        "approved_icd10_codes": ["E11.9", "E11.65"],
        "body": (
            "Semaglutide (Ozempic) is approved for T2DM glycaemic control and cardiovascular risk reduction. "
            "Criteria: HbA1c ≥7.5% documented, AND metformin tried first (≥3 months unless contraindicated). "
            "For cardiovascular risk reduction indication: established ASCVD or CKD stage ≥3. "
            "EXCLUSIONS: personal or family history of medullary thyroid carcinoma (MTC), "
            "MEN2 syndrome, pancreatitis history. "
            "NOT approved for weight loss only without diabetes (separate obesity policy applies). "
            "DOCUMENTATION: HbA1c within 90 days, prior metformin trial evidence."
        ),
        "step_therapy_required": True,
        "documentation_required": ["HbA1c_90days", "metformin_prior_trial"],
        "challenge_tags": []
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PATIENT NOTES (10 cases — messy, real-world style, injected with issues)
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_NOTES = [

    # ── CASE 1: NSCLC + Osimertinib — CLEAN APPROVAL (baseline demo)
    {
        "case_id": "CASE_001",
        "challenge_tags": ["MULTI_AGENT_HANDOFF"],
        "expected_outcome": "APPROVED",
        "scenario_name": "NSCLC + EGFR mutation — straightforward approval",
        "raw_note": """
Patient: Rajesh Kumar, DOB: 15/06/1958, MRN: MH-4421876
Insurance: BlueStar Health, Member ID: BS-9871234

CLINICAL NOTE — Oncology OPD
Date: 14 Jan 2025
Physician: Dr Ananya Sharma, DM Oncology

62-year-old male, ex-smoker (30 pack-years). Diagnosed with non-small cell lung cancer,
adenocarcinoma histology, Stage IIIB–IV, right upper lobe. EGFR exon 19 deletion 
confirmed on Foundation One CDx NGS panel (report attached). PD-L1 TPS 10% (not meeting 
high expressors threshold). ECOG performance status 1.

Plan: Initiate osimertinib (Tagrisso) 80mg OD as first-line targeted therapy per FLAURA 
trial data. No prior systemic therapy.

Documentation enclosed: pathology report, NGS molecular testing, staging CT chest-abdomen-pelvis.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "DOB", "MRN", "member_id"],
        "expected_entities": {
            "condition": "Non-Small Cell Lung Cancer",
            "drug": "Osimertinib",
            "icd10_code": "C34.10",
            "confidence": 0.93,
            "policy_id": "POL_002",
            "outcome": "APPROVED",
            "missing_docs": []
        }
    },

    # ── CASE 2: AFib + Apixaban — AMBIGUOUS CODE (I48.91 vs I48.11)
    {
        "case_id": "CASE_002",
        "challenge_tags": ["AMBIGUOUS_CODE"],
        "expected_outcome": "QUERY — clarify paroxysmal vs persistent AF",
        "scenario_name": "Atrial fibrillation — ambiguous ICD-10 code requires clinical clarification",
        "raw_note": """
Patient: Fatima Sheikh, 68F, MRN: KA-7732901
Insurer: BlueStar Health, Member #: BS-2234567

Cardiology OPD — 20 Jan 2025
Dr Suresh Pillai, DM Cardiology

68-year-old female with AF — on rate control with bisoprolol. CHA₂DS₂-VASc score = 4 
(female, age, HTN, heart failure). eGFR 52 mL/min. Requesting apixaban 2.5mg BD for 
stroke prevention.

Note: patient has had AF episodes documented for 3 years. Recent Holter (7 days) shows 
continuous AF — may be persistent/longstanding. Prior ECG showing AF attached.

Plan: Initiate apixaban (Eliquis) 2.5mg BD (dose-reduced for eGFR 52 + age).
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "Atrial Fibrillation",
            "drug": "Apixaban",
            "icd10_code": "I48.91 or I48.11 — AMBIGUOUS",
            "confidence": 0.88,
            "policy_id": "POL_004",
            "outcome": "QUERY",
            "clarification_needed": "Is this paroxysmal (I48.91) or persistent/longstanding (I48.11)? Duration and Holter results determine correct code.",
            "missing_docs": ["CHA2DS2_VASc_documented"]
        }
    },

    # ── CASE 3: ATTR Amyloidosis + Tafamidis — LOW CONFIDENCE SUSPEND
    {
        "case_id": "CASE_003",
        "challenge_tags": ["LOW_CONFIDENCE"],
        "expected_outcome": "SUSPENDED — Agent 2 confidence 0.78 < 0.90",
        "scenario_name": "ATTR amyloidosis — rare disease, ICD-10 confidence below threshold → workflow suspended",
        "raw_note": """
Pt: Vinod Rao, 72M, MRN: MH-5543210
Insurer: BlueStar, Mem: BS-9876001

Cardiology — Heart Failure Clinic — 22 Jan 2025
Dr Priya Menon, DM Cardiology (Heart Failure specialist)

72M with progressive exertional dyspnoea over 18 months. Echo: LV hypertrophy, 
LVEF 48%, biventricular thickening. Technetium-99m PYP scan grade 3 uptake. 
TTR gene sequencing: wild-type (non-hereditary form). Diagnosis: wild-type ATTR cardiac 
amyloidosis (transthyretin amyloid cardiomyopathy). NYHA class II-III.

Requesting tafamidis (Vyndamax) 61mg OD. Annual authorisation requested.

Supporting docs: PYP scan report, echo, cardiology attestation letter enclosed.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "ATTR Cardiac Amyloidosis",
            "drug": "Tafamidis",
            "icd10_code": "E85.4",
            "confidence": 0.78,
            "policy_id": "POL_005",
            "outcome": "SUSPENDED",
            "suspend_reason": "Agent 2 ICD-10 confidence 0.78 < 0.90 threshold for rare disease. Manual medical coder review required before submission.",
            "missing_docs": []
        }
    },

    # ── CASE 4: RA + Upadacitinib — POLICY CONFLICT (step therapy)
    {
        "case_id": "CASE_004",
        "challenge_tags": ["POLICY_CONFLICT"],
        "expected_outcome": "DENIED — step therapy not satisfied",
        "scenario_name": "RA + upadacitinib — JAK inhibitor requested without prior TNF inhibitor trial",
        "raw_note": """
Patient: Meera Krishnan, 48F, MRN: TN-2219087
BlueStar Health Member: BS-3312900

Rheumatology OPD — 25 Jan 2025
Dr Arun Nair, DM Rheumatology

48-year-old seropositive RA (RF+, Anti-CCP strongly positive). DAS28-CRP score 5.6 
(high disease activity). Currently on methotrexate 15mg weekly + hydroxychloroquine 
200mg BD for 6 months with inadequate response.

Plan: escalate to upadacitinib (Rinvoq) 15mg OD. Patient has NOT received prior TNF 
inhibitor therapy (adalimumab, etanercept, etc.).

Documentation: DAS28 score sheet, methotrexate trial evidence, TB Quantiferon negative.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "Rheumatoid Arthritis",
            "drug": "Upadacitinib",
            "icd10_code": "M05.79",
            "confidence": 0.95,
            "policy_id": "POL_003",
            "outcome": "DENIED",
            "denial_reason": "POL_003 requires prior TNF inhibitor failure before upadacitinib. No prior biologic documented. Step therapy not satisfied.",
            "missing_docs": ["prior_TNF_inhibitor_trial"]
        }
    },

    # ── CASE 5: HFrEF + SGLT2i — CLEAN APPROVAL with step therapy satisfied
    {
        "case_id": "CASE_005",
        "challenge_tags": ["MULTI_AGENT_HANDOFF"],
        "expected_outcome": "APPROVED",
        "scenario_name": "HFrEF + dapagliflozin — all criteria met, straightforward approval",
        "raw_note": """
Pt: Sanjay Patel, 65M, DOB: 12/03/1959, MRN: GJ-8812345
Insurance: BlueStar Health, ID: BS-5543210

Heart Failure Clinic — 28 Jan 2025
Dr Kavita Rao, DM Cardiology

65-year-old male, HFrEF, LVEF 32% on recent echo (report enclosed). HbA1c 8.1% 
(T2DM inadequately controlled — on metformin 1g BD). eGFR 38 mL/min 
(CKD stage 3b). Previously on metformin for 4 months with ongoing poor glycaemic 
control. NT-proBNP 1640 pg/mL.

Requesting dapagliflozin (Farxiga) 10mg OD for: (1) HFrEF (LVEF ≤40%), 
(2) T2DM with CKD — step therapy satisfied (metformin trial documented).

Docs enclosed: echo, eGFR, HbA1c, metformin prescription history.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "DOB", "MRN", "member_id"],
        "expected_entities": {
            "condition": "HFrEF + T2DM with CKD",
            "drug": "Dapagliflozin",
            "icd10_code": "I50.20",
            "confidence": 0.95,
            "policy_id": "POL_001",
            "outcome": "APPROVED",
            "missing_docs": []
        }
    },

    # ── CASE 6: NSCLC + POLICY CONFLICT (wrong drug for biomarker)
    {
        "case_id": "CASE_006",
        "challenge_tags": ["POLICY_CONFLICT", "PHI_LEAK"],
        "expected_outcome": "DENIED — policy conflict: pembrolizumab + EGFR mutation",
        "scenario_name": "NSCLC — pembrolizumab requested despite concurrent EGFR mutation (policy exclusion)",
        "raw_note": """
Patient name: Deepa Iyer, 55 years, Female
Date of Birth: 22/09/1969
Social Security Number: 987-65-4321 [SENSITIVE - mask this]
Phone: 9876543210
MRN: KA-4433221, BlueStar Member: BS-7712300

Oncology — 30 Jan 2025
Dr Ramesh Gupta

55F with stage IV NSCLC (adenocarcinoma). PD-L1 TPS 55% (high). However, NGS panel 
ALSO shows EGFR exon 19 deletion (sensitising mutation). Patient and family insistent 
on pembrolizumab due to availability. Oncologist uncertain about optimal sequencing.

Plan: requesting pembrolizumab (Keytruda) 200mg q3w + carboplatin.

Note: this combination is NOT standard of care with concurrent EGFR sensitising mutation.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "DOB", "SSN", "phone", "MRN", "member_id"],
        "phi_contains_ssn": True,
        "expected_entities": {
            "condition": "NSCLC with EGFR mutation",
            "drug": "Pembrolizumab",
            "icd10_code": "C34.10",
            "confidence": 0.93,
            "policy_id": "POL_002",
            "outcome": "DENIED",
            "denial_reason": "POL_002 explicitly excludes pembrolizumab when concurrent EGFR sensitizing mutation is present. Osimertinib is the preferred first-line agent.",
            "phi_masked": ["SSN", "phone"],
            "missing_docs": []
        }
    },

    # ── CASE 7: HFrEF + Entresto POLICY CONFLICT (concurrent ACEi)
    {
        "case_id": "CASE_007",
        "challenge_tags": ["POLICY_CONFLICT"],
        "expected_outcome": "DENIED — concurrent ACEi contraindication",
        "scenario_name": "HFrEF + sacubitril-valsartan — concurrent ACEi use violates policy exclusion",
        "raw_note": """
Pt: Ravi Menon, 70M, MRN: KL-9981234
BlueStar Member: BS-4421100

Heart Failure Clinic — 1 Feb 2025

70M, HFrEF, LVEF 35%. Currently on ramipril 10mg OD (ACEi), bisoprolol 5mg.
Doctor requesting addition of sacubitril-valsartan (Entresto) 49/51mg BD.

Plan: start Entresto alongside current ramipril.

Note: washout of ramipril NOT documented. No mention of stopping ramipril.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "HFrEF",
            "drug": "Sacubitril-valsartan",
            "icd10_code": "I50.20",
            "confidence": 0.95,
            "policy_id": "POL_007",
            "outcome": "DENIED",
            "denial_reason": "POL_007 EXCLUSION: concomitant ACEi use with sacubitril-valsartan is absolutely contraindicated (angioedema risk). Ramipril must be stopped with 36-hour washout period before Entresto initiation.",
            "missing_docs": ["ACEi_washout_documented"]
        }
    },

    # ── CASE 8: MS + Ocrelizumab — CLEAN APPROVAL
    {
        "case_id": "CASE_008",
        "challenge_tags": ["MULTI_AGENT_HANDOFF"],
        "expected_outcome": "APPROVED",
        "scenario_name": "RRMS + ocrelizumab — all criteria met",
        "raw_note": """
Patient: Sunita Rao, 38F, MRN: MH-3312456
BlueStar Member: BS-8877001

Neurology OPD — 3 Feb 2025
Dr Vikram Desai, DM Neurology

38F, relapsing-remitting MS (McDonald 2017 criteria, MRI confirmed). 2 relapses in 
prior 12 months. New Gd+ lesion on recent MRI brain. EDSS 2.5. 
No prior natalizumab. JC antibody positive.

Requesting ocrelizumab (Ocrevus) 300mg IV q2 weeks × 2, then 600mg q6 months.
Hepatitis B surface antigen: negative. Anti-HBc: negative. JC antibody: positive 
(noted — different mechanism than natalizumab, JC risk not applicable).

Docs: MRI report, neurologist diagnosis letter, Hep B serology.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "Relapsing-Remitting Multiple Sclerosis",
            "drug": "Ocrelizumab",
            "icd10_code": "G35",
            "confidence": 0.96,
            "policy_id": "POL_006",
            "outcome": "APPROVED",
            "missing_docs": []
        }
    },

    # ── CASE 9: MISSPELLED condition — LOW CONFIDENCE SUSPEND
    {
        "case_id": "CASE_009",
        "challenge_tags": ["LOW_CONFIDENCE", "FORM_SCHEMA_FAIL"],
        "expected_outcome": "SUSPENDED — misspelled condition, confidence 0.42",
        "scenario_name": "Misspelled diagnosis — Agent 2 cannot map with confidence ≥0.90",
        "raw_note": """
Pt: Anitha Sharma, 45F, MRN: DL-2287654
BlueStar Member: BS-1190034

GP Referral — 5 Feb 2025

45F with chronic widespread pain, fatigue, sleep disturbance. Diagnosed with 
"fibromylagia" (note: possible spelling variation). On pregabalin 75mg BD 
currently. Requesting prior auth for duloxetine 60mg OD.

DAS28 not applicable. Rheumatology review pending.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "fibromylagia (misspelled)",
            "drug": "Duloxetine",
            "icd10_code": "UNKNOWN — confidence 0.42",
            "confidence": 0.42,
            "policy_id": None,
            "outcome": "SUSPENDED",
            "suspend_reason": "Agent 2: Cannot map 'fibromylagia' to ICD-10 code with confidence ≥0.90. Closest match: M79.7 (fibromyalgia) at 0.42 confidence. SUSPEND — manual medical coder review required.",
            "missing_docs": ["correct_diagnosis_documentation"]
        }
    },

    # ── CASE 10: T2DM + Semaglutide — MISSING DOCUMENTATION
    {
        "case_id": "CASE_010",
        "challenge_tags": ["FORM_SCHEMA_FAIL"],
        "expected_outcome": "PEND — missing required documentation",
        "scenario_name": "T2DM + semaglutide — approval pending missing HbA1c documentation",
        "raw_note": """
Pt: Kiran Joshi, 52M, MRN: MH-7765432
BlueStar Health, Mem: BS-2230099

GP OPD — 7 Feb 2025

52M, T2DM on metformin 500mg BD × 2 years. Not well controlled clinically (patient 
reports polydipsia, polyuria). HbA1c — not documented in this note (last result 
reportedly 8.9% some months ago, exact date unknown).

Request: semaglutide (Ozempic) 0.5mg weekly SC.

No recent HbA1c report provided with this submission.
        """,
        "phi_present": True,
        "phi_fields": ["patient_name", "MRN", "member_id"],
        "expected_entities": {
            "condition": "Type 2 Diabetes Mellitus",
            "drug": "Semaglutide",
            "icd10_code": "E11.9",
            "confidence": 0.97,
            "policy_id": "POL_008",
            "outcome": "PEND",
            "pend_reason": "HbA1c result within 90 days is required per POL_008. Submitted note references historical result with unspecified date. PA held pending submission of recent HbA1c report.",
            "missing_docs": ["HbA1c_90days"]
        }
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PA FORM SCHEMA (output of Agent 4)
# ─────────────────────────────────────────────────────────────────────────────

PA_FORM_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PriorAuthorizationRequest",
    "type": "object",
    "required": [
        "request_id", "timestamp", "patient", "clinical",
        "coding", "policy_check", "workflow_status", "agent_trace"
    ],
    "properties": {
        "request_id":  {"type": "string", "pattern": "^PA-[0-9]{8}-[A-Z0-9]{6}$"},
        "timestamp":   {"type": "string", "format": "date-time"},
        "patient": {
            "type": "object",
            "required": ["member_id", "insurer", "phi_masked"],
            "properties": {
                "member_id":  {"type": "string"},
                "insurer":    {"type": "string"},
                "phi_masked": {"type": "boolean"},
                "age_band":   {"type": "string", "enum": ["18-30","31-45","46-60","61-75","75+"]}
            }
        },
        "clinical": {
            "type": "object",
            "required": ["condition", "drug_requested", "physician_npi"],
            "properties": {
                "condition":       {"type": "string"},
                "drug_requested":  {"type": "string"},
                "drug_tier":       {"type": "integer", "minimum": 1, "maximum": 5},
                "physician_npi":   {"type": "string"},
                "supporting_docs": {"type": "array", "items": {"type": "string"}}
            }
        },
        "coding": {
            "type": "object",
            "required": ["icd10_code", "icd10_description", "confidence_score", "coding_agent"],
            "properties": {
                "icd10_code":        {"type": "string"},
                "icd10_description": {"type": "string"},
                "confidence_score":  {"type": "number", "minimum": 0, "maximum": 1},
                "coding_agent":      {"type": "string"},
                "ambiguous_codes":   {"type": "array"}
            }
        },
        "policy_check": {
            "type": "object",
            "required": ["policy_id", "step_therapy_satisfied", "missing_documentation"],
            "properties": {
                "policy_id":               {"type": "string"},
                "step_therapy_satisfied":  {"type": "boolean"},
                "missing_documentation":   {"type": "array", "items": {"type": "string"}},
                "policy_conflicts":        {"type": "array"},
                "rag_chunks_used":         {"type": "integer"}
            }
        },
        "workflow_status": {
            "type": "object",
            "required": ["decision", "reason"],
            "properties": {
                "decision": {"type": "string",
                             "enum": ["APPROVED", "DENIED", "PEND", "SUSPENDED", "QUERY"]},
                "reason":   {"type": "string"},
                "suspended_reason": {"type": "string"}
            }
        },
        "agent_trace": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["agent", "timestamp", "action", "latency_ms"],
                "properties": {
                    "agent":      {"type": "string"},
                    "timestamp":  {"type": "string"},
                    "action":     {"type": "string"},
                    "latency_ms": {"type": "number"},
                    "output_keys": {"type": "array"}
                }
            }
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# WRITE ALL FILES
# ─────────────────────────────────────────────────────────────────────────────

def write_all():
    datasets = {
        "icd10_knowledge_graph.json": ICD10_KG,
        "drugs.json": DRUGS,
        "policy_documents.json": POLICY_DOCS,
        "patient_notes.json": PATIENT_NOTES,
        "pa_form_schema.json": PA_FORM_SCHEMA,
    }
    for fname, data in datasets.items():
        with open(OUT / fname, "w") as f:
            json.dump(data, f, indent=2)
        n = len(data) if isinstance(data, list) else "schema"
        print(f"  ✓ {fname} — {n}")

    print(f"\n{'='*55}")
    print("DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  ICD-10 KG nodes          : {len(ICD10_KG)}")
    print(f"  Drug formulary entries   : {len(DRUGS)}")
    print(f"  Policy documents         : {len(POLICY_DOCS)}")
    print(f"  Patient notes / PA cases : {len(PATIENT_NOTES)}")
    print()
    print("  Injected challenge distribution:")
    from collections import Counter
    tags = []
    for n in PATIENT_NOTES:
        tags.extend(n.get("challenge_tags", []))
    for t, c in Counter(tags).most_common():
        print(f"    [{t}]: {c} cases")

if __name__ == "__main__":
    write_all()
