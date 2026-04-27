# PriorAuth — Autonomous Multi-Agent Prior Authorization
### A2A × Agentic ICD-10 KG × Agentic CPT KG × Agentic RAG × HITL Router × Azure AI Foundry
**CitiusTech Gen AI & Agentic AI Training — Project 4**

---

## The Problem a Single LLM Cannot Reliably Solve

A monolithic LLM asked to process a prior authorization request:

```
"Patient: Deepa Iyer, DOB: 22/09/1969, SSN: 987-65-4321
 Diagnosis: NSCLC with EGFR del19 + PD-L1 55%
 Drug: Pembrolizumab 200mg q3w. Please approve."
```

will likely:
1. **Approve** — without checking BlueStar policy explicitly excludes pembrolizumab when EGFR mutation is present
2. **Send the SSN to the LLM context window** — HIPAA violation
3. **Return a free-text paragraph** — not a structured JSON payload the claims system can ingest
4. **Assign the wrong ICD-10 code** — NSCLC has 4 valid codes; wrong code means wrong policy branch
5. **Assign the wrong CPT code** — pemrolizumab is IV infusion (96413), not an office visit (99214); wrong CPT = claims audit risk
6. **Never flag for human review** — no confidence gates, no HITL routing, no oversight layer

The multi-agent system catches all six failures through specialisation, gating, and explicit routing.

---

## Agent Architecture (5 Agents)

```
                    ┌────────────────────────────────────────────────┐
                    │           A2A Message Bus (Observable)          │
                    │   All inter-agent messages logged + timestamped │
                    └────────────────────────────────────────────────┘
                                    │
Raw doctor note ──► [PHI GUARDRAIL] │
                    mask before LLM ▼
                     ┌─────────────────────────┐
                     │   Agent 1 — Extractor    │   ~85ms
                     │  Reads masked note        │
                     │  Extracts: condition,     │
                     │  drug, NPI, docs, facts   │
                     └────────────┬────────────┘
                                  │ entities payload
                                  ▼
                     ┌─────────────────────────┐
                     │  Agent 2 — ICD Coder     │   ~95ms
                     │  KG lookup: condition     │
                     │  → ICD-10 code            │
                     │  Confidence gate: <0.90   │
                     │  → SUSPEND (early exit)   │
                     └────────────┬────────────┘
                                  │ coded payload
                                  ▼
                     ┌─────────────────────────┐
                     │ Agent 2.5 — CPT Predictor│   ~55ms  ← NEW
                     │  KG lookup: condition +  │
                     │  drug + inferred route   │
                     │  → CPT procedure code    │
                     │  Confidence gate: <0.85  │
                     │  → CPT SUSPENDED (HITL)  │
                     └────────────┬────────────┘
                                  │ cpt payload
                                  ▼
                     ┌─────────────────────────┐
                     │  Agent 3 — Policy RAG    │   ~180ms
                     │  Retrieves policy docs    │
                     │  Checks: step therapy,    │
                     │  exclusions, missing docs │
                     │  → APPROVED/DENIED/PEND  │
                     └────────────┬────────────┘
                                  │ policy check
                                  ▼
                     ┌─────────────────────────┐
                     │  Agent 4 — Form Filler   │   ~65ms
                     │  HITLRouter (8 triggers)  │
                     │  Generates PA JSON        │
                     │  Schema validation        │
                     │  → Final PA payload       │
                     └─────────────────────────┘
```

**E2E Latency**: ~480ms (5-agent A2A) vs ~2,200ms (monolithic) — ~4.6× faster

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install jsonschema langchain-openai langchain-core numpy pandas matplotlib
```

### Step 2: Configure LLM (choose one)

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"
# or
export OPENAI_API_KEY="sk-..."
# or leave blank — MockLLM activates automatically
```

### Step 3: Generate CPT Synthetic Data

```bash
cd data/
python generate_cpt_dataset.py   # injects expected_cpt + expected_hitl into patient_notes.json
```

### Step 4: Run a Single Case

```bash
cd src/
python agents.py 1    # NSCLC + Osimertinib → APPROVED, CPT 99214
python agents.py 3    # ATTR → dual ICD-10+CPT SUSPENDED, HIGH priority HITL
python agents.py 8    # RRMS + Ocrelizumab → APPROVED, CPT 96365 (IV infusion)
```

### Step 5: Run the Full Demo

```bash
cd demo/
python demo.py                    # All 4 scenarios + 3 limitations
python demo.py --scenario 1       # Single scenario
python demo.py --limitations      # Limitations only
```

### Step 6: Evaluation + Charts

```bash
cd evaluation/
python eval_dashboard.py          # Metrics table + 4 charts
```

---

## Project Structure

```
priorauth-a2a/
├── data/
│   ├── generate_dataset.py          ← Generate ICD-10/drug/policy base data
│   ├── generate_cpt_dataset.py      ← Generate CPT synthetic data + inject into notes  ← NEW
│   ├── icd10_knowledge_graph.json   ← 16 ICD-10 nodes with aliases + confidence
│   ├── cpt_procedures.json          ← 20 CPT codes with routes + related drugs         ← NEW
│   ├── drugs.json                   ← 15 drugs with tiers + PA requirements
│   ├── policy_documents.json        ← 8 insurer policy docs (Agent 3 RAG corpus)
│   ├── patient_notes.json           ← 10 PA cases + expected_cpt + expected_hitl       ← UPDATED
│   └── pa_form_schema.json          ← JSON Schema (+ cpt_coding + hitl_routing)        ← UPDATED
│
├── src/
│   └── agents.py                    ← All 5 agents + HITLRouter + orchestrator         ← UPDATED
│
├── evaluation/
│   ├── eval_dashboard.py            ← Metrics + 4 charts (+ CPT + HITL)               ← UPDATED
│   ├── 01_multiagent_vs_monolithic.png   (5 metrics now)
│   ├── 02_decision_breakdown_latency.png
│   ├── 03_agent_waterfall.png            (5-agent waterfall)
│   └── 04_cpt_hitl_analysis.png          ← NEW
│
├── demo/
│   └── demo.py                      ← 4 scenarios + 3 limitations (CPT + HITL shown)  ← UPDATED
│
├── configs/
│   ├── azure_foundry_setup.md
│   ├── autogen_alternative.md
│   └── phi_patterns.md
│
└── README.md
```

---

## The Five Agents (Detailed)

### Agent 1 — Extractor

**Responsibility:** Read the doctor's note, mask all PHI, extract structured entities.

**PHI Guardrail (runs BEFORE any LLM call):**
```python
masked_note, redacted_types = mask_phi(raw_note)
# Patterns: SSN, DOB, phone, patient name, MRN, member ID, email
```

### Agent 2 — ICD-10 Coder (Agentic KG)

**Responsibility:** Map extracted condition to ICD-10 code via knowledge graph.

```python
CONFIDENCE_THRESHOLD = 0.90
if confidence < CONFIDENCE_THRESHOLD:
    # SUSPEND — early exit, propagated through Agent 3/4
    # Triggers: icd10_low_confidence HITL trigger
```

**Ambiguity detection:** Two candidates within 8% → `is_ambiguous=True` → `QUERY` decision

### Agent 2.5 — CPT Predictor (Agentic KG) ← NEW

**Responsibility:** Map condition + drug + inferred route to CPT procedure code.

**Why CPT matters for PA:**
- IV drugs (pembrolizumab, ocrelizumab) require infusion administration codes (96413, 96365)
- SC drugs (semaglutide) require injection codes (96372)
- Oral drugs require E/M office visit codes (99213–99215)
- Wrong CPT → wrong claims tier, potential audit, clawback liability

**Route inference:**
```python
DRUG_ROUTE_MAP = {
    "pembrolizumab": "intravenous",   # → CPT 96413 (chemo infusion)
    "ocrelizumab":   "intravenous",   # → CPT 96365 (therapeutic infusion)
    "semaglutide":   "subcutaneous",  # → CPT 96372 (SC injection)
    "osimertinib":   "oral",          # → CPT 99214 (E/M moderate)
    # ...
}
```

**Scoring (composite):**
```python
if icd10_match and drug_match:  score = 0.93  # strongest
elif icd10_match and route_match: score = 0.85
elif icd10_match only:          score = 0.70
elif drug_match only:           score = 0.65
```

**Confidence gate:** `CPT_CONFIDENCE_THRESHOLD = 0.85`
- CASE_003 (ATTR, rare): confidence 0.837 → CPT SUSPENDED → CPT coder queue
- CASE_009 (misspelled): confidence 0.579 → CPT SUSPENDED → CPT coder queue
- CPT suspension does NOT halt the PA workflow (unlike ICD-10) — it sets a HITL flag

**Key differentiator from ICD-10 coder:** CPT suspension allows the PA decision to proceed while routing the CPT coding task separately. This avoids blocking the clinical decision on a billing code issue.

### Agent 3 — Policy RAG (Agentic RAG)

**Responsibility:** Retrieve and check insurer policy against the coded claim.

Checks: approved indication, step therapy, required documentation, exclusions.
Propagates ICD-10 SUSPENDED immediately (skips LLM call, saves ~180ms).

### Agent 4 — Form Filler + HITL Router + Schema Validator

**Responsibility:** HITL routing → PA JSON generation → schema validation.

---

## Human-in-the-Loop (HITL) Framework ← NEW

The `HITLRouter` evaluates **8 triggers** after policy check and routes to one of 4 review queues.

### HITL Triggers

| Trigger | Priority | Queue | SLA |
|---------|----------|-------|-----|
| `icd10_low_confidence` | HIGH | MEDICAL_CODER_QUEUE | 2h |
| `cpt_low_confidence` | HIGH | CPT_CODER_QUEUE | 2h |
| `icd10_ambiguous` | MEDIUM | MEDICAL_CODER_QUEUE | 8h |
| `cpt_ambiguous` | MEDIUM | CPT_CODER_QUEUE | 8h |
| `high_cost_drug` (≥$5,000/mo) | MEDIUM | CLINICAL_MANAGER_QUEUE | 8h |
| `policy_conflict` | MEDIUM | COMPLIANCE_REVIEW_QUEUE | 8h |
| `step_therapy_violation` | MEDIUM | CLINICAL_REVIEW_QUEUE | 8h |
| `missing_critical_docs` | LOW | CLINICAL_REVIEW_QUEUE | 24h |

### HITL Routing per Case

| Case | Requires HITL | Priority | Triggers |
|------|--------------|----------|----------|
| CASE_001 | ✓ MEDIUM | high_cost_drug | Osimertinib $18.5k/month |
| CASE_002 | ✓ MEDIUM | icd10_ambiguous, cpt_ambiguous | AF type unclear |
| CASE_003 | ✓ **HIGH** | icd10_low_conf, cpt_low_conf, high_cost | Dual suspension, $22k/month |
| CASE_004 | ✓ MEDIUM | step_therapy_violation | TNF inhibitor step not met |
| CASE_005 | — none | fully automated | All criteria met, $580/month |
| CASE_006 | ✓ MEDIUM | policy_conflict, high_cost_drug | EGFR exclusion |
| CASE_007 | ✓ MEDIUM | policy_conflict | ACEi contraindication |
| CASE_008 | ✓ MEDIUM | high_cost_drug | Ocrelizumab $20k/month |
| CASE_009 | ✓ **HIGH** | icd10_low_conf, cpt_low_conf | Dual suspension, misspelling |
| CASE_010 | ✓ LOW | missing_critical_docs | HbA1c not provided |

**Key finding:** Monolithic LLM has **zero HITL** — no oversight regardless of risk. Multi-agent routes **9/10 cases** to human review with appropriate priority and SLA.

---

## CPT Code Prediction ← NEW

### CPT Knowledge Graph (20 codes)

Organized by drug administration route and specialty:

| Category | CPT Codes | Used For |
|----------|-----------|----------|
| Evaluation & Management | 99213, 99214, 99215 | Oral drug management (low/mod/high complexity) |
| Oncology IV Infusion | 96413, 96415 | Chemo/immunotherapy IV (initial + additional hours) |
| Therapeutic IV Infusion | 96365, 96366 | Biologic IV — MS, RA (initial + additional hours) |
| SC/IM Injection | 96372 | GLP-1, adalimumab, etanercept SC |
| Oncology Diagnostics | 81455 | NGS genomic panel |
| Rheumatology | 20610 | Joint aspiration/injection |
| Consultation | 99242 | New condition consult (pain, rare disease) |

### CPT Ground Truth (10 Cases)

| Case | Expected CPT | Confidence | Route | Notes |
|------|-------------|------------|-------|-------|
| CASE_001 | 99214 | 0.856 | oral | Oncology E/M moderate |
| CASE_002 | 99213 | 0.865 | oral | Cardiology follow-up |
| CASE_003 | **SUSPENDED** | 0.837 | oral | Rare disease, below 0.85 gate |
| CASE_004 | 99214 | 0.856 | oral | RA E/M moderate |
| CASE_005 | 99214 | 0.856 | oral | HFrEF E/M moderate |
| CASE_006 | 96413 | 0.884 | IV | Chemo infusion (pembrolizumab) |
| CASE_007 | 99213 | 0.865 | oral | HFrEF follow-up |
| CASE_008 | 96365 | 0.865 | IV | Biologic infusion (ocrelizumab) |
| CASE_009 | **SUSPENDED** | 0.579 | oral | Misspelled diagnosis, below gate |
| CASE_010 | 96372 | 0.874 | SC | GLP-1 injection (semaglutide) |

---

## Injected Challenge Patterns

| Challenge | Cases | What It Exposes |
|-----------|-------|-----------------|
| `[AMBIGUOUS_CODE]` | CASE_002 | AFib maps to I48.91 vs I48.11; CPT level also ambiguous |
| `[LOW_CONFIDENCE]` | CASE_003, CASE_009 | Dual ICD-10+CPT suspension → HIGH priority HITL |
| `[PHI_LEAK]` | CASE_006 | SSN in note — must be masked before LLM call (HIPAA) |
| `[POLICY_CONFLICT]` | CASE_004, CASE_006, CASE_007 | Step therapy, EGFR exclusion, concurrent ACEi |
| `[MULTI_AGENT_HANDOFF]` | CASE_001, CASE_005, CASE_008 | Full A2A trace with CPT prediction |
| `[FORM_SCHEMA_FAIL]` | CASE_009, CASE_010 | Schema validation catches missing/invalid fields |

---

## Evaluation Results

| Metric | Monolithic LLM | Multi-Agent A2A |
|--------|----------------|-----------------|
| **Decision Accuracy** | 70% (7/10) | **100% (10/10)** |
| **CPT Code Accuracy** | 10% (1/10) | **100% (10/10)** |
| **PHI Masking Rate** | 0% | **100%** |
| **Schema Validity** | 80% | **100%** |
| **Avg E2E Latency** | ~2,200ms | **~480ms** |
| **ICD-10 Conf Gate** | 0/10 triggered | **2/10 triggered** |
| **CPT Conf Gate** | 0/10 triggered | **2/10 triggered** |
| **HITL Triggered** | 0/10 (blind) | **9/10 (correct routing)** |
| **HITL Accuracy** | 0% | **100%** |

**4 charts generated by `eval_dashboard.py`:**
1. `01_multiagent_vs_monolithic.png` — 5-metric side-by-side comparison
2. `02_decision_breakdown_latency.png` — decision counts + per-case latency
3. `03_agent_waterfall.png` — 5-agent execution waterfall
4. `04_cpt_hitl_analysis.png` — CPT accuracy per case + HITL priority distribution

**Critical findings:**
- Monolithic LLM approves ALL 10 cases AND assigns wrong CPT codes in 9/10 cases
- Multi-agent catches 3 wrong approvals AND flags 2 cases for CPT coder review
- Zero HITL in monolithic means no oversight on $22,000/month tafamidis, no CPT coding review, and no compliance escalation for policy violations

---

## A2A Message Bus

Every inter-agent handoff is logged:

```
[10:32:15] Agent1_Extractor        → Agent2_ICDCoder         |   85ms | keys=[condition, drug, ...]
[10:32:15] Agent2_ICDCoder         → Agent2_5_CPTPredictor   |   95ms | keys=[icd10_code, confidence, ...]
[10:32:15] Agent2_5_CPTPredictor   → Agent3_PolicyRAG        |   55ms | keys=[primary_cpt_code, cpt_confidence, ...]
[10:32:15] Agent3_PolicyRAG        → Agent4_FormFiller       |  180ms | keys=[policy_id, decision, ...]
[10:32:16] Agent4_FormFiller       → OUTPUT                  |   65ms | keys=[request_id, cpt_coding, hitl_routing, ...]
```

---

## PA Case Summary

| Case | Condition | Drug | Decision | CPT | HITL Priority |
|------|-----------|------|----------|-----|---------------|
| CASE_001 | NSCLC + EGFR del19 | Osimertinib | ✅ APPROVED | 99214 | MEDIUM |
| CASE_002 | Atrial Fibrillation | Apixaban | ❓ QUERY | 99213 | MEDIUM |
| CASE_003 | ATTR Amyloidosis | Tafamidis | ⛔ SUSPENDED | ⛔ SUSPENDED | **HIGH** |
| CASE_004 | RA (no TNF) | Upadacitinib | ❌ DENIED | 99214 | MEDIUM |
| CASE_005 | HFrEF + T2DM-CKD | Dapagliflozin | ✅ APPROVED | 99214 | — none |
| CASE_006 | NSCLC + EGFR + PD-L1 | Pembrolizumab | ❌ DENIED | 96413 | MEDIUM |
| CASE_007 | HFrEF + ACEi | Sacubitril-val | ❌ DENIED | 99213 | MEDIUM |
| CASE_008 | RRMS | Ocrelizumab | ✅ APPROVED | 96365 | MEDIUM |
| CASE_009 | Misspelled fibromylagia | Duloxetine | ⛔ SUSPENDED | ⛔ SUSPENDED | **HIGH** |
| CASE_010 | T2DM (missing HbA1c) | Semaglutide | ⏳ PEND | 96372 | LOW |

---

## Azure AI Foundry Deployment

See `configs/azure_foundry_setup.md`. Each agent deploys as a dedicated Azure Foundry Agent:

```python
cpt_predictor_agent = client.agents.create_agent(
    model="gpt-4o",
    name="PA_CPTPredictor",
    instructions=CPT_PREDICTION_SYSTEM_PROMPT,
    toolset=ToolSet([cpt_kg_lookup_tool, route_inference_tool, confidence_gate_tool])
)

hitl_agent = client.agents.create_agent(
    model="gpt-4o",
    name="PA_HITLRouter",
    instructions=HITL_ROUTING_SYSTEM_PROMPT,
    toolset=ToolSet([cost_lookup_tool, trigger_evaluator_tool, queue_router_tool])
)
```

---

## AutoGen Alternative

```python
import autogen

extractor     = autogen.AssistantAgent("Extractor",     system_message=EXTRACTION_PROMPT)
icd_coder     = autogen.AssistantAgent("ICDCoder",      system_message=ICD_PROMPT)
cpt_predictor = autogen.AssistantAgent("CPTPredictor",  system_message=CPT_PROMPT)   # NEW
policy_rag    = autogen.AssistantAgent("PolicyRAG",     system_message=POLICY_PROMPT)
form_filler   = autogen.AssistantAgent("FormFiller",    system_message=FORM_PROMPT)
hitl_router   = autogen.AssistantAgent("HITLRouter",    system_message=HITL_PROMPT)  # NEW

group_chat = autogen.GroupChat(
    agents=[orchestrator, extractor, icd_coder, cpt_predictor, policy_rag, form_filler, hitl_router],
    speaker_selection_method="round_robin",
    max_round=6
)
```

---

*CitiusTech Gen AI & Agentic AI Training Program — Project 4 of 5*
