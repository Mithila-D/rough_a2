import sys, os, json, time, re, hashlib
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import threading

st.set_page_config(page_title="Pre-Auth AI System", page_icon="🏥", layout="wide")

st.markdown("""
<style>
body { background: #0f1117; }
.agent-card { border:1px solid #2d3748; border-radius:10px; padding:14px 18px; margin-bottom:10px; background:#1a202c; }
.agent-header { font-size:15px; font-weight:700; margin-bottom:4px; }
.agent-running { border-left:4px solid #ecc94b; }
.agent-done    { border-left:4px solid #48bb78; }
.agent-error   { border-left:4px solid #fc8181; }
.agent-pending { border-left:4px solid #4a5568; opacity:0.45; }
.io-label { font-size:10px; color:#a0aec0; text-transform:uppercase; letter-spacing:1px; margin-top:10px; margin-bottom:3px; }
.next-label { font-size:11px; color:#63b3ed; margin-top:6px; }
pre { font-size:11px !important; background:#2d3748 !important; padding:8px !important;
      border-radius:5px !important; overflow-x:auto !important; max-height:160px !important;
      white-space:pre-wrap !important; word-break:break-word !important; color:#ffffff !important; }
pre.out { background:#1e3a5f !important; color:#ffffff !important; }
.decision-approved { background:linear-gradient(135deg,#1a4731,#22543d); border:2px solid #48bb78; border-radius:14px; padding:22px 28px; margin:10px 0; }
.decision-denied   { background:linear-gradient(135deg,#4a1a1a,#742a2a); border:2px solid #fc8181; border-radius:14px; padding:22px 28px; margin:10px 0; }
.decision-human    { background:linear-gradient(135deg,#2d2a1a,#534a1a); border:2px solid #f6ad55; border-radius:14px; padding:22px 28px; margin:10px 0; }
.decision-title { font-size:30px; font-weight:800; letter-spacing:1px; margin-bottom:8px; }
.decision-reason { font-size:14px; color:#e2e8f0; line-height:1.7; margin-top:8px; }
.conf-row { display:flex; align-items:center; margin:7px 0; gap:12px; }
.conf-label { font-size:13px; color:#cbd5e0; min-width:240px; }
.conf-bar-bg { flex:1; background:#2d3748; border-radius:99px; height:12px; }
.conf-bar-fill { height:12px; border-radius:99px; transition:width 0.4s; }
.conf-val { font-size:13px; font-weight:700; min-width:44px; text-align:right; }
.hitl-badge { display:inline-block; background:#744210; color:#fefcbf; font-size:13px; font-weight:700;
              border-radius:6px; padding:3px 10px; margin-left:12px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("data/patient_notes.json") as f:      notes    = json.load(f)
    with open("data/icd10_knowledge_graph.json") as f: icd_db = json.load(f)
    with open("data/cpt_procedures.json") as f:     cpt_db   = json.load(f)
    with open("data/drugs.json") as f:               drugs_db = json.load(f)
    with open("data/policy_documents.json") as f:   policies = json.load(f)
    return notes, icd_db, cpt_db, drugs_db, policies

notes, icd_db, cpt_db, drugs_db, policies = load_data()
case_map = {n["case_id"]: n for n in notes}
case_ids = [n["case_id"] for n in notes]

# ── Lookup indexes ────────────────────────────────────────────────────────────
icd_by_code  = {item["code"]: item for item in icd_db}
cpt_by_code  = {c["code"]: c for c in cpt_db}
drug_by_id   = {d["id"]: d for d in drugs_db}
drug_by_name = {d["name"].lower(): d["id"] for d in drugs_db}
drug_by_brand= {d.get("brand","").lower(): d["id"] for d in drugs_db if d.get("brand")}

AGENTS = [
    ("phi",      "🧹 PHI Sanitizer",      "Regex patterns + Microsoft Presidio NER (spaCy) — dual-layer PHI de-identification"),
    ("entities", "🧬 Entity Extractor",   "Extracts conditions, drugs, visit type via LLM"),
    ("icd",      "🏷️ ICD Inference",      "Maps clinical text → ICD-10 code"),
    ("cpt",      "💊 CPT Inference",      "Maps clinical text → CPT procedure code"),
    ("policy",   "📜 Policy Evaluator",   "Checks ICD + drug coverage against payer policy"),
    ("critique", "🔍 Critique Agent",     "Flags low-confidence, missing codes, HITL triggers"),
    ("decision", "⚖️ Decision Assembler", "Assembles final authorization decision"),
    ("audit",    "📋 Audit Logger",       "Logs all steps to audit trail"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def snap(state):     return json.loads(state.model_dump_json())
def diff(b, a):      return {k: v for k, v in a.items() if b.get(k) != v}

def agent_input(key, s):
    return {
        "phi":      {"case_id": s.get("case_id"), "raw_text_chars": len(s.get("raw_text",""))},
        "entities": {"sanitized_text_preview": (s.get("sanitized_text") or "")[:200]},
        "icd":      {"entities": s.get("entities")},
        "cpt":      {"entities": s.get("entities"), "icd": s.get("icd")},
        "policy":   {"icd": s.get("icd"), "cpt": s.get("cpt"), "entities": s.get("entities")},
        "critique": {"icd": s.get("icd"), "cpt": s.get("cpt"), "policy": s.get("policy")},
        "decision": {"icd": s.get("icd"), "cpt": s.get("cpt"), "policy": s.get("policy"), "critique": s.get("critique")},
        "audit":    {"steps_in_log": len(s.get("audit_log", []))},
    }.get(key, {})

def phi_diff_html(san):
    return re.sub(r'(\[[A-Z_]+\])',
                  r'<span style="background:#742a2a;color:#fed7d7;border-radius:3px;padding:0 3px">\1</span>',
                  san)

def render(ph, key, label, desc, status="pending", inp=None, out=None,
           elapsed=None, err=None, phi_san=None):
    icon = {"pending":"⏳","running":"🔄","done":"✅","error":"❌"}[status]
    ts   = f" <span style='color:#a0aec0;font-size:11px'>{elapsed:.2f}s</span>" if elapsed else ""
    html = (f'<div class="agent-card agent-{status}">'
            f'<div class="agent-header">{icon} {label}{ts}</div>'
            f'<div style="font-size:12px;color:#718096;margin-bottom:4px">{desc}</div>')
    # NOTE: hide raw agent inputs in the UI to avoid noisy technical details
    # Inputs are still recorded in `results` for audit but not rendered here.
    if phi_san is not None:
        html += f'<div class="io-label">↑ PHI Masking — redacted tokens highlighted</div><pre class="out">{phi_diff_html(phi_san)}</pre>'
    elif out is not None:
        html += f'<div class="io-label">↑ Output — changed state fields</div><pre class="out">{json.dumps(out, indent=2, default=str)}</pre>'
    if err:
        html += f'<div style="color:#fc8181;font-size:12px;margin-top:6px">⚠️ {err[:500]}</div>'
    if status == "done" and key != "audit":
        idx = next((i for i,(k,_,_) in enumerate(AGENTS) if k==key), None)
        if idx is not None and idx+1 < len(AGENTS):
            html += f'<div class="next-label">⟶ passes state to: <b>{AGENTS[idx+1][1]}</b></div>'
    ph.markdown(html + "</div>", unsafe_allow_html=True)

def conf_bar_html(label, value, color=None):
    pct = int(round(value * 100))
    if color is None:
        color = "#48bb78" if pct >= 80 else ("#f6ad55" if pct >= 60 else "#fc8181")
    return (f'<div class="conf-row">'
            f'<span class="conf-label">{label}</span>'
            f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="conf-val" style="color:{color}">{pct}%</span>'
            f'</div>')

# ── HITL + Validation Logic ───────────────────────────────────────────────────
def find_drug(name):
    k = name.lower().strip()
    did = drug_by_name.get(k) or drug_by_brand.get(k)
    if did:
        return drug_by_id[did]
    for n2, did2 in drug_by_name.items():
        if k in n2 or n2 in k:
            return drug_by_id[did2]
    for b2, did2 in drug_by_brand.items():
        if b2 and (k in b2 or b2 in k):
            return drug_by_id[did2]
    return None

def find_policy_for_drug(drug_id):
    for pol in policies:
        if drug_id in pol.get("drug_ids", []):
            return pol
    return None

# ── FIX: Background drug filter — shared with entity_extractor ────────────────
# These are common maintenance/background medications that are NOT auth targets
# unless they appear explicitly in a request-context phrase.
_BACKGROUND_DRUGS = [
    "bisoprolol", "warfarin", "ramipril", "lisinopril", "enalapril",
    "metoprolol", "atenolol", "amlodipine", "furosemide", "spironolactone",
    "aspirin", "clopidogrel", "atorvastatin", "rosuvastatin",
    "omeprazole", "pantoprazole", "levothyroxine",
    "hydroxychloroquine", "methotrexate", "sulfasalazine",
    "pregabalin", "gabapentin", "duloxetine",
]

# Phrases that signal a drug is being REQUESTED for authorization
_REQUEST_PHRASES = [
    r"requesting\s+(\w[\w\s\-]+?)(?:\s+\d|\s+for|\s+\(|,|\.|$)",
    r"request(?:ing|ed)?\s+(?:prior\s+auth(?:orization)?\s+for\s+)?(\w[\w\s\-]+?)(?:\s+\d|\s+for|\s+\(|,|\.|$)",
    r"plan[:\s]+(?:start|initiate|add|begin|escalate\s+to|commence)\s+(\w[\w\s\-]+?)(?:\s+\d|\s+for|\s+\(|,|\.|$)",
    r"(?:start|initiate|add|escalate\s+to)\s+(\w[\w\s\-]+?)(?:\s+\d|\s+for|\s+\(|,|\.|$)",
]

def _hitl_is_background_drug(drug_name: str, sanitized_text: str) -> bool:
    """
    Returns True if drug_name is a known background/maintenance medication
    AND does NOT appear in a request-context phrase in the note.
    Such drugs should be skipped in HITL checks — they are not the auth target.
    """
    dl = drug_name.lower().strip()
    if dl not in {d.lower() for d in _BACKGROUND_DRUGS}:
        return False  # Not a background drug — apply normal HITL rules
    tl = sanitized_text.lower()
    for pat in _REQUEST_PHRASES:
        for m in re.finditer(pat, tl):
            phrase = m.group(1).strip().lower()
            if dl in phrase or phrase in dl:
                return False  # Found in a request phrase — it IS being requested
    return True  # Background drug not in any request phrase — skip HITL

def run_hitl_checks(icd_result, cpt_result, entities, sanitized_text=""):
    """
    Rules:
      1. CPT inferred but not in CPT database → HITL  (suggest alt lookup, never match ICD→CPT)
      2. Drug extracted but not in drug DB → HITL
      3. Drug in DB but no policy covers it → HITL
      4. Drug has policy but ICD not in approved list → HITL (off-label)

      FIX: Rules 2-4 now SKIP known background/maintenance drugs that are not
      the subject of the auth request (e.g. bisoprolol, ramipril on stable therapy).
      Checking these against the formulary always raises false HITL flags because
      common generics typically have no prior-auth coverage policy.

      NOTE: We never derive CPT from ICD. We never match ICD→CPT directly.
    """
    flags = []
    cpt_code   = (cpt_result or {}).get("code", "")
    icd_code   = (icd_result or {}).get("code", "")
    drugs_list = (entities or {}).get("drugs", [])

    # Rule 1 – CPT not in database
    if cpt_code and cpt_code not in cpt_by_code:
        flags.append({
            "type": "CPT_NOT_IN_DATABASE",
            "severity": "HUMAN_REQUIRED",
            "reason": (f"Procedure code **{cpt_code}** was inferred from clinical text "
                       f"but does NOT exist in the internal CPT database. "
                       f"Automated policy matching is impossible without a valid code."),
            "suggestion": ("Use an alternative lookup to verify: "
                           "AMA CPT Lookup (ama-assn.org/practice-management/cpt), "
                           "CMS Medicare Physician Fee Schedule, or a specialty billing encoder. "
                           "Do NOT auto-approve. Reassign to medical coder.")
        })

    # Rules 2–4 – drug checks
    for drug_name in drugs_list:
        if not drug_name:
            continue

        # FIX: Skip background/maintenance drugs not being requested for auth.
        # e.g. "currently on bisoprolol 5mg" or "on ramipril 10mg OD" — these are
        # stable background meds, not the subject of this prior-auth request.
        # Running HITL on them produces false DRUG_NOT_IN_FORMULARY flags that
        # incorrectly force HUMAN_REVIEW even when the real decision should be DENY/APPROVED.
        if _hitl_is_background_drug(drug_name, sanitized_text):
            continue

        drug_rec = find_drug(drug_name)

        # Rule 2: drug not in formulary DB at all
        if not drug_rec:
            flags.append({
                "type": "DRUG_NOT_IN_FORMULARY",
                "severity": "HUMAN_REQUIRED",
                "reason": (f"Drug **'{drug_name}'** was extracted from clinical text "
                           f"but is NOT found in the formulary/drug database. "
                           f"Automated coverage check cannot proceed."),
                "suggestion": (f"Manually verify '{drug_name}' against the payer's formulary PDF "
                               f"or contact the pharmacy hotline. Check for spelling variants or brand/generic confusion.")
            })
            continue

        # Rule 3: drug in DB, but no policy covers it
        pol = find_policy_for_drug(drug_rec["id"])
        if not pol:
            flags.append({
                "type": "DRUG_NO_COVERAGE_POLICY",
                "severity": "HUMAN_REQUIRED",
                "reason": (f"Drug **'{drug_name}'** ({drug_rec.get('name','')} / {drug_rec.get('brand','')}) "
                           f"is in the formulary but NO payer coverage policy exists for it. "
                           f"Cannot make an automated prior-auth decision."),
                "suggestion": ("Escalate to clinical pharmacist or Pharmacy Benefit Manager (PBM) for manual policy review. "
                               "Request an exception or Letter of Medical Necessity if clinically appropriate.")
            })
            continue

        # Rule 4: drug has policy, but current ICD not in the policy's approved ICD list
        approved_icds = pol.get("approved_icd10_codes", [])
        if icd_code and approved_icds and icd_code not in approved_icds:
            flags.append({
                "type": "DRUG_OFF_LABEL_INDICATION",
                "severity": "HUMAN_REQUIRED",
                "reason": (f"Drug **'{drug_name}'** has a coverage policy ({pol.get('title','')}) "
                           f"but it is approved only for ICD-10 codes: {approved_icds}. "
                           f"The patient's current diagnosis code is **{icd_code}**, which is NOT on the approved list. "
                           f"This appears to be an off-label or non-covered indication."),
                "suggestion": (f"Check if a medical necessity exception letter applies. "
                               f"Verify diagnosis coding — the clinical text may support a different, covered ICD-10. "
                               f"Policy: {pol.get('title','')}")
            })

    return flags

def compute_confidences(entities, icd_result, cpt_result, policy_result, hitl_flags, final_decision):
    """
    Display scores for the confidence panel.
    Entity extraction score removed (not part of weighted decision score).
    Weighted average: CPT x0.30, ICD x0.30, Policy x0.40 (redistributed if absent).
    Pulled directly from final_decision when available.
    """
    # Use canonical scoring util so logic is consistent and auditable
    from utils.confidence import compute_weighted_score

    # ICD and CPT confidences from upstream agents (0-1)
    icd_conf = float((icd_result or {}).get("confidence", 0)) if icd_result else None
    cpt_conf = float((cpt_result or {}).get("confidence", 0)) if cpt_result else None
    # protect against inferred-but-not-in-db CPTs (downgrade confidence)
    cpt_code = (cpt_result or {}).get("code", "")
    if cpt_code and cpt_code not in cpt_by_code:
        cpt_conf = min(cpt_conf or 0.0, 0.25)

    # Compute a heuristic drug confidence (average per-requested drug)
    drug_list = (entities or {}).get("drugs", [])
    if not drug_list:
        drug_conf = None
    else:
        scores = []
        for dn in drug_list:
            dr = find_drug(dn)
            if not dr:
                scores.append(0.25)
                continue
            pol = find_policy_for_drug(dr["id"]) if dr else None
            s = 0.85 if dr else 0.25
            if pol:
                s = min(1.0, s + 0.10)
                # if ICD matches approved list, boost slightly
                approved_icds = pol.get("approved_icd10_codes", [])
                if icd_result and icd_result.get("code") and icd_result.get("code") in approved_icds:
                    s = min(1.0, s + 0.05)
            scores.append(s)
        drug_conf = round(sum(scores) / len(scores), 2) if scores else None

    # Policy score mapping: APPROVED=1.0, DENY=0.0, other/absent=0.5 (or 0.1 if explicitly absent)
    if policy_result:
        pol_raw = (policy_result.get("decision") or "").upper()
        policy_score = {"APPROVED": 1.0, "DENY": 0.0}.get(pol_raw, 0.5)
    else:
        policy_score = 0.10

    ambiguity_count = len(hitl_flags or [])

    score_obj = compute_weighted_score(
        icd_conf,
        cpt_conf,
        drug_conf,
        policy_score,
        ambiguity_count=ambiguity_count,
    )

    # Mirror previous return shape for UI usage
    return {
        "icd_matching": score_obj["component_scores"]["icd"],
        "cpt_matching": score_obj["component_scores"]["cpt"],
        "drug":         score_obj["component_scores"].get("drug", 0.0),
        "policy_match": score_obj["component_scores"]["policy"],
        "decision":     score_obj["total"],
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    case_choices = ["New Note (manual)"] + case_ids
    default_index = 1 if len(case_ids) > 0 else 0
    selected = st.selectbox("Select Case", case_choices, index=default_index)
    if selected == "New Note (manual)":
        cd = {}
    else:
        cd = case_map[selected]
    st.markdown("---")
    st.subheader("📋 Case Info")
    st.markdown(f"**Scenario:** {cd.get('scenario_name','N/A')}")
    exp   = cd.get("expected_outcome","N/A")
    ecol  = "green" if exp == "APPROVED" else "red"
    st.markdown(f"**Expected:** :{ecol}[{exp}]")
    for tag in cd.get("challenge_tags",[]):
        st.badge(tag, color="orange")
    phi_fields = cd.get("phi_fields",[])
    if phi_fields:
        st.caption(f"PHI fields: {', '.join(phi_fields)}")
    st.markdown("---")
    st.subheader("🤖 LLM Backend")
    backend_choice = st.radio("Choose backend", ["Ollama (local Llama 3)", "Gemini (cloud)"], index=0)
    if "Ollama" in backend_choice:
        os.environ["LLM_BACKEND"] = "ollama"
        ollama_model = st.selectbox("Ollama model", ["llama3","llama3.1","llama3.2","mistral","phi3"], index=0)
        os.environ["OLLAMA_MODEL"] = ollama_model
        st.info("Make sure `ollama serve` is running.")
        st.markdown("```bash\nollama pull llama3\nollama serve\n```")
    else:
        os.environ["LLM_BACKEND"] = "gemini"
        gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY",""))
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        st.caption("Uses `gemini-2.0-flash-lite` with auto-retry on 429.")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏥 Prior Authorization AI System")
st.markdown("Multi-agent LangGraph workflow — live agent I/O tracing with PHI masking")
st.divider()

col_note, col_graph = st.columns([3, 2], gap="large")
with col_note:
    st.subheader("📝 Patient Note")
    if selected == "New Note (manual)":
        raw_text = st.text_area("Clinical Text (editable)", value="", height=250)
    else:
        raw_text = st.text_area("Clinical Text (editable)", value=cd.get("raw_note","" ).strip(), height=250)

with col_graph:
    st.subheader("🗺️ LangGraph Pipeline")
    svg_h = len(AGENTS) * 48 + 20
    n_svg = a_svg = ""
    for i, (key, label, _) in enumerate(AGENTS):
        y = i * 48 + 10
        n_svg += f'<rect x="5" y="{y}" width="240" height="34" rx="7" fill="#2d3748" stroke="#4a5568" stroke-width="1.5"/>'
        n_svg += f'<text x="125" y="{y+22}" text-anchor="middle" fill="#e2e8f0" font-size="12" font-family="monospace">{label}</text>'
        if i < len(AGENTS)-1:
            ay = y+34
            a_svg += f'<line x1="125" y1="{ay}" x2="125" y2="{ay+14}" stroke="#63b3ed" stroke-width="2" marker-end="url(#arr)"/>'
    st.markdown(f"""<svg viewBox="0 0 250 {svg_h}" xmlns="http://www.w3.org/2000/svg"
        style="width:100%;max-width:280px;display:block;margin:0 auto">
      <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#63b3ed"/></marker></defs>
      {n_svg}{a_svg}</svg>""", unsafe_allow_html=True)

st.divider()

if st.button("🚀 Run Pre-Auth Workflow", type="primary", use_container_width=True):

    import importlib
    import utils.llm as _llm_mod
    importlib.reload(_llm_mod)

    # Kick off SBERT pre-warm in background to avoid blocking the UI on first encode.
    try:
        import utils.sbert_encoder as _sbert_mod
        threading.Thread(target=_sbert_mod._get_encoder, daemon=True).start()
    except Exception as _e:
        print("[SBERT] prewarm failed:", _e)

    st.subheader("⚙️ Live Agent Execution")
    placeholders = {key: st.empty() for key,_,_ in AGENTS}
    for key, label, desc in AGENTS:
        render(placeholders[key], key, label, desc)

    import agents.phi_sanitizer      as _phi
    import agents.entity_extractor   as _ent
    import agents.icd_inference      as _icd
    import agents.cpt_inference      as _cpt
    import agents.policy_evaluator   as _pol
    import agents.critique_agent     as _cri
    import agents.decision_assembler as _dec
    import agents.audit_logger       as _aud

    for mod in [_phi,_ent,_icd,_cpt,_pol,_cri,_dec,_aud]:
        importlib.reload(mod)

    MOD_MAP = {
        "phi":      (_phi,  "phi_sanitizer"),
        "entities": (_ent,  "entity_extractor"),
        "icd":      (_icd,  "icd_inference"),
        "cpt":      (_cpt,  "cpt_inference"),
        "policy":   (_pol,  "policy_evaluator"),
        "critique": (_cri,  "critique_agent"),
        "decision": (_dec,  "decision_assembler"),
        "audit":    (_aud,  "audit_logger"),
    }

    results = {}
    originals = {}

    def make_wrap(key, label, desc, fn):
        def wrapper(state):
            before = snap(state)
            render(placeholders[key], key, label, desc, status="running")
            t0 = time.time()
            try:
                result  = fn(state)
                elapsed = time.time() - t0
                after   = snap(result)
                inp_d   = agent_input(key, before)
                out_d   = diff(before, after)
                out_d.pop("audit_log", None)
                if key == "phi":
                    render(placeholders[key], key, label, desc, status="done",
                           inp=inp_d, elapsed=elapsed, phi_san=after.get("sanitized_text",""))
                else:
                    render(placeholders[key], key, label, desc, status="done",
                           inp=inp_d, out=out_d or {"note":"no field changes"}, elapsed=elapsed)
                results[key] = {"output":out_d, "elapsed":elapsed, "error":None}
                return result
            except Exception as e:
                elapsed = time.time() - t0
                results[key] = {"elapsed":elapsed, "error":str(e)}
                render(placeholders[key], key, label, desc, status="error", elapsed=elapsed, err=str(e))
                raise
        return wrapper

    for key, (mod, fn_name) in MOD_MAP.items():
        originals[key] = getattr(mod, fn_name)
        _, label, desc = next(a for a in AGENTS if a[0]==key)
        setattr(mod, fn_name, make_wrap(key, label, desc, originals[key]))

    t0_total = time.time()
    error_occurred = False
    final_state    = None

    try:
        import graph.workflow as _wf
        importlib.reload(_wf)
        from state.case_state import CaseState
        graph = _wf.build_graph()
        # For manual New Note entries, compute a stable runtime case id from the note text
        if selected == "New Note (manual)":
            txt_norm = re.sub(r"\s+", " ", (raw_text or "").strip().lower())
            run_case_id = f"manual:{hashlib.sha256(txt_norm.encode('utf-8')).hexdigest()[:12]}"
        else:
            run_case_id = selected
        final_state = graph.invoke(CaseState(case_id=run_case_id, raw_text=raw_text))
    except Exception as e:
        error_occurred = True
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            m = re.search(r"retry in ([\d.]+)s", err_str)
            wait = m.group(1) if m else "~60"
            st.error(f"⏳ **Gemini quota hit.** Wait **{wait}s** or switch to Ollama in the sidebar.")
        elif "Connection refused" in err_str or "ConnectError" in err_str:
            st.error("❌ **Cannot connect to Ollama.** Make sure `ollama serve` is running.")
        else:
            st.error(f"❌ {e}")
    finally:
        for key, (mod, fn_name) in MOD_MAP.items():
            if key in originals:
                setattr(mod, fn_name, originals[key])

    total = time.time() - t0_total

    if not error_occurred and final_state:

        # ── Pull state (graph.invoke may return dict or CaseState object) ─
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _audit(obj):
            if isinstance(obj, dict):
                return obj.get("audit_log", [])
            return getattr(obj, "audit_log", [])

        entities       = _get(final_state, "entities") or {}
        icd_result     = _get(final_state, "icd") or {}
        cpt_result     = _get(final_state, "cpt") or {}
        policy_result  = _get(final_state, "policy") or {}
        final_decision = _get(final_state, "final_decision") or {}

        # ── FIX: pass sanitized_text into HITL so background drug filter works ──
        sanitized_text = _get(final_state, "sanitized_text") or ""
        hitl_flags = run_hitl_checks(icd_result, cpt_result, entities,
                                     sanitized_text=sanitized_text)

        # ── Confidence scores ─────────────────────────────────────────────
        conf = compute_confidences(entities, icd_result, cpt_result,
                                   policy_result, hitl_flags, final_decision)
        avg_conf  = round(conf.get("decision", 0), 2)
        avg_color = "#48bb78" if avg_conf >= 0.75 else ("#f6ad55" if avg_conf >= 0.55 else "#fc8181")

        # ── Verdict — read final_decision first (assembler is authoritative) ──
        if final_decision.get("decision") == "HUMAN_REVIEW" or hitl_flags:
            verdict     = "HUMAN REVIEW NEEDED"
            verdict_css = "human"
            icon        = "🔶"
            fd_reason   = (final_decision or {}).get("reason", "")
            if fd_reason and not hitl_flags:
                reason_html = f"<strong>Human review required.</strong> {fd_reason}"
            else:
                reason_html = (
                    f"<strong>{len(hitl_flags)} issue(s) detected</strong> that require human review "
                    f"before a prior-authorization decision can be made. See flags below for details."
                )
        else:
            raw = (final_decision.get("decision") or policy_result.get("decision","UNKNOWN")).upper()
            if raw == "HUMAN_REVIEW":
                verdict     = "HUMAN REVIEW NEEDED"
                verdict_css = "human"
                icon        = "🔶"
                fd_reason   = (final_decision or {}).get("reason", "")
                reason_html = (
                    f"<strong>Human review required.</strong> {fd_reason}"
                ) if fd_reason else (
                    "<strong>Automated decision blocked.</strong> Issues flagged by the critique agent require clinical review."
                )
            elif raw == "HITL_NEEDED":
                verdict     = "HITL needed"
                verdict_css = "human"
                icon        = "🔶"
                fd_reason   = (final_decision or {}).get("reason", "")
                ambiguity_list = (final_decision or {}).get("ambiguity_count", 0)
                reason_html = (
                    f"<strong>Additional information required before a decision can be made.</strong> {fd_reason}"
                ) if fd_reason else (
                    "<strong>Ambiguities detected.</strong> Missing or unclear clinical information must be resolved by a human reviewer."
                )
                verdict     = "HUMAN REVIEW NEEDED"
                verdict_css = "human"
                icon        = "🔶"
                fd_reason   = (final_decision or {}).get("reason", "")
                reason_html = (
                    f"<strong>Human review required.</strong> {fd_reason}"
                ) if fd_reason else (
                    "<strong>Automated decision blocked.</strong> Issues flagged by the critique agent require clinical review."
                )
            elif raw == "APPROVED":
                verdict     = "APPROVED"
                verdict_css = "approved"
                icon        = "✅"
                drugs_str   = ", ".join(entities.get("drugs", [])) or "none mentioned"
                reason_html = (
                    f"ICD-10 <strong>{icd_result.get('code','')}</strong> and "
                    f"CPT <strong>{cpt_result.get('code','')}</strong> are valid covered codes. "
                    f"Drug(s) <strong>{drugs_str}</strong> are on formulary with an active policy "
                    f"covering the indicated diagnosis. All prior-authorization criteria are met."
                )
            elif raw in ("DENY", "DENIED"):
                verdict     = "DENIED"
                verdict_css = "denied"
                icon        = "❌"
                fd_reason   = (final_decision or {}).get("reason", "")
                if fd_reason:
                    reason_html = fd_reason
                else:
                    reason_html = (
                        f"Policy evaluation returned <strong>DENY</strong> for "
                        f"ICD-10 <strong>{icd_result.get('code','')}</strong>. "
                        f"The requested drug or procedure does not meet coverage criteria under the "
                        f"applicable payer policy. Review step-therapy requirements or alternative diagnoses."
                    )
            else:
                verdict     = raw or "UNKNOWN"
                verdict_css = "human"
                icon        = "⚠️"
                reason_html = f"Automated decision could not be determined. Raw outcome: <strong>{raw}</strong>"

        st.divider()

        # ── Clean, human-friendly JSON output for non-technical users
        st.subheader("🧾 Clean Extraction Output")
        clean_entities = {
            "entities": {
                "conditions": entities.get("conditions", []),
                "drugs_needed": entities.get("drugs", []),
                "procedures_needed": entities.get("procedures_needed", []),
                "previous_procedures": entities.get("previous_current_procedures", []),
                "previous_drugs": entities.get("previous_current_drugs", []),
                "habits_or_notes": entities.get("habits_or_notes", []),
                "documents": entities.get("documents_provided", []),
                "visit_type": entities.get("visit_type", "")
            }
        }
        st.code(json.dumps(clean_entities, indent=2), language="json")

        # ════════════════════════════════════════════════════════════════
        #  DECISION BANNER
        # ════════════════════════════════════════════════════════════════
        st.subheader("🏁 Prior Authorization Decision")

        hitl_badge = '<span class="hitl-badge">👤 HUMAN NEEDED</span>' if hitl_flags else ""
        st.markdown(f"""
        <div class="decision-{verdict_css}">
            <div class="decision-title">{icon} {verdict}{hitl_badge}</div>
            <div class="decision-reason"><strong>Reason:</strong> {reason_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── HITL Flag Cards ───────────────────────────────────────────────
        if hitl_flags:
            st.markdown("#### 🚨 Human-in-the-Loop Flags")
            for i, flag in enumerate(hitl_flags, 1):
                sev_color = "#fc8181" if flag["severity"] == "HUMAN_REQUIRED" else "#f6ad55"
                st.markdown(f"""
                <div style="border-left:4px solid {sev_color};background:#1a202c;border-radius:8px;
                            padding:14px 18px;margin-bottom:10px;">
                    <div style="font-size:13px;font-weight:700;color:{sev_color};margin-bottom:4px">
                        🚩 Flag {i} — {flag['type']}
                    </div>
                    <div style="font-size:13px;color:#e2e8f0;line-height:1.6">
                        <strong>Reason:</strong> {flag['reason']}
                    </div>
                    <div style="font-size:12px;color:#90cdf4;margin-top:6px;line-height:1.5">
                        <strong>Suggested action:</strong> {flag.get('suggestion','')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ════════════════════════════════════════════════════════════════
        #  CONFIDENCE SCORES
        # ════════════════════════════════════════════════════════════════
        st.subheader("📊 Confidence Scores")

        score_labels = {
            "icd_matching":  "1️⃣  ICD-10 Code Match (diagnosis accuracy)  ·  weight 0.30",
            "cpt_matching":  "2️⃣  CPT Procedure Match (procedure accuracy) ·  weight 0.25",
            "drug":          "3️⃣  Drug Match / Coverage (drug certainty) ·  weight 0.20",
            "policy_match":  "4️⃣  Policy Coverage Match (payer rule alignment) ·  weight 0.15",
            "decision":      "5️⃣  Weighted Decision Score (final confidence)",
        }
        fd_scoring = (final_decision or {}).get("scoring_logic", "")
        fd_weights = (final_decision or {}).get("weights_used", {})
        w_icd = fd_weights.get("icd", 0.3)
        w_cpt = fd_weights.get("cpt", 0.3)
        w_pol = fd_weights.get("policy", 0.4)
        avg_conf_weighted = round(
            w_icd * conf["icd_matching"] + w_cpt * conf["cpt_matching"] + w_pol * conf["policy_match"], 2
        )
        avg_color = "#48bb78" if avg_conf_weighted >= 0.75 else ("#f6ad55" if avg_conf_weighted >= 0.55 else "#fc8181")
        bars_html = "".join(conf_bar_html(score_labels[k], conf.get(k, 0.0)) for k in score_labels)
        scoring_note = f'<div style="font-size:11px;color:#718096;margin-top:6px">Scoring: {fd_scoring}</div>' if fd_scoring else ""
        bars_html += f'<hr style="border-color:#2d3748;margin:10px 0">{conf_bar_html("⭐ Weighted Average (ICD·0.30 + CPT·0.25 + Drug·0.20 + Policy·0.15)", avg_conf_weighted, avg_color)}{scoring_note}'

        st.markdown(f"""
        <div style="background:#1a202c;border:1px solid #2d3748;border-radius:12px;padding:18px 22px;">
            {bars_html}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ════════════════════════════════════════════════════════════════
        #  EXTRACTED CLINICAL DETAILS
        # ════════════════════════════════════════════════════════════════
        st.subheader("🔬 Extracted Clinical Details")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**🏷️ ICD-10 Diagnosis**")
            if icd_result and icd_result.get("code"):
                rec  = icd_by_code.get(icd_result["code"])
                desc = rec.get("description","") if rec else "unknown"
                pct  = int(conf["icd_matching"] * 100)
                col  = "green" if pct >= 80 else ("orange" if pct >= 60 else "red")
                st.markdown(f"`{icd_result['code']}` — {desc}")
                st.markdown(f":{col}[Confidence: {pct}%]")
            else:
                st.warning("No ICD-10 code inferred")

        with col_b:
            st.markdown("**💊 CPT Procedure**")
            if cpt_result and cpt_result.get("code"):
                code    = cpt_result["code"]
                in_db   = code in cpt_by_code
                cpt_rec = cpt_by_code.get(code)
                cpt_desc= cpt_rec.get("description","") if cpt_rec else "⚠️ NOT IN CPT DATABASE"
                pct     = int(conf["cpt_matching"] * 100)
                col     = "green" if pct >= 80 else ("orange" if pct >= 60 else "red")
                st.markdown(f"`{code}` — {cpt_desc}")
                if not in_db:
                    st.error("⚠️ Code not in CPT database → Human review required")
                else:
                    st.markdown(f":{col}[Confidence: {pct}%]")
            else:
                st.warning("No CPT code inferred")

        with col_c:
            st.markdown("**💊 Drugs Requested (for Auth)**")
            drugs_list = entities.get("drugs", [])
            drugs_with_dur = entities.get("drugs_requested_with_duration", [])
            dur_map = {
                (e["drug"].lower() if isinstance(e, dict) else ""): e.get("duration", "not specified")
                for e in drugs_with_dur if isinstance(e, dict)
            }
            if drugs_list:
                for dn in drugs_list:
                    dr  = find_drug(dn)
                    dur = dur_map.get(dn.lower(), "not specified")
                    dur_str = f" · *{dur}*" if dur and dur != "not specified" else ""
                    if dr:
                        pol = find_policy_for_drug(dr["id"])
                        if pol:
                            approved = icd_result.get("code","") in pol.get("approved_icd10_codes",[])
                            badge    = "✅ Covered (on-label)" if approved else "⚠️ Off-label / ICD mismatch"
                        else:
                            badge = "⚠️ No policy found"
                        st.markdown(f"• **{dn}**{dur_str} — {badge}")
                    else:
                        st.markdown(f"• **{dn}**{dur_str} — ❌ Not in formulary")
            else:
                st.info("No drugs extracted from note")

        # ── Additional entity fields row ──────────────────────────────────
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.markdown("**🕐 Previous / Current Drugs**")
            prev_drugs = entities.get("previous_current_drugs", [])
            if prev_drugs:
                for d in prev_drugs:
                    st.markdown(f"• {d}")
            else:
                st.caption("None documented")

        with col_e:
            st.markdown("**🔧 Previous / Current Procedures**")
            prev_procs = entities.get("previous_current_procedures", [])
            if prev_procs:
                for p in prev_procs:
                    st.markdown(f"• {p}")
            else:
                st.caption("None documented")

        with col_f:
            st.markdown("**📄 Documents Provided**")
            docs = entities.get("documents_provided", [])
            if docs:
                for d in docs:
                    st.markdown(f"• {d}")
            else:
                st.caption("None documented")

        st.divider()

        # ════════════════════════════════════════════════════════════════
        #  METRICS
        # ════════════════════════════════════════════════════════════════
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Time",      f"{total:.2f}s")
        m2.metric("Agents Ran",      f"{sum(1 for v in results.values() if not v.get('error'))}/{len(AGENTS)}")
        m3.metric("HITL Flags",      str(len(hitl_flags)))
        m4.metric("Avg Confidence",  f"{avg_conf:.0%}")

        st.divider()

        # ════════════════════════════════════════════════════════════════
        #  AGENT SUMMARY
        # ════════════════════════════════════════════════════════════════
        st.subheader("📋 Agent I/O Summary")
        rows = []
        for key, label, _ in AGENTS:
            r = results.get(key, {})
            out_keys = ", ".join(k for k in r.get("output", {}).keys() if k != "note") or "—"
            rows.append({"Agent": label,
                         "Status": "✅" if not r.get("error") else "❌",
                         "Time (s)": f"{r.get('elapsed',0):.2f}",
                         "Output fields": out_keys})
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("📋 Full Audit Log"):
            for entry in _audit(final_state):
                st.json(entry)