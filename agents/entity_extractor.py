import json, re, os, hashlib
from utils.llm import llm

# Simple file-based cache for entity extraction (keyed by sanitized text hash)
_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
_CACHE_FILE = os.path.join(_CACHE_DIR, "entity_extractor.json")

def _load_cache() -> dict:
    try:
        if not os.path.isdir(_CACHE_DIR):
            os.makedirs(_CACHE_DIR, exist_ok=True)
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_cache(cache: dict) -> None:
    try:
        # atomic write
        tmp = _CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        os.replace(tmp, _CACHE_FILE)
    except Exception:
        pass


def entity_extractor(state):
    prompt = """Extract medical entities from the clinical note below.

CRITICAL DISTINCTION — this is a PRIOR AUTHORIZATION system:
- Only extract drugs that are being REQUESTED for prior authorization (new drugs, additions, escalations) into the "drugs_requested_with_duration" list and `drugs` list.
- Extract drugs the patient is currently taking (background/maintenance) into "previous_current_drugs".
- Extract drugs the patient has previously used or discontinued into "previous_current_drugs" as well (mark context if clear).
- Do NOT mix requested drugs and background drugs.
- The drug being requested is usually in phrases like: "requesting X", "plan: start X", "addition of X", "escalate to X", "initiate X", "requesting approval for X".

Also extract the following keys (use empty lists / empty string if absent):
- "conditions": list of detected clinical conditions/diagnoses (e.g. "heart failure").
- "drugs": list of requested drug names (canonical short names) — used downstream as `entities.drugs`.
- "drugs_requested_with_duration": list of objects with "drug" and "duration" (duration = "not specified" if not present).
- "procedures_needed": list of procedures being requested for authorization (e.g. "NGS panel").
- "previous_current_procedures": procedures the patient has already had.
- "previous_current_drugs": drugs the patient is currently or previously taking.
- "habits_or_notes": short strings for habits or special notes (e.g. "smoker").
- "documents_provided": referenced documents (e.g. "NGS report").
- "visit_type": one of: cardiology, oncology, neurology, pulmonology, nephrology, endocrinology, rheumatology, general_medicine, surgery, hematology, dermatology, gastroenterology, psychiatric, other. Use empty string if unknown.

Return ONLY valid JSON — no preamble, no markdown, no backticks. Exactly include keys shown below (empty lists/strings if missing):
{{
    "conditions": ["condition1"],
    "drugs": ["upadacitinib"],
    "drugs_requested_with_duration": [{{"drug": "drug1", "duration": "not specified"}}],
    "procedures_needed": ["procedure1"],
    "previous_current_drugs": ["drug2"],
    "previous_current_procedures": ["procedure1"],
    "habits_or_notes": ["smoker"],
    "documents_provided": ["doc1"],
    "visit_type": "cardiology"
}}

TEXT:
""" + (state.sanitized_text or "")

    # Try cache first: prefer `case_id` when present (stable), otherwise hash normalized text
    cache = _load_cache()
    key = None
    case_id = getattr(state, "case_id", None) or (state.__dict__.get("case_id") if isinstance(state, object) else None)
    if case_id:
        key = f"case:{case_id}"
    else:
        txt = (state.sanitized_text or "").strip().lower()
        # Normalize whitespace to avoid trivial mismatches
        txt = re.sub(r"\s+", " ", txt)
        key = hashlib.sha256(txt.encode("utf-8")).hexdigest()

    if key in cache:
        parsed = cache[key]
        state.entities = parsed
        state.audit_log.append({"step": "ENTITIES_FROM_CACHE"})
        return state

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if model wrapped the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # First try: parse whole response
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract the first JSON object in the response body
            m = re.search(r"\{(?:.|\n)*\}", raw)
            if m:
                sub = m.group(0)
                try:
                    parsed = json.loads(sub)
                except json.JSONDecodeError:
                    raise
            else:
                raise

        # Ensure expected keys exist and normalise
        parsed.setdefault("conditions", [])
        parsed.setdefault("drugs", [])
        parsed.setdefault("drugs_requested_with_duration", [])
        parsed.setdefault("procedures_needed", [])
        parsed.setdefault("previous_current_drugs", [])
        parsed.setdefault("previous_current_procedures", [])
        parsed.setdefault("habits_or_notes", [])
        parsed.setdefault("documents_provided", [])
        parsed.setdefault("visit_type", "")

        # Normalise: build canonical "drugs" list from requested drugs (for downstream agents)
        requested_with_dur = parsed.get("drugs_requested_with_duration", [])
        if isinstance(requested_with_dur, list):
            parsed["drugs"] = [
                entry["drug"] if isinstance(entry, dict) else str(entry)
                for entry in requested_with_dur
            ]
        else:
            parsed["drugs"] = []

        # Ensure all new fields are present with defaults
        parsed.setdefault("previous_current_drugs", [])
        parsed.setdefault("previous_current_procedures", [])
        parsed.setdefault("documents_provided", [])

        # Post-process: strip background drugs the LLM hallucinated into the requested list
        parsed["drugs"] = _filter_llm_drugs(
            parsed.get("drugs", []),
            state.sanitized_text or ""
        )
        # Sync filtered drugs back into drugs_requested_with_duration
        filtered_set = {d.lower() for d in parsed["drugs"]}
        parsed["drugs_requested_with_duration"] = [
            entry for entry in requested_with_dur
            if isinstance(entry, dict) and entry.get("drug", "").lower() in filtered_set
        ]

        # Heuristic / regex-based fallbacks to populate missing fields
        text_for_heuristics = (state.sanitized_text or "")
        _heuristic_fill(parsed, text_for_heuristics)

        # Save to cache (use same key logic)
        try:
            cache[key] = parsed
            _save_cache(cache)
            state.audit_log.append({"step": "ENTITIES_CACHED"})
        except Exception:
            pass

        state.entities = parsed
        state.audit_log.append({"step": "ENTITIES_EXTRACTED",
                                 "drugs": state.entities.get("drugs", []),
                                 "conditions": state.entities.get("conditions", [])})
    except Exception as e:
        # Try to recover: log raw output and run a fuller heuristic fallback
        try:
            raw_snippet = raw if isinstance(raw, str) else ""
        except Exception:
            raw_snippet = ""
        parsed_fb = _fallback_extract(state.sanitized_text or "")
        # ensure full schema
        parsed_fb.setdefault("conditions", [])
        parsed_fb.setdefault("drugs", [])
        parsed_fb.setdefault("drugs_requested_with_duration", [])
        parsed_fb.setdefault("procedures_needed", [])
        parsed_fb.setdefault("previous_current_drugs", [])
        parsed_fb.setdefault("previous_current_procedures", [])
        parsed_fb.setdefault("habits_or_notes", [])
        parsed_fb.setdefault("documents_provided", [])
        parsed_fb.setdefault("visit_type", "")

        # run heuristics to try to populate missing pieces
        _heuristic_fill(parsed_fb, state.sanitized_text or "")

        state.entities = parsed_fb
        state.audit_log.append({"step": "ENTITY_FALLBACK_USED", "error": str(e), "raw": raw_snippet,
                                 "drugs": state.entities.get("drugs", [])})

    return state


def _filter_llm_drugs(drugs: list, text: str) -> list:
    """
    Post-process LLM drug list:
    - Keep AUTH_REQUEST_DRUGS always (they are only ever auth targets).
    - For BACKGROUND_DRUGS, only keep if found near a request-context phrase.
    - Unknown drugs (not in either list) are kept and passed to HITL.
    """
    tl = text.lower()
    requested_tokens = _get_requested_tokens(tl)
    auth_set       = {d.lower() for d in _AUTH_REQUEST_DRUGS}
    background_set = {d.lower() for d in _BACKGROUND_DRUGS}

    filtered = []
    for drug in drugs:
        dl = drug.lower().strip()
        if dl in auth_set:
            filtered.append(drug)
        elif dl in background_set:
            if any(tok in dl or dl in tok for tok in requested_tokens):
                filtered.append(drug)
            # else: background drug not in request context — drop it
        else:
            filtered.append(drug)  # unknown drug — let HITL handle

    return filtered


def _heuristic_fill(parsed: dict, text: str) -> None:
    """Populate common missing fields from simple regex heuristics."""
    tl = (text or "").lower()

    # procedures_needed: look for requested procedures or named tests/reports
    if not parsed.get("procedures_needed"):
        procs = set()
        for pat in [r"ngs", r"pyp scan", r"pyp scan", r"pyp", r"technetium-99m pyp", r"echo", r"echocardiogram", r"ct scan", r"ct", r"mri", r"biopsy", r"ngs panel"]:
            if re.search(pat, tl):
                procs.add(pat)
        parsed["procedures_needed"] = list(procs)

    # previous_current_procedures: look for 'underwent', 'had X done', 'report attached'
    if not parsed.get("previous_current_procedures"):
        prev = set()
        for m in re.finditer(r"(?:underwent|had|performed|report attached|report enclosed|report available)[:\s]+([A-Za-z0-9\-\s]+)", text, flags=re.IGNORECASE):
            prev.add(m.group(1).strip())
        # fallback: collect phrases like 'echo', 'pyp scan'
        for pat in [r"echo", r"pyp scan", r"ct scan", r"mri", r"ngs"]:
            if re.search(pat, tl):
                prev.add(pat)
        parsed["previous_current_procedures"] = list(prev)

    # previous_current_drugs: use history patterns
    if not parsed.get("previous_current_drugs"):
        prev_drugs = set()
        hist_patterns = [r"previously\s+on\s+([A-Za-z0-9\-]+)", r"was\s+on\s+([A-Za-z0-9\-]+)", r"tried\s+([A-Za-z0-9\-]+)", r"discontinued\s+([A-Za-z0-9\-]+)"]
        for pat in hist_patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                prev_drugs.add(m.group(1).strip())
        parsed["previous_current_drugs"] = list(prev_drugs)

    # habits_or_notes
    if not parsed.get("habits_or_notes"):
        notes = []
        if re.search(r"smok|smoker", tl):
            notes.append("smoker")
        if re.search(r"alcohol|drinks|etoh", tl):
            notes.append("alcohol use")
        if re.search(r"non-?compliance|noncompliant", tl):
            notes.append("non-compliant")
        parsed["habits_or_notes"] = notes

    # documents_provided: look for 'report', 'attestation', 'letter', 'report enclosed'
    if not parsed.get("documents_provided"):
        docs = set()
        for m in re.finditer(r"(pyp scan report|echo report|cardiology attestation|attestation letter|report attached|report enclosed|ngs report|pathology report|lab results)", text, flags=re.IGNORECASE):
            docs.add(m.group(1).strip())
        parsed["documents_provided"] = list(docs)

    # visit_type inference
    if not parsed.get("visit_type"):
        vt = ""
        if re.search(r"cardio|heart|cardiology|cardiac", tl):
            vt = "cardiology"
        elif re.search(r"oncolog|cancer|tumor|neoplasm", tl):
            vt = "oncology"
        elif re.search(r"neuro|neurology", tl):
            vt = "neurology"
        parsed["visit_type"] = vt


def _get_requested_tokens(text_lower: str) -> set:
    """Return word tokens extracted from request-context phrases in the text."""
    tokens = set()
    for pat in _REQUEST_PHRASES:
        for m in re.finditer(pat, text_lower):
            phrase = m.group(1).strip().lower()
            for tok in re.split(r"[\s\-]+", phrase):
                if len(tok) > 3:
                    tokens.add(tok)
    return tokens


# Drugs that can only appear as AUTH REQUEST targets
_AUTH_REQUEST_DRUGS = [
    "tafamidis", "vyndamax", "vyndaqel",
    "dapagliflozin", "farxiga",
    "empagliflozin", "jardiance",
    "apixaban", "eliquis",
    "rivaroxaban", "xarelto",
    "sacubitril", "entresto",
    "osimertinib", "tagrisso",
    "pembrolizumab", "keytruda",
    "trastuzumab", "herceptin",
    "adalimumab", "humira",
    "ocrelizumab", "ocrevus",
    "upadacitinib", "rinvoq",
    "semaglutide", "ozempic", "wegovy",
    "metformin",
]

# Drugs that are typically background/maintenance medications
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

_KNOWN_CONDITIONS = [
    "ATTR amyloidosis", "transthyretin amyloid cardiomyopathy",
    "cardiac amyloidosis", "wild-type ATTR",
    "heart failure", "HFrEF", "atrial fibrillation", "AFib",
    "hypertension", "NSCLC", "rheumatoid arthritis",
    "multiple sclerosis", "RRMS", "relapsing-remitting MS",
    "diabetes", "CKD", "COPD",
    "breast cancer", "amyloidosis",
]


def _fallback_extract(text: str) -> dict:
    tl = text.lower()
    requested_drug_tokens = _get_requested_tokens(tl)

    _HISTORY_PATTERNS = [
        r"previously\s+on\s+(\w+)",
        r"was\s+on\s+(\w+)",
        r"tried\s+(\w+)",
        r"discontinued\s+(\w+)",
        r"stopped\s+(\w+)",
        r"history\s+of\s+(\w+)\s+use",
        r"past\s+(?:use\s+of\s+)?(\w+)",
        r"(\w+)\s+(?:was|were)\s+(?:previously|formerly)\s+(?:used|prescribed|taken)",
    ]
    historical_drugs = set()
    for pattern in _HISTORY_PATTERNS:
        for m in re.finditer(pattern, tl):
            historical_drugs.add(m.group(1).lower())

    drugs = []
    seen = set()

    for d in _AUTH_REQUEST_DRUGS:
        if d in tl and d.lower() not in historical_drugs:
            drugs.append(d)

    for d in _BACKGROUND_DRUGS:
        if d in tl and d.lower() not in historical_drugs:
            if any(tok in d or d in tok for tok in requested_drug_tokens):
                drugs.append(d)

    for d in sorted(drugs, key=len, reverse=True):
        if not any(d in s for s in seen):
            seen.add(d)

    conditions = [c for c in _KNOWN_CONDITIONS if c.lower() in tl]
    return {"conditions": conditions, "drugs": list(seen), "visit_type": ""}