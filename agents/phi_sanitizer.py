"""
PHI Sanitizer — dual-layer de-identification
============================================
Layer 1 : Regex patterns (fast, deterministic) — HIPAA-aligned
Layer 2 : Microsoft Presidio NER (spaCy-based) — catches names/orgs/locations
          that regex misses (e.g. uncommon names, informal references)

Presidio is optional: if it is not installed the sanitizer falls back to
regex-only mode and logs a warning in the audit trail.
"""

import re

# ── Regex patterns (HIPAA-aligned) ───────────────────────────────────────────
_PATTERNS = [
    # Full name variants  "Pt: Vinod Rao," / "Patient: Rajesh Kumar,"
    (r'\b(Patient|Pt)\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', r'\1: [NAME]'),
    # Doctor name after "Dr" / "Dr."
    (r'\bDr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', '[PHYSICIAN]'),
    # Date of Birth
    (r'\b(DOB|Date of Birth)\s*[:\-]?\s*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}', r'\1: [DOB]'),
    # MRN / Member ID / Mem ID
    (r'\b(MRN|Member\s*ID|Mem(?:ber)?)\s*[:\-]?\s*[A-Z0-9\-]+', r'\1: [ID]'),
    # Insurance member numbers like "BS-9876001"
    (r'\b[A-Z]{2,4}-\d{6,10}\b', '[MEMBER_ID]'),
    # Standalone dates  "22 Jan 2025" / "14/01/2025" / "January 22, 2025"
    (r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', '[DATE]'),
    (r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b', '[DATE]'),
    (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', '[DATE]'),
    # Phone numbers
    (r'\b(\+91[-\s]?)?\d{10}\b', '[PHONE]'),
    (r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', '[PHONE]'),
    # Email
    (r'\b[\w.+-]+@[\w.-]+\.\w{2,}\b', '[EMAIL]'),
    # Aadhaar / PAN (India)
    (r'\b\d{4}\s\d{4}\s\d{4}\b', '[AADHAAR]'),
    (r'\b[A-Z]{5}\d{4}[A-Z]\b', '[PAN]'),
    # Address fragments / PIN codes
    (r'\b\d{6}\b', '[PINCODE]'),
]

_COMPILED = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _PATTERNS]


def _regex_mask(text: str) -> str:
    for pattern, replacement in _COMPILED:
        text = pattern.sub(replacement, text)
    return text


# ── Presidio NER layer ────────────────────────────────────────────────────────
_presidio_available = False
_analyzer = None

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Use the spaCy en_core_web_lg model if available, fall back to sm
    try:
        import spacy
        _model = "en_core_web_lg" if spacy.util.is_package("en_core_web_lg") else "en_core_web_sm"
    except Exception:
        _model = "en_core_web_sm"

    _provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": _model}],
    })
    _analyzer = AnalyzerEngine(nlp_engine=_provider.create_engine(), supported_languages=["en"])
    _presidio_available = True
except Exception:
    pass   # Presidio not installed — regex-only mode


# Entity types Presidio should redact (HIPAA + clinical context)
_PRESIDIO_ENTITIES = [
    "PERSON",           # patient names, physician names not caught by regex
    "LOCATION",         # addresses, cities
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_SSN",
    "MEDICAL_LICENSE",
    "NRP",              # nationality / religion / political affiliation
]

_PRESIDIO_LABEL_MAP = {
    "PERSON":           "[NAME]",
    "LOCATION":         "[LOCATION]",
    "PHONE_NUMBER":     "[PHONE]",
    "EMAIL_ADDRESS":    "[EMAIL]",
    "DATE_TIME":        "[DATE]",
    "US_DRIVER_LICENSE":"[ID]",
    "US_PASSPORT":      "[ID]",
    "US_SSN":           "[SSN]",
    "MEDICAL_LICENSE":  "[LICENSE]",
    "NRP":              "[NRP]",
}


def _presidio_mask(text: str) -> tuple[str, int]:
    """Run Presidio NER and replace detected PHI spans. Returns (masked_text, count)."""
    if not _presidio_available or _analyzer is None:
        return text, 0

    try:
        results = _analyzer.analyze(text=text, language="en", entities=_PRESIDIO_ENTITIES)
        # Sort by start position descending so replacements don't shift offsets
        results = sorted(results, key=lambda r: r.start, reverse=True)
        count = 0
        for r in results:
            label = _PRESIDIO_LABEL_MAP.get(r.entity_type, f"[{r.entity_type}]")
            # Only replace if not already masked by regex (avoid double-masking)
            span = text[r.start:r.end]
            if not (span.startswith("[") and span.endswith("]")):
                text = text[:r.start] + label + text[r.end:]
                count += 1
        return text, count
    except Exception:
        return text, 0


def phi_sanitizer(state):
    original = state.raw_text

    # Layer 1: regex
    after_regex = _regex_mask(original)
    regex_count = after_regex.count("[")

    # Layer 2: Presidio NER (on top of regex output)
    sanitized, presidio_count = _presidio_mask(after_regex)

    state.sanitized_text = sanitized
    state.audit_log.append({
        "step":             "PHI_SANITIZED",
        "mode":             "regex+presidio" if _presidio_available else "regex_only",
        "presidio_model":   _model if _presidio_available else "unavailable",
        "regex_redactions": regex_count,
        "presidio_redactions": presidio_count,
        "total_redactions": sanitized.count("["),
        "original_len":     len(original),
        "sanitized_len":    len(sanitized),
    })
    return state
