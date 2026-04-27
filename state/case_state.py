from pydantic import BaseModel
from typing import Dict, Any, List, Optional


class CaseState(BaseModel):
    case_id: str

    # ── Input ─────────────────────────────────────────────────────────────────
    raw_text: str
    sanitized_text: Optional[str] = None

    # ── Agent outputs ─────────────────────────────────────────────────────────
    entities:    Dict[str, Any]  = {}   # Agent 2: conditions, drugs, visit_type
    icd:         Dict[str, Any]  = {}   # Agent 3: ICD-10 code + confidence
    cpt:         Dict[str, Any]  = {}   # Agent 4: CPT code + confidence
    drug_checks: List[Dict[str, Any]] = []  # Agent 5: per-drug validation results
    policy:      Dict[str, Any]  = {}   # Agent 6: policy evaluation results
    ambiguities: List[str]       = []   # Agent 7: list of missing/unclear items
    critique:    Dict[str, Any]  = {}   # Agent 9: quality flags

    # ── Decision ──────────────────────────────────────────────────────────────
    final_decision: Dict[str, Any] = {}  # Agent 8: scored decision

    # ── Governance ────────────────────────────────────────────────────────────
    audit_log: List[Dict[str, Any]] = []
