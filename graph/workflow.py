"""
LangGraph Workflow — 9-Agent Prior Authorization Pipeline
=========================================================

Agent order matches the design doc exactly:

  Agent 1  phi          PHI Masking (deterministic, regex + Presidio)
  Agent 2  entities     Entity Extraction (LLM)
  Agent 3  icd          ICD-10 Inference (SBERT + alias + LLM)
  Agent 4  cpt          CPT Inference (SBERT + alias + LLM)
  Agent 5  drug         Drug Validation (deterministic, formulary lookup)
  Agent 6  policy       Policy Evaluation (RAG + structured rules)
  Agent 7  ambiguity    Ambiguity Detection (LLM — semantic gap finding)
  Agent 8  decision     Confidence Scoring + Final Decision
  Agent 9  critique     Critique Agent (LLM quality control)
  audit    audit        Audit Logger (always last)
"""

from langgraph.graph import StateGraph
from state.case_state import CaseState

from agents.phi_sanitizer      import phi_sanitizer
from agents.entity_extractor   import entity_extractor
from agents.icd_inference      import icd_inference
from agents.cpt_inference      import cpt_inference
from agents.drug_validator     import drug_validator
from agents.policy_evaluator   import policy_evaluator
from agents.ambiguity_detector import ambiguity_detector
from agents.decision_assembler import decision_assembler
from agents.critique_agent     import critique_agent
from agents.audit_logger       import audit_logger


def build_graph():
    graph = StateGraph(CaseState)

    # Register nodes
    graph.add_node("phi",       phi_sanitizer)       # Agent 1
    graph.add_node("entities",  entity_extractor)    # Agent 2
    graph.add_node("icd",       icd_inference)       # Agent 3
    graph.add_node("cpt",       cpt_inference)       # Agent 4
    graph.add_node("drug",      drug_validator)      # Agent 5
    graph.add_node("policy",    policy_evaluator)    # Agent 6
    graph.add_node("ambiguity", ambiguity_detector)  # Agent 7
    graph.add_node("decision",  decision_assembler)  # Agent 8
    graph.add_node("critique",  critique_agent)      # Agent 9
    graph.add_node("audit",     audit_logger)

    # Entry point
    graph.set_entry_point("phi")

    # Linear pipeline edges (design doc flow)
    graph.add_edge("phi",       "entities")
    graph.add_edge("entities",  "icd")
    graph.add_edge("icd",       "cpt")
    graph.add_edge("cpt",       "drug")
    graph.add_edge("drug",      "policy")
    graph.add_edge("policy",    "ambiguity")
    graph.add_edge("ambiguity", "decision")
    graph.add_edge("decision",  "critique")
    graph.add_edge("critique",  "audit")

    return graph.compile()
