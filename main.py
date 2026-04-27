"""
Entry point — runs one PA case through the full 9-agent LangGraph pipeline.
Edit 'case_id' and 'raw_text' to test different cases.

To use Ollama (local, recommended):
  Set LLM_BACKEND=ollama in .env  (default — no key needed)
  Then: ollama pull llama3 && ollama serve

To use Gemini:
  Set LLM_BACKEND=gemini in .env
  Set GEMINI_API_KEY=your_key  in .env
"""

import json
from graph.workflow import build_graph
from state.case_state import CaseState

graph = build_graph()

initial_state = CaseState(
    case_id="CASE_003",
    raw_text="""
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
    """
)

final_state = graph.invoke(initial_state)

print("\n" + "=" * 60)
print("  FINAL DECISION OUTPUT")
print("=" * 60)
print(json.dumps(final_state.final_decision, indent=2))
