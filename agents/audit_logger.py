"""
Audit Logger — final node
===========================
Prints a clean, structured summary of the full agent trace.
This is what a compliance auditor or developer would review.
"""


def audit_logger(state):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CASE: {state.case_id}")
    print(sep)

    # Agent trace
    print("\n📋 AGENT TRACE:")
    for entry in state.audit_log:
        step = entry.get("step", "?")
        rest = {k: v for k, v in entry.items() if k != "step"}
        if rest:
            print(f"  [{step}]  {rest}")
        else:
            print(f"  [{step}]")

    # Drug checks (Agent 5)
    if state.drug_checks:
        print("\n💊 DRUG VALIDATION (Agent 5):")
        for c in state.drug_checks:
            status = "✅ FOUND" if c.get("found") else "❌ UNKNOWN"
            print(f"  {c['drug']:30s}  {status}  | {c.get('note', '')}")

    # Ambiguities (Agent 7)
    if state.ambiguities:
        print(f"\n⚠️  AMBIGUITIES DETECTED (Agent 7) — {len(state.ambiguities)} issue(s):")
        for a in state.ambiguities:
            print(f"  - {a}")
    else:
        print("\n✅ AMBIGUITIES (Agent 7): none detected")

    # Critique flags (Agent 9)
    if state.critique.get("flags"):
        print(f"\n🔍 CRITIQUE FLAGS (Agent 9):")
        for f in state.critique["flags"]:
            print(f"  ⚑ {f}")
    else:
        print("\n✅ CRITIQUE (Agent 9): no flags")

    # Final decision (Agent 8)
    fd = state.final_decision
    print(f"\n{'─'*60}")
    print(f"  DECISION   : {fd.get('decision', 'N/A')}")
    print(f"  CONFIDENCE : {fd.get('confidence', 'N/A')}")
    if fd.get("reason"):
        print(f"  REASON     : {fd.get('reason')}")
    if fd.get("scoring_logic"):
        print(f"  SCORING    : {fd.get('scoring_logic')}")
    if fd.get("weights_used"):
        w = fd["weights_used"]
        print(f"  WEIGHTS    : ICD={w.get('icd')} CPT={w.get('cpt')} "
              f"Drug={w.get('drug')} Policy={w.get('policy')}")
    print(sep)

    return state
