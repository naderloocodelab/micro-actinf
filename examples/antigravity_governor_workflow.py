#!/usr/bin/env python3
"""
antigravity_governor_workflow.py
================================
Production Demonstration of Google Antigravity Hard Cognitive Governance.

Demonstrates the Complete Two-Phase Closed-Loop Lifecycle:
  PHASE 1: Cognitive State & Policy Prescription
    1. Agent receives user prompt -> actinf_observe(obs_type, context)
    2. Model inquires prescribed policy -> actinf_prescribe_policy()
    3. Model locks output mode to prescribed policy (e.g. PRAGMATIC_EXECUTE, AUDIT_DIAGNOSE)

  PHASE 2: Action Safety Gating & Outcome Feedback
    4. Agent formulates tool call -> actinf_evaluate_action(tool, action_type, risk)
    5. Governor returns verdict:
       - ALLOW: Execute tool immediately
       - MODIFY: Downgrade tool (e.g. read-only)
       - ASK_CONFIRMATION: Interrupt and query human operator
       - DENY: Block action (e.g. infinite loop, out-of-regime)
    6. Tool completes -> actinf_record_outcome(action, outcome_obs, success, progress_delta)
    7. Governor executes online Dirichlet learning & preference credit assignment.
"""

import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_server")))
from libactinf import MicroActInfEngine


def simulate_governed_turn(engine: MicroActInfEngine,
                           turn_id: int,
                           user_message: str,
                           obs_type: str,
                           proposed_tool: str,
                           action_type: str,
                           risk_level: str,
                           confidence_threshold: float = 0.80,
                           simulated_tool_success: bool = True):
    print("\n" + "=" * 80)
    print(f"[TURN {turn_id}] User Input: \"{user_message}\"")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Pre-Execution Observation
    # -------------------------------------------------------------
    obs_res = engine.observe(obs_type)
    print(f"1. [MCP actinf_observe] Obs Type: {obs_type}")
    print(f"   --> Dominant Regime: {obs_res['dominant_regime']}")
    print(f"   --> Confidence: {obs_res['confidence']}% | Shannon Entropy: {obs_res['shannon_entropy_nats']} nats")

    # -------------------------------------------------------------
    # Step 2: Policy Prescription Lock
    # -------------------------------------------------------------
    policy = engine.prescribe_policy()
    print(f"2. [MCP actinf_prescribe_policy]")
    print(f"   --> Prescribed Policy: {policy['prescribed_action']}")
    print(f"   --> Mandatory Directive: {policy['directive']}")

    # -------------------------------------------------------------
    # Step 3: Action Safety Gating (Governor Interception)
    # -------------------------------------------------------------
    print(f"3. [Agent Proposes Action]")
    print(f"   --> Tool: {proposed_tool} | Action Type: {action_type} | Risk Level: {risk_level}")

    verdict = engine.evaluate_action(
        proposed_tool=proposed_tool,
        action_type=action_type,
        risk_level=risk_level,
        confidence_threshold=confidence_threshold
    )
    print(f"4. [MCP actinf_evaluate_action]")
    print(f"   --> Verdict: {verdict['verdict']} (Code {verdict['verdict_code']})")
    print(f"   --> Reason: {verdict['reason']}")

    if verdict["verdict"] == "DENY":
        print("   [GOVERNOR INTERVENTION] Action BLOCKED to prevent loop or risk violation.")
        return
    elif verdict["verdict"] == "ASK_CONFIRMATION":
        print("   [HUMAN-IN-THE-LOOP] Execution paused. Human operator confirms action [Y].")

    # -------------------------------------------------------------
    # Step 4: Tool Execution & Feedback Loop
    # -------------------------------------------------------------
    print(f"5. [Tool Execution Executed] -> Success: {simulated_tool_success}")

    outcome = engine.record_outcome(
        action=action_type,
        outcome_obs=obs_type,
        success=simulated_tool_success,
        progress_delta=0.25 if simulated_tool_success else -0.10
    )
    print(f"6. [MCP actinf_record_outcome & Learning]")
    print(f"   --> Progress Index: {outcome['progress_index']}")
    print(f"   --> Dirichlet Prior Updated: {outcome['updated_preference']}")


def main():
    print("================================================================================")
    print("MICRO-ACTINF: GOOGLE ANTIGRAVITY HARD COGNITIVE GOVERNOR RUNTIME")
    print("================================================================================")

    engine = MicroActInfEngine()
    engine.reset()

    # Scenario 1: User asks for a lock-free queue in C
    simulate_governed_turn(
        engine=engine,
        turn_id=1,
        user_message="یک بافر حلقوی بدون قفل (Lock-free Ring Buffer) در C11 با حافظه استاتیک بنویس.",
        obs_type="code_request",
        proposed_tool="write_to_file",
        action_type="EDIT",
        risk_level="EDIT",
        confidence_threshold=0.80,
        simulated_tool_success=True
    )

    # Scenario 2: GCC compiler error occurs during compilation
    simulate_governed_turn(
        engine=engine,
        turn_id=2,
        user_message="خطای کامپایلر: undefined reference to atomic_fetch_add in line 42.",
        obs_type="error_log",
        proposed_tool="replace_file_content",
        action_type="EDIT",
        risk_level="EDIT",
        confidence_threshold=0.80,
        simulated_tool_success=True
    )

    # Scenario 3: High-risk migration / destructive database drop
    simulate_governed_turn(
        engine=engine,
        turn_id=3,
        user_message="دیتابیس تستی رو پاک کن و مایگریشن جدید رو اعمال کن.",
        obs_type="code_request",
        proposed_tool="run_command(drop_tables.sh)",
        action_type="EXECUTE",
        risk_level="CRITICAL",
        confidence_threshold=0.90,
        simulated_tool_success=True
    )

    # Scenario 4: Repetitive stuck loop simulation (Agent tries same edit 3 times with 0 progress)
    print("\n" + "#" * 80)
    print("# SIMULATING LOOP DETECTION & AUTOMATED BREAKOUT")
    print("#" * 80)
    for rep in range(1, 4):
        simulate_governed_turn(
            engine=engine,
            turn_id=4 + rep,
            user_message="همین خط کد رو دوباره چک کن و مجدداً بازنویسی کن.",
            obs_type="code_request",
            proposed_tool="replace_file_content",
            action_type="EDIT",
            risk_level="EDIT",
            confidence_threshold=0.80,
            simulated_tool_success=False
        )

    # Final Summary State
    state = engine.get_state()
    print("\n" + "=" * 80)
    print("FINAL COGNITIVE GOVERNOR STATE")
    print("=" * 80)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
