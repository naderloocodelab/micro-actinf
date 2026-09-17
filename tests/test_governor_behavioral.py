"""
test_governor_behavioral.py
Automated Behavioral, Gating & Learning Verification for Micro-ActInf Governor.

Tests:
  1. Loop Trap Detection & Breakout:
     Consecutive identical actions in the same state trigger loop detection,
     transition evaluate_action from ALLOW to DENY, and steer prescribed_policy
     to EPISTEMIC_EXPLORE.
  2. Action Safety & Risk Gating:
     Low-risk read actions are ALLOWed.
     High-risk actions (risk=HIGH/CRITICAL) trigger ASK_CONFIRMATION.
  3. Outcome Credit Assignment Learning:
     Successful outcomes reinforce preference C(o) and increase progress index.
     Failed outcomes apply negative reinforcement and reduce confidence.
  4. Full Multi-Turn Governance Cycle:
     End-to-end simulation of 8-turn real-world workflow with full state tracking.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp_server")))
from libactinf import MicroActInfEngine


class TestGovernorBehavioral(unittest.TestCase):

    def setUp(self):
        self.engine = MicroActInfEngine()
        self.engine.reset()

    def test_c_engine_loaded(self):
        """Verify that canonical C11 engine is actively loaded via ctypes."""
        self.assertIsNotNone(self.engine.c_lib, "C11 canonical dynamic library must be loaded")
        state = self.engine.get_state()
        self.assertIn("C11", state["engine_backend"])

    def test_loop_detection_and_deny_breakout(self):
        """
        Verify that 3+ consecutive identical actions in the same dominant state
        trigger loop detection, cause evaluate_action to return DENY, and
        steer prescribe_policy away from the stuck action.
        """
        # Set agent to CODE_GENERATION regime
        self.engine.observe("code_request")
        state0 = self.engine.get_state()
        self.assertEqual(state0["regime_index"], 1, "Must be in CODE_GENERATION")

        # Step 1: Propose code edit -> should be ALLOWed
        v1 = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")
        self.assertEqual(v1["verdict"], "ALLOW")

        # Record outcome of step 1 (same state, same action)
        self.engine.record_outcome("EDIT", "code_request", success=True, progress_delta=0.0)

        # Step 2: Propose same action again
        self.engine.observe("code_request")
        v2 = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")
        self.engine.record_outcome("EDIT", "code_request", success=True, progress_delta=0.0)

        # Step 3: Propose same action a 3rd time (loop condition)
        self.engine.observe("code_request")
        v3 = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")

        # Check loop state
        state3 = self.engine.get_state()
        self.assertTrue(state3["loop_detected"], "Loop must be detected after 3 identical steps")

        # In loop state, repeated stuck action must be DENIED
        v_loop = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")
        self.assertEqual(v_loop["verdict"], "DENY", "Repeated action during loop must be DENIED")
        self.assertFalse(v_loop["allowed"])

        # Prescribed policy must break out to EPISTEMIC_EXPLORE
        pol = self.engine.prescribe_policy()
        self.assertIn("EPISTEMIC_EXPLORE", pol["prescribed_action"])

    def test_action_safety_and_risk_gating(self):
        """
        Verify safety gating:
        - Low risk -> ALLOW
        - High risk / Critical risk -> ASK_CONFIRMATION
        """
        self.engine.observe("code_request")

        # Low risk read tool
        v_read = self.engine.evaluate_action("view_file", "READ", "READ")
        self.assertEqual(v_read["verdict"], "ALLOW")
        self.assertTrue(v_read["allowed"])
        self.assertFalse(v_read["requires_confirmation"])

        # Normal edit tool
        v_edit = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")
        self.assertEqual(v_edit["verdict"], "ALLOW")

        # High risk execute command when confidence threshold is strict (0.98 > 0.95)
        v_high = self.engine.evaluate_action("run_migration", "EXECUTE", "HIGH", confidence_threshold=0.98)
        self.assertEqual(v_high["verdict"], "ASK_CONFIRMATION")
        self.assertTrue(v_high["requires_confirmation"])

        # Critical / Destructive risk tool (ALWAYS requires confirmation regardless of confidence)
        v_crit = self.engine.evaluate_action("drop_all_tables", "EXECUTE", "CRITICAL")
        self.assertEqual(v_crit["verdict"], "ASK_CONFIRMATION")
        self.assertTrue(v_crit["requires_confirmation"])

    def test_outcome_credit_assignment_learning(self):
        """
        Verify outcome-driven credit assignment:
        - Success reinforces prior preference C(o) and increases progress_index.
        - Failure penalizes C(o).
        """
        self.engine.observe("code_request")
        init_progress = self.engine.get_state()["progress_index"]

        # Record a successful compilation
        res_succ = self.engine.record_outcome("EXECUTE", "test_output", success=True, progress_delta=0.25)
        self.assertEqual(res_succ["status"], "LEARNING_APPLIED")
        self.assertGreater(res_succ["progress_index"], init_progress)
        self.assertGreater(res_succ["updated_preference"], 0.0)

        # Record a failure (error_log)
        curr_progress = res_succ["progress_index"]
        res_fail = self.engine.record_outcome("EXECUTE", "error_log", success=False, progress_delta=-0.1)
        self.assertEqual(res_fail["status"], "LEARNING_APPLIED")
        self.assertLess(res_fail["progress_index"], curr_progress)

    def test_full_multiturn_governor_lifecycle(self):
        """
        Simulate an entire 8-step autonomous task cycle:
        1. User asks question -> observe('general_chat') -> prescribe('EPISTEMIC_EXPLORE')
        2. User requests feature -> observe('code_request') -> prescribe('PRAGMATIC_EXECUTE')
        3. Agent proposes edit -> evaluate_action -> ALLOW
        4. Tool returns success -> record_outcome(success=True)
        5. Agent encounters error -> observe('error_log') -> prescribe('AUDIT_DIAGNOSE')
        6. Agent proposes fix -> evaluate_action -> ALLOW
        7. Agent runs tests -> observe('test_output') -> prescribe('CONVERGE_CONCLUDE')
        8. Agent verifies passes -> record_outcome(success=True)
        """
        # Turn 1: General inquiry
        s1 = self.engine.observe("general_chat")
        p1 = self.engine.prescribe_policy()
        self.assertIn("EPISTEMIC_EXPLORE", p1["prescribed_action"])

        # Turn 2: Code request
        s2 = self.engine.observe("code_request")
        p2 = self.engine.prescribe_policy()
        self.assertIn("PRAGMATIC_EXECUTE", p2["prescribed_action"])

        # Turn 3: Evaluate proposed code generation action
        v3 = self.engine.evaluate_action("write_to_file", "EDIT", "EDIT")
        self.assertEqual(v3["verdict"], "ALLOW")

        # Turn 4: Tool execution succeeds
        r4 = self.engine.record_outcome("EDIT", "code_request", success=True, progress_delta=0.2)
        self.assertEqual(r4["status"], "LEARNING_APPLIED")

        # Turn 5: Compiler error occurs
        s5 = self.engine.observe("error_log")
        p5 = self.engine.prescribe_policy()
        self.assertIn("AUDIT_DIAGNOSE", p5["prescribed_action"])

        # Turn 6: Evaluate fix action
        v6 = self.engine.evaluate_action("replace_file_content", "EDIT", "EDIT")
        self.assertEqual(v6["verdict"], "ALLOW")
        r6 = self.engine.record_outcome("EDIT", "code_request", success=True, progress_delta=0.2)

        # Turn 7: Test outputs pass
        s7 = self.engine.observe("test_output")
        p7 = self.engine.prescribe_policy()
        self.assertIn("CONVERGE_CONCLUDE", p7["prescribed_action"])

        # Turn 8: Final outcome recorded
        r8 = self.engine.record_outcome("VERIFY", "test_output", success=True, progress_delta=0.3)
        self.assertGreater(r8["progress_index"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
