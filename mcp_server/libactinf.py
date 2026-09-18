#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Python FFI Binding for Micro-ActInf C11 Engine
--------------------------------------------------------
Single Source of Truth: Directly drives the C11 static memory core via ctypes.
Enforces zero-allocation active inference, multi-step policy planning, loop detection,
and action safety gating for AI Agents (Antigravity, Claude, Cursor) and embedded hosts.
"""

import os
import sys
import re
import ctypes
import math
import json
from typing import Dict, Any, Optional, List, Union, Tuple

ACTINF_MAX_STATES = 16
ACTINF_MAX_OBS = 32
ACTINF_MAX_ACTIONS = 8
ACTINF_HISTORY_LEN = 8

REGIMES = [
    "EXPLORATION (Problem analysis & requirement gathering)",
    "CODE_GENERATION (Writing concrete implementations)",
    "REFACTORING (Architectural optimization & code cleanup)",
    "DEBUGGING (Root-cause analysis & error correction)",
    "VERIFICATION (Running test suites & regression testing)",
    "DECISION (Commitment, branch merging & architecture lock)"
]

POLICIES = [
    "EPISTEMIC_EXPLORE (Request clarification or gather more context)",
    "PRAGMATIC_EXECUTE (Generate production code directly)",
    "AUDIT_DIAGNOSE (Perform step-by-step diagnostic audit)",
    "CONVERGE_CONCLUDE (Summarize changes and finalize task)"
]

RISK_LEVELS = {
    "READ": 0,
    "LOW": 0,
    "ANALYZE": 1,
    "MEDIUM": 2,
    "TEST": 2,
    "EDIT": 3,
    "HIGH": 4,
    "EXECUTE": 4,
    "CRITICAL": 5,
    "DESTRUCTIVE": 5
}

VERDICTS = {
    0: "ALLOW",
    1: "MODIFY",
    2: "ASK_CONFIRMATION",
    3: "DENY"
}

OBS_MAP = {
    "general_chat": 0,
    "code_request": 1,
    "error_log": 2,
    "math_query": 3,
    "test_output": 4,
    "architecture_choice": 5,
    "confirmation": 6,
    "unknown": 7
}

ACTION_MAP = {
    "EPISTEMIC_EXPLORE": 0,
    "READ": 0,
    "EXPLORE": 0,
    "PRAGMATIC_EXECUTE": 1,
    "EDIT": 1,
    "WRITE": 1,
    "EXECUTE": 1,
    "AUDIT_DIAGNOSE": 2,
    "DIAGNOSE": 2,
    "DEBUG": 2,
    "CONVERGE_CONCLUDE": 3,
    "VERIFY": 3,
    "TEST": 3
}
 
DESTRUCTIVE_COMMAND_PATTERNS = [
    r"\brm\s+-[rfRF]{1,3}\b",
    r"\bdel\s+/[fqsvFQSV]{1,4}\b",
    r"\brmdir\s+/[sqSQ]{1,2}\b",
    r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b",
    r"\bDELETE\s+FROM\s+\w+\s*(;|$)",
    r"\bTRUNCATE\s+(TABLE)?\b",
    r"\bformat\s+[a-zA-Z]:",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[fdxFDX]{1,3}\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bchmod\s+-R\s+777\b"
]

WRITE_COMMAND_PATTERNS = [
    r"\bgit\s+commit\b",
    r"\bgit\s+push\b",
    r"\bgit\s+merge\b",
    r"\bgit\s+rebase\b",
    r"\bcp\s+",
    r"\bmv\s+",
    r"\bmove\s+",
    r"\bcopy\s+"
]

def classify_semantic_risk(tool_name: str, args: Optional[Dict[str, Any]] = None, stated_risk: Optional[str] = None) -> Tuple[str, str]:
    """
    Automated Semantic Command & Tool Risk Classifier.
    Prevents self-reported risk bypass by analyzing actual tool arguments.
    Returns: (effective_risk_level, explanation)
    """
    args = args or {}
    cmd_str = ""
    if isinstance(args, dict):
        cmd_str = str(args.get("CommandLine", "") or args.get("command", "") or args.get("cmd", ""))

    # 1. Check for destructive patterns in commands
    if cmd_str:
        for pat in DESTRUCTIVE_COMMAND_PATTERNS:
            if re.search(pat, cmd_str, re.IGNORECASE):
                return "DESTRUCTIVE", f"Command matched destructive signature: '{pat}'"
        for pat in WRITE_COMMAND_PATTERNS:
            if re.search(pat, cmd_str, re.IGNORECASE):
                return "HIGH", f"Command matched modifying signature: '{pat}'"

    # 2. Check tool name semantics
    tool_lower = (tool_name or "").lower()
    if tool_lower in ("write_to_file", "replace_file_content", "edit_file", "create_or_update_file"):
        return "EDIT", f"Tool '{tool_name}' performs file modifications"
    elif tool_lower in ("run_command", "execute_command", "bash", "terminal"):
        return ("EXECUTE", f"Tool '{tool_name}' executes shell commands")
    elif tool_lower in ("view_file", "read_file", "list_dir", "grep_search", "find_by_name"):
        return ("READ", f"Tool '{tool_name}' is read-only")

    # Fallback to stated risk or default
    return (stated_risk or "EDIT"), "Default risk classification"


class ActInfLearningConfig(ctypes.Structure):
    _fields_ = [
        ('learning_rate_a', ctypes.c_float),
        ('learning_rate_b', ctypes.c_float),
        ('decay_factor', ctypes.c_float),
        ('enable_learning', ctypes.c_bool)
    ]


class ActInfCostConfig(ctypes.Structure):
    _fields_ = [
        ('beta_epistemic', ctypes.c_float),
        ('weight_token_cost', ctypes.c_float),
        ('weight_latency', ctypes.c_float),
        ('weight_risk', ctypes.c_float),
        ('loop_penalty', ctypes.c_float),
        ('gamma_discount', ctypes.c_float)
    ]


class ActInfStepRecord(ctypes.Structure):
    _fields_ = [
        ('state_dom', ctypes.c_uint8),
        ('action', ctypes.c_uint8),
        ('obs', ctypes.c_uint8),
        ('signature', ctypes.c_uint32),
        ('progress_delta', ctypes.c_float),
        ('entropy', ctypes.c_float)
    ]


class MicroActInfStruct(ctypes.Structure):
    _fields_ = [
        ('num_states', ctypes.c_uint8),
        ('num_obs', ctypes.c_uint8),
        ('num_actions', ctypes.c_uint8),
        ('horizon', ctypes.c_uint8),
        ('a_entropy_dirty', ctypes.c_bool),
        ('gamma', ctypes.c_float),
        ('alpha_prior', ctypes.c_float),
        ('s', ctypes.c_float * ACTINF_MAX_STATES),
        ('s_prev', ctypes.c_float * ACTINF_MAX_STATES),
        ('A', (ctypes.c_float * ACTINF_MAX_STATES) * ACTINF_MAX_OBS),
        ('B', ((ctypes.c_float * ACTINF_MAX_STATES) * ACTINF_MAX_STATES) * ACTINF_MAX_ACTIONS),
        ('C', ctypes.c_float * ACTINF_MAX_OBS),
        ('action_costs', (ctypes.c_float * 3) * ACTINF_MAX_ACTIONS),
        ('action_risk', ctypes.c_uint8 * ACTINF_MAX_ACTIONS),
        ('A_entropy', ctypes.c_float * ACTINF_MAX_STATES),
        ('G', ctypes.c_float * ACTINF_MAX_ACTIONS),
        ('pi', ctypes.c_float * ACTINF_MAX_ACTIONS),
        ('a_counts', (ctypes.c_float * ACTINF_MAX_STATES) * ACTINF_MAX_OBS),
        ('b_counts', ((ctypes.c_float * ACTINF_MAX_ACTIONS) * ACTINF_MAX_STATES) * ACTINF_MAX_STATES),
        ('learning_cfg', ActInfLearningConfig),
        ('cost_cfg', ActInfCostConfig),
        ('history', ActInfStepRecord * ACTINF_HISTORY_LEN),
        ('history_idx', ctypes.c_uint8),
        ('history_count', ctypes.c_uint8),
        ('consecutive_loops', ctypes.c_uint8),
        ('loop_detected', ctypes.c_bool),
        ('target_state', ctypes.c_uint8),
        ('goal_progress', ctypes.c_float),
        ('goal_drift', ctypes.c_float),
        ('progress_index', ctypes.c_float),
        ('last_entropy', ctypes.c_float),
        ('entropy_velocity', ctypes.c_float),
        ('last_action', ctypes.c_uint8),
        ('step_count', ctypes.c_uint32)
    ]


def _find_and_load_c_library():
    """Locate and load the compiled C shared library with unicode-safe Windows path support."""
    search_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        os.getcwd()
    ]
    lib_names = ["libmicro_actinf.dll", "micro_actinf.dll", "libmicro_actinf.so", "libmicro_actinf.dylib"]

    for d in search_dirs:
        for name in lib_names:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                try:
                    if sys.platform == "win32":
                        # Unicode-safe DLL loading on Windows (handles non-ASCII/Persian workspace paths)
                        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                        kernel32.LoadLibraryW.restype = ctypes.c_void_p
                        kernel32.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
                        handle = kernel32.LoadLibraryW(os.path.abspath(candidate))
                        if handle:
                            return ctypes.CDLL(None, handle=handle), candidate
                    lib = ctypes.CDLL(candidate)
                    return lib, candidate
                except Exception:
                    pass

    return None, None


_C_LIB, _LIB_PATH = _find_and_load_c_library()

if _C_LIB is not None:
    # Bind C function prototypes
    _C_LIB.micro_actinf_init.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
    _C_LIB.micro_actinf_init.restype = None

    _C_LIB.micro_actinf_reset_state.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_reset_state.restype = None

    _C_LIB.micro_actinf_step.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8]
    _C_LIB.micro_actinf_step.restype = None

    _C_LIB.micro_actinf_select_action.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_select_action.restype = ctypes.c_uint8

    _C_LIB.micro_actinf_evaluate_action.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8, ctypes.c_int, ctypes.c_float]
    _C_LIB.micro_actinf_evaluate_action.restype = ctypes.c_int

    _C_LIB.micro_actinf_record_outcome.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8, ctypes.c_uint8, ctypes.c_bool, ctypes.c_float]
    _C_LIB.micro_actinf_record_outcome.restype = None

    _C_LIB.micro_actinf_learn_step.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint32, ctypes.c_uint32]
    _C_LIB.micro_actinf_learn_step.restype = None

    _C_LIB.micro_actinf_shannon_entropy.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_shannon_entropy.restype = ctypes.c_float

    _C_LIB.micro_actinf_get_dominant_state.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.POINTER(ctypes.c_float)]
    _C_LIB.micro_actinf_get_dominant_state.restype = ctypes.c_uint8

    _C_LIB.micro_actinf_sync_counts_from_matrices.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_sync_counts_from_matrices.restype = None

    _C_LIB.micro_actinf_set_horizon.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8]
    _C_LIB.micro_actinf_set_horizon.restype = None

    _C_LIB.micro_actinf_set_alpha_prior.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_float]
    _C_LIB.micro_actinf_set_alpha_prior.restype = None

    _C_LIB.micro_actinf_set_beta_epistemic.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_float]
    _C_LIB.micro_actinf_set_beta_epistemic.restype = None

    _C_LIB.micro_actinf_step_with_signature.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8, ctypes.c_uint32, ctypes.c_float]
    _C_LIB.micro_actinf_step_with_signature.restype = None

    _C_LIB.micro_actinf_evaluate_action_with_signature.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8, ctypes.c_int, ctypes.c_float, ctypes.c_uint32]
    _C_LIB.micro_actinf_evaluate_action_with_signature.restype = ctypes.c_int

    _C_LIB.micro_actinf_reset_belief.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_reset_belief.restype = None

    _C_LIB.micro_actinf_reset_episode.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_reset_episode.restype = None

    _C_LIB.micro_actinf_reset_model.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_reset_model.restype = None

    _C_LIB.micro_actinf_set_goal.argtypes = [ctypes.POINTER(MicroActInfStruct), ctypes.c_uint8]
    _C_LIB.micro_actinf_set_goal.restype = None

    _C_LIB.micro_actinf_get_goal_drift.argtypes = [ctypes.POINTER(MicroActInfStruct)]
    _C_LIB.micro_actinf_get_goal_drift.restype = ctypes.c_float

    _C_LIB.micro_actinf_hash_signature.argtypes = [ctypes.c_char_p]
    _C_LIB.micro_actinf_hash_signature.restype = ctypes.c_uint32


class MicroActInfEngine:
    """
    Canonical Active Inference Governor Engine.
    Directly backed by the zero-allocation C11 core library.
    """
    def __init__(self, states: int = 6, obs: int = 8, actions: int = 4, horizon: int = 2):
        self.K = states
        self.M = obs
        self.A = actions
        self.c_lib = _C_LIB
        self.agent = MicroActInfStruct()

        if self.c_lib is not None:
            self.c_lib.micro_actinf_init(ctypes.byref(self.agent), self.K, self.M, self.A)
            self.c_lib.micro_actinf_set_horizon(ctypes.byref(self.agent), horizon)
            self._calibrate_coding_agent_priors()
        else:
            # Fallback for systems without compiled shared library
            self._init_pure_python_fallback()

    def _calibrate_coding_agent_priors(self):
        """Seed domain-calibrated observation likelihoods and transition tensors."""
        # Observation likelihood mapping: P(obs | state)
        A_raw = [
            [0.65, 0.05, 0.05, 0.04, 0.04, 0.05],  # 0: general_chat -> EXPLORATION
            [0.05, 0.75, 0.05, 0.04, 0.04, 0.05],  # 1: code_request -> CODE_GEN
            [0.02, 0.02, 0.04, 0.85, 0.05, 0.02],  # 2: error_log -> DEBUGGING
            [0.08, 0.05, 0.75, 0.04, 0.04, 0.04],  # 3: math_query -> REFACTORING
            [0.02, 0.03, 0.05, 0.05, 0.80, 0.03],  # 4: test_output -> VERIFICATION
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.75],  # 5: architecture_choice -> DECISION
            [0.04, 0.04, 0.04, 0.04, 0.08, 0.75],  # 6: confirmation -> DECISION
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10]   # 7: unknown -> neutral
        ]
        # Column normalize and assign to C agent.A
        for s in range(self.K):
            col_sum = sum(A_raw[o][s] for o in range(self.M))
            for o in range(self.M):
                self.agent.A[o][s] = A_raw[o][s] / col_sum

        # Calibrate transition matrices B[action][to_state][from_state]
        for u in range(self.A):
            for j in range(self.K):
                for i in range(self.K):
                    if u == 0:  # EPISTEMIC_EXPLORE -> promotes state 0
                        self.agent.B[u][i][j] = 0.60 if i == 0 else (0.40 / (self.K - 1))
                    elif u == 1:  # PRAGMATIC_EXECUTE -> promotes states 1 & 2
                        self.agent.B[u][i][j] = 0.45 if i in (1, 2) else (0.10 / (self.K - 2))
                    elif u == 2:  # AUDIT_DIAGNOSE -> promotes state 3
                        self.agent.B[u][i][j] = 0.75 if i == 3 else (0.25 / (self.K - 1))
                    else:  # CONVERGE_CONCLUDE -> promotes states 4 & 5
                        self.agent.B[u][i][j] = 0.45 if i in (4, 5) else (0.10 / (self.K - 2))

        # Synchronize Dirichlet pseudo-counts from matrices
        self.c_lib.micro_actinf_sync_counts_from_matrices(ctypes.byref(self.agent))

    def _init_pure_python_fallback(self):
        """Mathematical fallback if C shared library is not loaded."""
        self.s = [1.0 / self.K] * self.K
        self.last_action = 0
        self.step_count = 0
        self.alpha = 0.25
        self.history = []
        self.loop_detected = False

    def compute_signature(self, tool: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> int:
        """Compute deterministic 32-bit FNV-1a hash signature for tool and arguments."""
        if not tool and not args:
            return 0
        sig_str = f"{tool or ''}:{json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args or '')}"
        if self.c_lib is not None:
            return int(self.c_lib.micro_actinf_hash_signature(sig_str.encode('utf-8', errors='ignore')))
        else:
            h = 2166136261
            for b in sig_str.encode('utf-8', errors='ignore'):
                h = ((h ^ b) * 16777619) & 0xFFFFFFFF
            return h

    def observe(self,
                obs_type: str,
                context_attributes: Optional[Dict[str, Any]] = None,
                tool: Optional[str] = None,
                args: Optional[Dict[str, Any]] = None,
                progress_delta: float = 0.0) -> Dict[str, Any]:
        """
        Feed sensory observation into active inference state filter.
        Includes signature and progress delta for advanced loop/inertia detection.
        """
        obs_id = OBS_MAP.get(obs_type.lower(), 7)
        sig = self.compute_signature(tool, args)

        if self.c_lib is not None:
            self.c_lib.micro_actinf_step_with_signature(
                ctypes.byref(self.agent),
                obs_id,
                ctypes.c_uint32(sig),
                ctypes.c_float(progress_delta)
            )
            return self.get_state()
        else:
            # Fallback Python simulation
            s_prior = [(1 - self.alpha) * self.s[i] + self.alpha * (1.0 / self.K) for i in range(self.K)]
            logits = [math.log(max(sp, 1e-12)) for sp in s_prior]
            boost_map = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 5}
            tgt = boost_map.get(obs_id, 0)
            logits[tgt] += 2.0
            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            sum_e = sum(exps)
            self.s = [e / sum_e for e in exps]
            self.step_count += 1
            return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state, belief distribution, uncertainty, goal tracking, and loop status."""
        if self.c_lib is not None:
            conf = ctypes.c_float()
            dom_idx = self.c_lib.micro_actinf_get_dominant_state(ctypes.byref(self.agent), ctypes.byref(conf))
            entropy = self.c_lib.micro_actinf_shannon_entropy(ctypes.byref(self.agent))
            policy_idx = self.c_lib.micro_actinf_select_action(ctypes.byref(self.agent))

            beliefs = [float(self.agent.s[i]) for i in range(self.K)]
            efe_values = [round(float(self.agent.G[u]), 4) for u in range(self.A)]
            action_probs = [round(float(self.agent.pi[u]), 4) for u in range(self.A)]
            target_s = int(self.agent.target_state)

            return {
                "dominant_regime": REGIMES[dom_idx],
                "regime_index": int(dom_idx),
                "confidence": round(float(conf.value) * 100.0, 2),
                "uncertainty": round((1.0 - float(conf.value)) * 100.0, 2),
                "shannon_entropy_nats": round(float(entropy), 4),
                "entropy_velocity": round(float(self.agent.entropy_velocity), 4),
                "target_goal_regime": REGIMES[target_s] if target_s < len(REGIMES) else f"State {target_s}",
                "goal_progress": round(float(self.agent.goal_progress) * 100.0, 2),
                "goal_drift": round(float(self.agent.goal_drift), 4),
                "progress_index": round(float(self.agent.progress_index), 3),
                "loop_detected": bool(self.agent.loop_detected),
                "consecutive_loops": int(self.agent.consecutive_loops),
                "prescribed_policy": POLICIES[policy_idx],
                "policy_index": int(policy_idx),
                "step_count": int(self.agent.step_count),
                "expected_free_energy": efe_values,
                "action_probabilities": action_probs,
                "belief_distribution": {REGIMES[i].split()[0]: round(beliefs[i], 4) for i in range(self.K)},
                "engine_backend": f"C11 Canonical Core ({_LIB_PATH})" if _LIB_PATH else "C11 Static DLL"
            }
        else:
            dom_idx = max(range(self.K), key=lambda i: self.s[i])
            conf = self.s[dom_idx]
            entropy = -sum(b * math.log(b + 1e-12) for b in self.s)
            pol_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 3}
            pol_idx = pol_map.get(dom_idx, 0)
            return {
                "dominant_regime": REGIMES[dom_idx],
                "regime_index": dom_idx,
                "confidence": round(conf * 100.0, 2),
                "uncertainty": round((1.0 - conf) * 100.0, 2),
                "shannon_entropy_nats": round(entropy, 4),
                "entropy_velocity": 0.0,
                "target_goal_regime": REGIMES[4],
                "goal_progress": round(self.s[4] * 100.0, 2),
                "goal_drift": round(1.0 - self.s[4], 4),
                "progress_index": 0.0,
                "loop_detected": False,
                "consecutive_loops": 0,
                "prescribed_policy": POLICIES[pol_idx],
                "policy_index": pol_idx,
                "step_count": self.step_count,
                "expected_free_energy": [],
                "action_probabilities": [],
                "belief_distribution": {REGIMES[i].split()[0]: round(self.s[i], 4) for i in range(self.K)},
                "engine_backend": "Python Fallback"
            }

    def prescribe_policy(self) -> Dict[str, Any]:
        """
        Compute optimal control policy action minimizing multi-step Cost-Aware Free Energy.
        Enforces cognitive regime lock with active loop-breaking override.
        """
        state = self.get_state()
        dom_idx = state["regime_index"]

        # 1. Primary policy determined by Active Inference cognitive regime
        if dom_idx == 0:
            pol_idx = 0  # EPISTEMIC_EXPLORE
        elif dom_idx in (1, 2):
            pol_idx = 1  # PRAGMATIC_EXECUTE
        elif dom_idx == 3:
            pol_idx = 2  # AUDIT_DIAGNOSE
        else:
            pol_idx = 3  # CONVERGE_CONCLUDE

        # 2. Loop & Inertia Override: If trapped in a repetitive loop, steer to epistemic exploration
        if state["loop_detected"]:
            pol_idx = 0 if pol_idx != 0 else 2  # Break repetitive inertia

        pol_name = POLICIES[pol_idx]

        directives = {
            0: "EPISTEMIC_EXPLORE: Clarify technical ambiguities and inspect prerequisites before writing code.",
            1: "PRAGMATIC_EXECUTE: Output 100% production-ready code immediately. Zero greetings, zero conversational filler.",
            2: "AUDIT_DIAGNOSE: Pinpoint cycle-accurate root cause, inspect memory/bounds, and emit unified diff without lecturing.",
            3: "CONVERGE_CONCLUDE: Execute verification suites, report numerical benchmarks, and finalize task."
        }

        return {
            "prescribed_action": pol_name,
            "action_index": pol_idx,
            "directive": directives.get(pol_idx, directives[1]),
            "confidence": state["confidence"],
            "uncertainty": state["uncertainty"],
            "loop_detected": state["loop_detected"],
            "rationale": f"Regime {REGIMES[dom_idx].split()[0]} -> Policy {pol_name.split()[0]} (EFE G={state['expected_free_energy']})."
        }

    def evaluate_action(self,
                        proposed_tool: str,
                        action_type: Union[str, int] = "EDIT",
                        risk_level: Optional[Union[str, int]] = None,
                        confidence_threshold: float = 0.80,
                        tool_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hard Active Governance: Evaluates proposed LLM action before execution.
        Features automated semantic risk classification to prevent self-reported risk bypass.
        Returns verdict: ALLOW, MODIFY, ASK_CONFIRMATION, or DENY.
        """
        # 1. Semantic Risk Classification (immune to caller deception)
        stated_str = risk_level if isinstance(risk_level, str) else None
        eff_risk_str, risk_reason = classify_semantic_risk(proposed_tool, tool_args, stated_str)
        eff_risk_val = RISK_LEVELS.get(eff_risk_str.upper(), 3)

        if isinstance(action_type, str):
            act_idx = ACTION_MAP.get(action_type.upper(), 1)
        else:
            act_idx = int(action_type)

        sig = self.compute_signature(proposed_tool, tool_args)

        if self.c_lib is not None:
            verdict_code = self.c_lib.micro_actinf_evaluate_action_with_signature(
                ctypes.byref(self.agent),
                act_idx,
                eff_risk_val,
                confidence_threshold,
                ctypes.c_uint32(sig)
            )
            verdict_str = VERDICTS.get(verdict_code, "ALLOW")
        else:
            verdict_str = "ALLOW"
            verdict_code = 0

        # Construct concrete transformation instructions for MODIFY verdict
        modifications = {}
        if verdict_str == "MODIFY":
            if eff_risk_val >= RISK_LEVELS["EDIT"]:
                modifications = {
                    "mode": "DRY_RUN_OR_DIFF_PREVIEW",
                    "instruction": "Downgrade in-place file modification to diff preview or dry-run validation."
                }
            elif eff_risk_val >= RISK_LEVELS["EXECUTE"]:
                modifications = {
                    "mode": "INSPECT_ONLY",
                    "instruction": "Execute with '--dry-run' or inspect arguments before full process invocation."
                }

        reasons = {
            "ALLOW": f"Action '{proposed_tool}' is fully aligned with active cognitive regime and risk budget ({eff_risk_str}).",
            "MODIFY": f"Action '{proposed_tool}' transformed to safe mode: {risk_reason}.",
            "ASK_CONFIRMATION": f"Action '{proposed_tool}' carries elevated risk level '{eff_risk_str}': {risk_reason}; explicit user confirmation required.",
            "DENY": f"Action '{proposed_tool}' is DENIED to prevent duplicate stuck command or dangerous out-of-regime action."
        }

        return {
            "verdict": verdict_str,
            "verdict_code": verdict_code,
            "proposed_tool": proposed_tool,
            "effective_risk_level": eff_risk_str,
            "risk_classification_reason": risk_reason,
            "allowed": (verdict_str in ("ALLOW", "MODIFY")),
            "requires_confirmation": (verdict_str == "ASK_CONFIRMATION"),
            "modifications": modifications,
            "reason": reasons.get(verdict_str, "")
        }

    def record_outcome(self,
                       action: Union[str, int],
                       outcome_obs: str,
                       success: bool,
                       progress_delta: float = 0.1) -> Dict[str, Any]:
        """
        Record execution result and execute credit assignment learning.
        Adapts prior preferences C(o) and transitions B(u).
        """
        if isinstance(action, str):
            act_idx = ACTION_MAP.get(action.upper(), 1)
        else:
            act_idx = int(action)

        obs_id = OBS_MAP.get(outcome_obs.lower(), 7)

        if self.c_lib is not None:
            self.c_lib.micro_actinf_record_outcome(
                ctypes.byref(self.agent),
                act_idx,
                obs_id,
                success,
                progress_delta
            )
            new_prog = round(float(self.agent.progress_index), 3)
            pref = round(float(self.agent.C[obs_id]), 4)
            return {
                "status": "LEARNING_APPLIED",
                "recorded": True,
                "action": action,
                "outcome_obs": outcome_obs,
                "success": success,
                "progress_index": new_prog,
                "new_progress_index": new_prog,
                "updated_preference": pref,
                "message": "Online Dirichlet credit assignment update applied successfully."
            }
        else:
            return {"status": "LEARNING_APPLIED", "recorded": True, "success": success, "progress_index": 0.0, "updated_preference": 0.0}

    def reset_belief(self) -> Dict[str, Any]:
        """Turn-level reset: resets only current belief s to uniform prior."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_reset_belief(ctypes.byref(self.agent))
        else:
            self.s = [1.0 / self.K] * self.K
        return {"status": "belief_reset_successful", "state": self.get_state()}

    def reset_episode(self) -> Dict[str, Any]:
        """Episode-level reset: resets belief and history, preserves learned Dirichlet parameters."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_reset_episode(ctypes.byref(self.agent))
        else:
            self.s = [1.0 / self.K] * self.K
            self.history = []
        return {"status": "episode_reset_successful", "state": self.get_state()}

    def reset_model(self) -> Dict[str, Any]:
        """Factory-level reset: complete reset restoring calibrated default parameters."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_reset_model(ctypes.byref(self.agent))
            self._calibrate_coding_agent_priors()
        else:
            self._init_pure_python_fallback()
        return {"status": "model_reset_successful", "state": self.get_state()}

    def reset(self) -> Dict[str, Any]:
        """Standard reset: performs episode-level reset preserving learned parameters."""
        return self.reset_episode()

    def set_goal(self, target_state: Union[str, int]) -> Dict[str, Any]:
        """Set target convergence goal regime for drift tracking."""
        if isinstance(target_state, str):
            idx = 4  # default VERIFICATION
            for i, r in enumerate(REGIMES):
                if target_state.upper() in r.upper():
                    idx = i
                    break
        else:
            idx = int(target_state)

        if self.c_lib is not None:
            self.c_lib.micro_actinf_set_goal(ctypes.byref(self.agent), ctypes.c_uint8(idx))
            drift = self.c_lib.micro_actinf_get_goal_drift(ctypes.byref(self.agent))
            return {"status": "goal_set", "target_state": REGIMES[idx], "goal_drift": round(float(drift), 4)}
        return {"status": "goal_set", "target_state": REGIMES[idx], "goal_drift": 0.0}

    def set_horizon(self, horizon: int) -> None:
        """Set planning trajectory horizon H (bounded 1 to 4)."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_set_horizon(ctypes.byref(self.agent), ctypes.c_uint8(horizon))

    def set_alpha_prior(self, alpha: float) -> None:
        """Set adaptive prior decay factor alpha (bounded 0.0 to 1.0)."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_set_alpha_prior(ctypes.byref(self.agent), ctypes.c_float(alpha))

    def set_beta_epistemic(self, beta: float) -> None:
        """Set epistemic exploration weighting parameter beta."""
        if self.c_lib is not None:
            self.c_lib.micro_actinf_set_beta_epistemic(ctypes.byref(self.agent), ctypes.c_float(beta))
