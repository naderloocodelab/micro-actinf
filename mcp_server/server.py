#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Context Protocol (MCP) Server for Micro-ActInf
----------------------------------------------------
Provides Active Inference State Tracking & Policy Prescription as an MCP service
for AI Agents (Antigravity, Claude Desktop, Cursor, Ollama).

Transport: Standard I/O (JSON-RPC 2.0 stdio via FastMCP)
"""

import math
import json
import sys

# Cognitive Regimes for Autonomous Coding Agents
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


class ActiveInferenceState:
    def __init__(self):
        self.K = 6  # States
        self.M = 8  # Observations
        self.A = 4  # Actions
        self.beliefs = [1.0 / self.K] * self.K
        self.last_action = 0
        self.step_count = 0
        self.alpha = 0.25  # Forgetting/recency mixing factor (prevents Bayesian lock-in)

        self.obs_map = {
            "general_chat": 0,
            "code_request": 1,
            "error_log": 2,
            "math_query": 3,
            "test_output": 4,
            "architecture_choice": 5,
            "confirmation": 6,
            "unknown": 7
        }

        # Raw likelihood table A_raw[obs][state]
        A_raw = [
            [0.65, 0.05, 0.08, 0.04, 0.04, 0.05],  # 0: general_chat
            [0.05, 0.75, 0.10, 0.04, 0.04, 0.05],  # 1: code_request
            [0.02, 0.02, 0.04, 0.85, 0.05, 0.02],  # 2: error_log
            [0.25, 0.10, 0.50, 0.05, 0.05, 0.05],  # 3: math_query
            [0.02, 0.03, 0.05, 0.05, 0.80, 0.03],  # 4: test_output
            [0.05, 0.05, 0.25, 0.05, 0.05, 0.60],  # 5: architecture_choice
            [0.04, 0.04, 0.04, 0.04, 0.08, 0.75],  # 6: confirmation
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10]   # 7: unknown
        ]

        # Column-normalize A matrix (P(o | s))
        self.A_mat = [[0.0] * self.K for _ in range(self.M)]
        for s in range(self.K):
            col_sum = sum(A_raw[o][s] for o in range(self.M))
            for o in range(self.M):
                self.A_mat[o][s] = A_raw[o][s] / col_sum

        # Transition matrix B[action][to_state][from_state]
        self.B = [[[0.0] * self.K for _ in range(self.K)] for _ in range(self.A)]

        # u=0: EPISTEMIC_EXPLORE promotes exploration (0)
        for j in range(self.K):
            self.B[0][0][j] = 0.60
            for i in range(1, self.K):
                self.B[0][i][j] = 0.40 / (self.K - 1)

        # u=1: PRAGMATIC_EXECUTE promotes code generation (1) & refactoring (2)
        for j in range(self.K):
            self.B[1][1][j] = 0.70
            for i in range(self.K):
                if i != 1:
                    self.B[1][i][j] = 0.30 / (self.K - 1)

        # u=2: AUDIT_DIAGNOSE promotes debugging (3)
        for j in range(self.K):
            self.B[2][3][j] = 0.75
            for i in range(self.K):
                if i != 3:
                    self.B[2][i][j] = 0.25 / (self.K - 1)

        # u=3: CONVERGE_CONCLUDE promotes verification (4) & decision (5)
        for j in range(self.K):
            self.B[3][4][j] = 0.45
            self.B[3][5][j] = 0.45
            for i in range(self.K):
                if i not in (4, 5):
                    self.B[3][i][j] = 0.10 / (self.K - 2)

    def observe(self, obs_type: str) -> dict:
        obs_id = self.obs_map.get(obs_type.lower(), 7)

        # 1. Prior prediction via Markov transition tensor B(u_{t-1}) and decay factor
        s_prior = [0.0] * self.K
        for i in range(self.K):
            val = sum(self.B[self.last_action][i][j] * self.beliefs[j] for j in range(self.K))
            s_prior[i] = (1.0 - self.alpha) * val + self.alpha * (1.0 / self.K)

        # 2. Bayesian likelihood update: s_{t} = (A_{o_t, :} * s_prior) / norm
        unnorm = [self.A_mat[obs_id][i] * s_prior[i] for i in range(self.K)]
        total = sum(unnorm)
        if total > 1e-12:
            self.beliefs = [p / total for p in unnorm]
        else:
            self.beliefs = [1.0 / self.K] * self.K

        self.step_count += 1
        return self.get_state()

    def get_state(self) -> dict:
        dominant_idx = max(range(self.K), key=lambda i: self.beliefs[i])
        entropy = -sum(b * math.log(b + 1e-12) for b in self.beliefs)
        return {
            "dominant_regime": REGIMES[dominant_idx],
            "regime_index": dominant_idx,
            "confidence": round(self.beliefs[dominant_idx] * 100, 2),
            "shannon_entropy_nats": round(entropy, 4),
            "step_count": self.step_count,
            "belief_distribution": {REGIMES[i].split()[0]: round(self.beliefs[i], 4) for i in range(self.K)}
        }

    def prescribe_policy(self) -> dict:
        state = self.get_state()
        dom = state["regime_index"]
        if dom == 0:
            action_idx = 0  # EPISTEMIC_EXPLORE
        elif dom in (1, 2):
            action_idx = 1  # PRAGMATIC_EXECUTE
        elif dom == 3:
            action_idx = 2  # AUDIT_DIAGNOSE
        else:
            action_idx = 3  # CONVERGE_CONCLUDE

        self.last_action = action_idx
        return {
            "prescribed_action": POLICIES[action_idx],
            "action_index": action_idx,
            "directive": f"Directly focus on {POLICIES[action_idx].split()[0]}. Avoid drifting into unrelated discussions."
        }


# Instantiate shared state filter
state_filter = ActiveInferenceState()

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("micro-actinf")

    @mcp.tool()
    def actinf_observe(obs_type: str) -> str:
        """Feed an incoming user/system observation into the Active Inference POMDP filter.
        Parameters:
            obs_type: One of 'general_chat', 'code_request', 'error_log', 'math_query', 'test_output', 'architecture_choice', 'confirmation', 'unknown'
        """
        return json.dumps(state_filter.observe(obs_type), indent=2)

    @mcp.tool()
    def actinf_get_state() -> str:
        """Get current cognitive regime, belief distribution, and Shannon entropy."""
        return json.dumps(state_filter.get_state(), indent=2)

    @mcp.tool()
    def actinf_prescribe_policy() -> str:
        """Get mathematically optimal next engineering policy action to prevent context drift."""
        return json.dumps(state_filter.prescribe_policy(), indent=2)

    def main():
        mcp.run(transport="stdio")

except ImportError:
    # Pure standard library fallback with strict JSON-RPC 2.0 compliance
    def handle_request(req: dict) -> dict:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Notifications (no id) must be handled silently with no response
        if req_id is None or method.startswith("notifications/"):
            return None

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "micro-actinf", "version": "1.0.0"}
                }
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "actinf_observe",
                            "description": "Feed an incoming user/system observation into the Active Inference POMDP filter.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "obs_type": {
                                        "type": "string",
                                        "enum": ["general_chat", "code_request", "error_log", "math_query", "test_output", "architecture_choice", "confirmation", "unknown"]
                                    }
                                },
                                "required": ["obs_type"]
                            }
                        },
                        {
                            "name": "actinf_get_state",
                            "description": "Get current cognitive regime, belief distribution, and Shannon entropy.",
                            "inputSchema": {"type": "object", "properties": {}}
                        },
                        {
                            "name": "actinf_prescribe_policy",
                            "description": "Get mathematically optimal next engineering policy action to prevent context drift.",
                            "inputSchema": {"type": "object", "properties": {}}
                        }
                    ]
                }
            }

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name == "actinf_observe":
                res = state_filter.observe(args.get("obs_type", "unknown"))
            elif tool_name == "actinf_get_state":
                res = state_filter.get_state()
            elif tool_name == "actinf_prescribe_policy":
                res = state_filter.prescribe_policy()
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
            }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method {method}"}}

    def main():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                pass


if __name__ == "__main__":
    main()
