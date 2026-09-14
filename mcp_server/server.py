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

    def _softmax(self, vec):
        max_v = max(vec)
        exps = [math.exp(v - max_v) for v in vec]
        sum_exps = sum(exps)
        return [e / sum_exps for e in exps]

    def observe(self, obs_type: str) -> dict:
        obs_id = self.obs_map.get(obs_type.lower(), 7)
        logits = [math.log(max(b, 1e-12)) for b in self.beliefs]
        if obs_id == 1:
            logits[1] += 1.5  # Code gen
        elif obs_id == 2:
            logits[3] += 2.0  # Debugging
        elif obs_id == 3:
            logits[2] += 1.5  # Refactoring
        elif obs_id == 4:
            logits[4] += 1.8  # Verification
        elif obs_id == 5:
            logits[5] += 1.5  # Decision
        else:
            logits[0] += 1.0  # Exploration

        self.beliefs = self._softmax(logits)
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
            action_idx = 0  # Explore
        elif dom in (1, 2):
            action_idx = 1  # Pragmatic execute
        elif dom == 3:
            action_idx = 2  # Audit
        else:
            action_idx = 3  # Conclude

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
