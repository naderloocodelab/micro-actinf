#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Context Protocol (MCP) Cognitive Governor Server for Micro-ActInf
-----------------------------------------------------------------------
Provides Active Inference Cognitive State Tracking, Policy Prescription,
Action Safety Evaluation (ALLOW/MODIFY/DENY/ASK), and Outcome Credit Learning
for Autonomous AI Agents (Google Antigravity, Claude Desktop, Cursor).

Architecture: Single Source of Truth — Powered directly by C11 Static Memory Engine.
Transport: Standard I/O (JSON-RPC 2.0 stdio via FastMCP & robust zero-dependency fallback)
"""

import os
import sys
import json
from typing import Optional, Dict, Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

# Import the Canonical C11 Engine Binding
try:
    from libactinf import MicroActInfEngine, REGIMES, POLICIES, RISK_LEVELS, VERDICTS
except ImportError:
    # Ensure current directory is in path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from libactinf import MicroActInfEngine, REGIMES, POLICIES, RISK_LEVELS, VERDICTS

# Instantiate canonical Active Inference Governor
governor = MicroActInfEngine()
state_filter = governor  # Backward-compatible alias for existing scripts

# Tool schemas and documentation
TOOL_DEFINITIONS = [
    {
        "name": "actinf_observe",
        "description": "Feed an incoming user prompt, system event, or tool result into the Active Inference POMDP filter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "obs_type": {
                    "type": "string",
                    "enum": [
                        "general_chat",
                        "code_request",
                        "error_log",
                        "math_query",
                        "test_output",
                        "architecture_choice",
                        "confirmation",
                        "unknown"
                    ],
                    "description": "Categorized sensory observation type."
                },
                "context_attributes": {
                    "type": "string",
                    "description": "Optional JSON string with additional metadata (e.g. failure_signal, risk, context_pressure)."
                },
                "tool": {
                    "type": "string",
                    "description": "Optional tool name for execution signature tracking."
                },
                "args": {
                    "type": "object",
                    "description": "Optional tool arguments for execution signature tracking."
                },
                "progress_delta": {
                    "type": "number",
                    "description": "Progress delta associated with this observation step."
                }
            },
            "required": ["obs_type"]
        }
    },
    {
        "name": "actinf_get_state",
        "description": "Get current cognitive regime, belief distribution, uncertainty, goal tracking, and loop status.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "actinf_prescribe_policy",
        "description": "Get mathematically optimal next engineering policy action minimizing multi-step Expected Free Energy.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "actinf_evaluate_action",
        "description": "Hard Governor Gate: Evaluates proposed tool and arguments against cognitive regime and automated semantic safety model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposed_tool": {
                    "type": "string",
                    "description": "Name of the tool the agent intends to call (e.g. write_to_file, run_command, replace_file_content)."
                },
                "tool_args": {
                    "type": "object",
                    "description": "Actual arguments to the tool (e.g. CommandLine, TargetFile) for semantic risk inspection."
                },
                "action_type": {
                    "type": "string",
                    "enum": ["EPISTEMIC_EXPLORE", "PRAGMATIC_EXECUTE", "AUDIT_DIAGNOSE", "CONVERGE_CONCLUDE"],
                    "description": "Category of the action being taken."
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["READ", "ANALYZE", "TEST", "EDIT", "EXECUTE", "DESTRUCTIVE"],
                    "description": "Optional self-reported risk profile (will be verified by semantic classifier)."
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence required to permit high-risk execution (default: 0.80)."
                }
            },
            "required": ["proposed_tool"]
        }
    },
    {
        "name": "actinf_record_outcome",
        "description": "Record tool execution outcome and execute credit assignment learning to improve future policy selection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["EPISTEMIC_EXPLORE", "PRAGMATIC_EXECUTE", "AUDIT_DIAGNOSE", "CONVERGE_CONCLUDE"],
                    "description": "Action that was executed."
                },
                "outcome_obs": {
                    "type": "string",
                    "enum": [
                        "general_chat",
                        "code_request",
                        "error_log",
                        "math_query",
                        "test_output",
                        "architecture_choice",
                        "confirmation",
                        "unknown"
                    ],
                    "description": "Resulting observation type observed after tool execution."
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the action resulted in positive progress or an error/failure."
                },
                "progress_delta": {
                    "type": "number",
                    "description": "Quantitative progress improvement metric (e.g. 0.1 to 1.0)."
                }
            },
            "required": ["action", "outcome_obs", "success"]
        }
    },
    {
        "name": "actinf_set_goal",
        "description": "Set target convergence goal regime and compute trajectory drift metric.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_state": {
                    "type": "string",
                    "description": "Target goal regime name (e.g. VERIFICATION, DECISION, CODE_GENERATION)."
                }
            },
            "required": ["target_state"]
        }
    },
    {
        "name": "actinf_reset",
        "description": "Reset cognitive state (turn belief, episode history, or full model re-initialization).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reset_type": {
                    "type": "string",
                    "enum": ["belief", "episode", "model"],
                    "description": "Granular reset scope: 'belief' (current belief vector only), 'episode' (belief and history, preserving learning), or 'model' (full factory reset)."
                }
            }
        }
    }
]


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch tool execution to the canonical C11 engine."""
    if tool_name == "actinf_observe":
        obs_type = args.get("obs_type", "unknown")
        ctx = None
        if "context_attributes" in args and isinstance(args["context_attributes"], str):
            try:
                ctx = json.loads(args["context_attributes"])
            except Exception:
                ctx = None
        tool = args.get("tool")
        tool_args = args.get("args")
        delta = float(args.get("progress_delta", 0.0))
        return governor.observe(obs_type, ctx, tool=tool, args=tool_args, progress_delta=delta)

    elif tool_name == "actinf_get_state":
        return governor.get_state()

    elif tool_name == "actinf_prescribe_policy":
        return governor.prescribe_policy()

    elif tool_name == "actinf_evaluate_action":
        tool = args.get("proposed_tool", "unknown")
        act_type = args.get("action_type", "PRAGMATIC_EXECUTE")
        risk = args.get("risk_level")
        conf_th = float(args.get("confidence_threshold", 0.80))
        tool_args = args.get("tool_args") or args.get("args")
        return governor.evaluate_action(tool, act_type, risk, conf_th, tool_args=tool_args)

    elif tool_name == "actinf_record_outcome":
        act = args.get("action", "PRAGMATIC_EXECUTE")
        outcome_obs = args.get("outcome_obs", "unknown")
        success = bool(args.get("success", True))
        delta = float(args.get("progress_delta", 0.10))
        return governor.record_outcome(act, outcome_obs, success, delta)

    elif tool_name == "actinf_set_goal":
        target = args.get("target_state", "VERIFICATION")
        return governor.set_goal(target)

    elif tool_name == "actinf_reset":
        rtype = args.get("reset_type", "episode")
        if rtype == "belief":
            return governor.reset_belief()
        elif rtype == "model":
            return governor.reset_model()
        return governor.reset_episode()

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# FastMCP Server Setup
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("micro-actinf")

    @mcp.tool()
    def actinf_observe(obs_type: str,
                       context_attributes: str = "",
                       tool: Optional[str] = None,
                       args: Optional[Dict[str, Any]] = None,
                       progress_delta: float = 0.0) -> str:
        """Feed sensory observation into the Active Inference POMDP filter with signature and progress tracking."""
        res = execute_tool("actinf_observe", {
            "obs_type": obs_type,
            "context_attributes": context_attributes,
            "tool": tool,
            "args": args,
            "progress_delta": progress_delta
        })
        return json.dumps(res, indent=2)

    @mcp.tool()
    def actinf_get_state() -> str:
        """Get current cognitive regime, belief distribution, uncertainty, goal tracking, and loop status."""
        return json.dumps(execute_tool("actinf_get_state", {}), indent=2)

    @mcp.tool()
    def actinf_prescribe_policy() -> str:
        """Get mathematically optimal next engineering policy action minimizing Expected Free Energy."""
        return json.dumps(execute_tool("actinf_prescribe_policy", {}), indent=2)

    @mcp.tool()
    def actinf_evaluate_action(proposed_tool: str,
                              action_type: str = "PRAGMATIC_EXECUTE",
                              risk_level: Optional[str] = None,
                              confidence_threshold: float = 0.80,
                              tool_args: Optional[Dict[str, Any]] = None) -> str:
        """Hard Governor Gate: Evaluates proposed LLM action against automated semantic risk model (ALLOW, MODIFY, ASK_CONFIRMATION, DENY)."""
        args = {
            "proposed_tool": proposed_tool,
            "action_type": action_type,
            "risk_level": risk_level,
            "confidence_threshold": confidence_threshold,
            "tool_args": tool_args
        }
        return json.dumps(execute_tool("actinf_evaluate_action", args), indent=2)

    @mcp.tool()
    def actinf_record_outcome(action: str,
                             outcome_obs: str,
                             success: bool,
                             progress_delta: float = 0.10) -> str:
        """Record execution outcome and execute credit assignment learning."""
        args = {
            "action": action,
            "outcome_obs": outcome_obs,
            "success": success,
            "progress_delta": progress_delta
        }
        return json.dumps(execute_tool("actinf_record_outcome", args), indent=2)

    @mcp.tool()
    def actinf_set_goal(target_state: str = "VERIFICATION") -> str:
        """Set target convergence goal regime and compute trajectory drift metric."""
        return json.dumps(execute_tool("actinf_set_goal", {"target_state": target_state}), indent=2)

    @mcp.tool()
    def actinf_reset(reset_type: str = "episode") -> str:
        """Reset belief state (scope: 'belief', 'episode', or 'model')."""
        return json.dumps(execute_tool("actinf_reset", {"reset_type": reset_type}), indent=2)

    def main():
        mcp.run(transport="stdio")

except ImportError:
    # Pure standard library fallback with strict JSON-RPC 2.0 compliance
    def handle_request(req: dict) -> Optional[dict]:
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if req_id is None or (method and method.startswith("notifications/")):
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
                    "serverInfo": {"name": "micro-actinf", "version": "2.0.0"}
                }
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOL_DEFINITIONS}
            }

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            try:
                res = execute_tool(tool_name, args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]}
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method {method} not found"}}

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
            except Exception:
                pass


if __name__ == "__main__":
    main()
