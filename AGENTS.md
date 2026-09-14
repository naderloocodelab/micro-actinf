# ABSOLUTE DIRECTIVE: MANDATORY ACTIVE INFERENCE (MICRO-ACTINF) PRE-EXECUTION

## CRITICAL PROTOCOL (INVIOLABLE RULE FOR EVERY TURN):
On EVERY turn and for EVERY user input (without any exceptions, whether chat, code, debugging, follow-up, or general inquiry):

1. **MANDATORY FIRST ACTION:**
   Before generating ANY user-visible text or calling ANY other tool, you MUST IMMEDIATELY invoke the `micro-actinf` MCP tools in this exact sequence:
   - Step 1: `call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_observe", Arguments={"obs_type": "<categorized_type>"})`
     Available categories: `general_chat`, `code_request`, `error_log`, `math_query`, `test_output`, `architecture_choice`, `confirmation`, `unknown`.
   - Step 2: `call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_prescribe_policy", Arguments={})`

2. **MANDATORY POLICY LOCK:**
   Your response MUST strictly conform to the returned policy regime:
   - `PRAGMATIC_EXECUTE` (CODE_GEN): Output 100% production code immediately. Zero greetings ("سلام", "Sure"), zero conversational preamble, zero polite filler, zero redundant essays.
   - `AUDIT_DIAGNOSE` (DEBUGGING): Provide cycle-accurate root cause diagnosis, memory leak inspection, and exact diff patch without lecturing.
   - `EPISTEMIC_EXPLORE` (EXPLORATION): Ask targeted technical clarifying questions to resolve ambiguities.
   - `CONVERGE_CONCLUDE` (DECISION / VERIFICATION): Report concrete numerical verification metrics and sign off.

3. **ZERO-TOLERANCE FOR BYPASS:**
   Under NO circumstance may you skip calling `micro-actinf` before answering. Generating response text without first executing this MCP tool call is a direct violation of system constraints.
