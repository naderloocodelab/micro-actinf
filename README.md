# 🧠 Micro-ActInf: Ultra-Fast $O(1)$ Active Inference & Hard Cognitive Governor

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Standard](https://img.shields.io/badge/Standard-C11%20MISRA--C%3A2012%20Inspired-blue.svg)]()
[![Memory](https://img.shields.io/badge/RAM-20.69%20KB%20(Zero--Heap)-cyan.svg)]()
[![Latency](https://img.shields.io/badge/Inference%2BLearning-2.89%20%CE%BCs%20%2F%20step-green.svg)]()
[![Decision Cycle](https://img.shields.io/badge/Decision%20Cycle-50.62%20%CE%BCs-purple.svg)]()
[![Protocol](https://img.shields.io/badge/Protocol-MCP%20JSON--RPC%202.0%20(7%20Tools)-orange.svg)]()
[![Target](https://img.shields.io/badge/AI%20Agents-Antigravity%20%7C%20Claude%20%7C%20Cursor-magenta.svg)]()

> **Eliminate context drift, hallucination loops, and conversational token bloat in AI agents (Google Antigravity, Claude Desktop, Cursor, OpenAI Swarm) using Karl Friston's Free Energy Principle and a zero-heap 20.69 KB canonical C11 POMDP governor.**

---

## 📑 Table of Contents
- [Why Micro-ActInf? (The Problem & Solution)](#-why-micro-actinf-the-problem--solution)
- [Single Source of Truth Architecture](#-single-source-of-truth-architecture)
- [Feature Comparison vs Existing Frameworks](#-feature-comparison)
- [The Two-Phase Hard Governance Lifecycle](#-the-two-phase-hard-governance-lifecycle)
- [The 7 MCP Governance Tools](#-the-7-mcp-governance-tools)
- [Semantic Safety Inspection & Risk Neutralization](#-semantic-safety-inspection--risk-neutralization)
- [FNV-1a Execution Fingerprinting & Loop Prevention](#-fnv-1a-execution-fingerprinting--loop-prevention)
- [The 6 Cognitive Regimes & 4 Action Policies](#-the-6-cognitive-regimes--4-action-policies)
- [Google Antigravity Deep Integration Guide](#-google-antigravity-deep-integration-guide)
- [Setup for Other AI Agents (Claude Desktop, Cursor, Ollama)](#-setup-for-other-ai-agents)
- [C11 Core & Embedded Systems](#-c11-core--embedded-systems)
- [Mathematical Foundation & Complexity Calibration](#-mathematical-foundation--complexity-calibration)
- [Empirical Benchmarks & Verification](#-empirical-benchmarks--verification)
- [🇮🇷 راهنمای جامع فارسی و فعال‌سازی در Google Antigravity](#-راهنمای-جامع-فارسی-و-فعالسازی-در-google-antigravity)

---

## 💡 Why Micro-ActInf? (The Problem & Solution)

### 🚨 The Autonomous AI Agent Crisis:
1. **Context Drift & Goal Degradation:** In long multi-turn sessions, LLMs suffer from attention dilution, forgetting architectural constraints and drifting into irrelevant tangents.
2. **Infinite Debugging Loops:** When facing compiler errors or failed unit tests, LLMs frequently propose the exact same failing edits repeatedly with zero progress.
3. **Conversational Token Waste:** Up to 40% of generated tokens are squandered on polite filler (*"Certainly!", "I'd be happy to help!"*), apologies, and essays when only working production code is required.
4. **Self-Reported Safety Bypass:** Autonomous agents often hallucinate `risk_level="LOW"` for destructive operations (`rm -rf`, `DROP TABLE`), bypassing naive safety guards.
5. **Soft Constraints Fail:** Prompt instructions (*"be concise"*, *"focus on tests"*) are probabilistic suggestions that inevitably degrade under heavy context windows.

### 🛡️ The Micro-ActInf Solution:
**Micro-ActInf** transforms the Model Context Protocol (MCP) from a passive text helper into an **active Bayesian cognitive governor**:
- **Single Source of Truth:** A lightning-fast canonical C11 engine (`libmicro_actinf.dll` / `libmicro_actinf.so`) powers both native C applications and Python MCP server via direct `ctypes` FFI bindings with zero duplicated math.
- **Variational POMDP Filtering:** Continuously updates the agent's belief state $\mathbf{s}_t$ across a 6-regime probability simplex with adaptive recency decay $\alpha = 0.25$ to eliminate Bayesian inertia.
- **Multi-Step Cost-Aware Expected Free Energy ($G$):** Evaluates multi-step policy trajectories ($H \in [1, 4]$) balancing epistemic uncertainty reduction against pragmatic goals, token costs, latency budgets, and operational risk.
- **FNV-1a Behavioral Fingerprinting:** Hashes tools and arguments to detect recurring commands. If a repetitive action yields $\le 0.001$ progress delta, it is unconditionally `DENIED`.
- **Semantic Command Classifier:** Neutralizes self-reported risk bypass by intercepting shell commands and file mutations, automatically escalating destructive patterns to `DESTRUCTIVE` or forcing `MODIFY` dry-runs.
- **Goal Drift & Progress Tracking:** Continuously measures Total Variation distance against the target goal regime ($1 - \text{TV}$) to detect cognitive drift before errors cascade.

---

## 🏛️ Single Source of Truth Architecture

```
                                 ┌────────────────────────────────────────────────────────┐
                                 │           Canonical C11 Engine (Core Truth)            │
                                 │      20.69 KB Static BSS · Zero Dynamic Malloc         │
                                 │       libmicro_actinf.dll / libmicro_actinf.so         │
                                 └───────────────────────────┬────────────────────────────┘
                                                             │
                                   ┌─────────────────────────┴─────────────────────────┐
                                   ▼                                                   ▼
                   ┌───────────────────────────────┐                   ┌───────────────────────────────┐
                   │    Python CTYPES FFI Layer    │                   │   Direct C11 Native Linkage   │
                   │    (mcp_server/libactinf.py)  │                   │   (Embedded, Robotics, Game)  │
                   └───────────────┬───────────────┘                   └───────────────┬───────────────┘
                                   │                                                   │
                   ┌───────────────┴───────────────┐                                   │
                   ▼                               ▼                                   │
   ┌───────────────────────────────┐ ┌───────────────────────────────┐                 │
   │  FastMCP Standard Server      │ │ Stdio JSON-RPC 2.0 Fallback   │                 │
   │  (Claude, Antigravity, Cursor)│ │ (Zero Dependency Python STL)  │                 │
   └───────────────┬───────────────┘ └───────────────┬───────────────┘                 │
                   │                                 │                                 │
                   └────────────────┬────────────────┘                                 │
                                    ▼                                                  ▼
                     ┌─────────────────────────────┐                    ┌─────────────────────────────┐
                     │     AI Agent Runtime        │                    │   Real-Time Robotics / MCU  │
                     │  Google Antigravity         │                    │   ARM Cortex-M4, RISC-V     │
                     │  Anthropic Claude Desktop   │                    │   Unreal / Unity Game AI    │
                     │  Cursor / Windsurf / Copilot│                    │   > 340,000 decisions/sec   │
                     └─────────────────────────────┘                    └─────────────────────────────┘
```

---

## ⚖️ Feature Comparison

| Capability | Standard LLM Prompting | LangGraph / AutoGen | **Micro-ActInf (This Engine)** |
| :--- | :---: | :---: | :---: |
| **Runtime Footprint** | 0 KB (uncontrolled) | 250 MB – 1.2 GB (Python/PyTorch) | **20.69 KB (Zero Heap, L1 Cache)** |
| **Combined Step Latency** | N/A | 50 ms – 300 ms | **2.89 µs (C11) / < 1 ms (MCP Stdio)** |
| **Decision Cycle Latency** | N/A | > 100 ms | **50.62 µs (Full Multi-Step EFE)** |
| **Cognitive State Tracking** | Stochastic text memory | Static State Machine | **Variational Bayes POMDP on Simplex** |
| **Goal Drift Tracking** | ❌ None | Manual checkpoints | **Continuous Total Variation ($1 - \text{TV}$)** |
| **Loop & Inertia Prevention** | ❌ Fails frequently | Simple retry counter | **FNV-1a Fingerprint & Zero-Delta Gate** |
| **Safety Bypass Neutralization** | ❌ Blind trust | Manual regex hooks | **Autonomous Semantic Command Classifier** |
| **Pre-Execution Safety Gating**| ❌ No gating | Custom Python hooks | **Hard MCP Action Gate (ALLOW/MODIFY/DENY)** |
| **Granular Resets** | All-or-nothing | Script restart | **3-Tier (Belief / Episode / Model Reset)** |
| **Real-Time Adaptation** | Token-heavy in-context | Slow offline fine-tuning | **$O(1)$ Online Dirichlet Learning** |
| **Memory Allocation** | Dynamic heap | Dynamic heap | **Zero `malloc` (MISRA-C:2012 Inspired)** |
| **MCP Integration** | ❌ No | Partial / Complex | **Native 7-Tool MCP FastMCP / JSON-RPC** |

---

## 🔄 The Two-Phase Hard Governance Lifecycle

```
===================================================================================================
PHASE 1: Cognitive State & Policy Lock (Pre-Execution)
===================================================================================================
  [ User Prompt / System Event ]
                │
                ▼
  [ Categorize Observation: obs_type ] ──► (e.g. code_request, error_log, test_output)
                │
                ▼
  [ MCP: actinf_observe(obs_type) ]
    ├── Bayesian State Update: s_t = Softmax( ln A_{o,:} + ln s_prior )
    ├── Recency Decay α = 0.25 (Inertia-Immune)
    └── Shannon Entropy Velocity: ΔH = H_t - H_{t-1}
                │
                ▼
  [ MCP: actinf_prescribe_policy() ]
    ├── Multi-Step Horizon EFE Minimization: G(u) = -(Pragmatic + β*Epistemic) + Costs
    └── Returns Mandatory Policy Regime:
        ├── PRAGMATIC_EXECUTE  ──► Output 100% production code, zero conversational preamble
        ├── AUDIT_DIAGNOSE     ──► Pinpoint root-cause & output exact diff patch, zero lecturing
        ├── EPISTEMIC_EXPLORE  ──► Formulate targeted clarifying questions
        └── CONVERGE_CONCLUDE  ──► Execute verification suites & sign off

===================================================================================================
PHASE 2: Hard Action Safety Gating, Execution Fingerprinting & Credit Assignment
===================================================================================================
  [ Agent Proposes Tool Action ] (e.g. replace_file_content, run_command)
                │
                ▼
  [ MCP: actinf_evaluate_action(proposed_tool, action_type, risk_level, confidence_threshold, tool_args) ]
    ├── Semantic Command Classifier: Inspects tool + args for destructive patterns (rm -rf, DROP TABLE, del /f)
    │   └── Neutralizes self-reported risk bypass ──► Promotes risk to DESTRUCTIVE if matched
    ├── FNV-1a Signature Check: Has this exact tool + arguments run previously with delta <= 0.001?
    │   └── Zero-Progress Stuck Command ──► Returns DENY unconditionally
    ├── Loop Check: Has (state, action) looped >= 3 times? ──► Returns DENY
    ├── Destructive Gating: Is action CRITICAL/DESTRUCTIVE? ──► Returns ASK_CONFIRMATION or MODIFY (dry-run)
    ├── Confidence Threshold: Is belief confidence sufficient? ──► Returns ALLOW or MODIFY
    └── If ALLOW: Execute tool immediately
                │
                ▼
  [ Tool Execution Completes ] ──► (success = True / False, delta)
                │
                ▼
  [ MCP: actinf_record_outcome(action, outcome_obs, success, progress_delta) ]
    ├── Online Dirichlet Expectation Adaptation: a_{o,s}, b_{s',s,u}
    ├── Context Goal Tracking: Updates goal_progress and goal_drift (1 - Total Variation)
    └── Utility Credit Assignment: C(o) += η * Δ, Progress Index Tracking
```

---

## 🛠️ The 7 MCP Governance Tools

Micro-ActInf exposes 7 standardized tools via Model Context Protocol (MCP):

### 1. `actinf_observe(obs_type, context_attributes, tool, args, progress_delta)`
- **Purpose:** Updates Bayesian belief state on the 6-regime simplex upon receiving user input, system event, or action execution.
- **Arguments:**
  - `obs_type` (str, required): `general_chat`, `code_request`, `error_log`, `math_query`, `test_output`, `architecture_choice`, `confirmation`, `unknown`.
  - `context_attributes` (dict, optional): Additional telemetry.
  - `tool` (str, optional): Tool identifier for FNV-1a signature calculation.
  - `args` (dict, optional): Arguments payload for FNV-1a signature calculation.
  - `progress_delta` (float, optional): Observed progress increment.
- **Output:** Current dominant cognitive regime, confidence percentage, Shannon entropy in nats, entropy velocity, goal progress, and belief distribution.

### 2. `actinf_get_state()`
- **Purpose:** Cycle-accurate snapshot of the internal POMDP governor.
- **Output:** Full belief simplex, entropy velocity, loop status, target goal regime, goal progress ($1 - \text{TV}$), context goal drift, progress index, EFE values per action, and active engine backend (`libmicro_actinf.dll` / `.so`).

### 3. `actinf_prescribe_policy()`
- **Purpose:** Returns the mathematically optimal control action policy and user-facing directive.
- **Output:** `prescribed_action` (`PRAGMATIC_EXECUTE`, `AUDIT_DIAGNOSE`, `EPISTEMIC_EXPLORE`, `CONVERGE_CONCLUDE`), action index, strict behavioral directive, and rationale.

### 4. `actinf_evaluate_action(proposed_tool, action_type, risk_level, confidence_threshold, tool_args)`
- **Purpose:** Hard pre-execution safety gate with autonomous semantic inspection and FNV-1a signature loop blocking.
- **Arguments:**
  - `proposed_tool` (str): Tool identifier (e.g. `replace_file_content`, `run_command`, `write_to_file`).
  - `action_type` (str): `READ`, `EDIT`, `EXECUTE`, `DIAGNOSE`, `VERIFY`.
  - `risk_level` (str): `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `DESTRUCTIVE`.
  - `confidence_threshold` (float): Minimum confidence required (default `0.80`).
  - `tool_args` (dict/str, optional): Arguments payload evaluated by the Semantic Command Classifier.
- **Verdicts:**
  - `ALLOW` (0): Proceed with execution immediately.
  - `MODIFY` (1): Downgrade action parameters with concrete instructions (e.g. `DRY_RUN_OR_DIFF_PREVIEW`, `INSPECT_ONLY`).
  - `ASK_CONFIRMATION` (2): Pause and prompt human operator.
  - `DENY` (3): Block execution (stuck loop, zero progress repeat, or regime violation).

### 5. `actinf_record_outcome(action, outcome_obs, success, progress_delta)`
- **Purpose:** Reinforces or penalizes prior preferences $C(o)$ and transitions $B(u)$ based on execution feedback.
- **Arguments:**
  - `action` (str/int): Executed action.
  - `outcome_obs` (str): Resulting observation category.
  - `success` (bool): Whether the action achieved its objective.
  - `progress_delta` (float): Progress increment (default `0.10` to `0.25`).

### 6. `actinf_set_goal(target_regime)`
- **Purpose:** Establishes an explicit cognitive destination on the simplex.
- **Arguments:**
  - `target_regime` (str/int, required): `EXPLORATION`, `CODE_GENERATION`, `REFACTORING`, `DEBUGGING`, `VERIFICATION`, `DECISION`.
- **Output:** Sets canonical target state and computes baseline goal progress ($1 - \text{TV}$) and goal drift.

### 7. `actinf_reset(reset_type)`
- **Purpose:** Provides 3-tier granular reset flexibility without losing learned parameters unnecessarily.
- **Arguments:**
  - `reset_type` (str, optional):
    - `"belief"` (0, default): Turn-level reset. Resets belief simplex $\mathbf{s}_t$ to uniform prior; retains history buffer and learned Dirichlet counts.
    - `"episode"` (1): Episode/session boundary reset. Clears 8-step history buffer, loop flags, and belief simplex; retains learned transition tensors $\mathbf{B}$ and preferences $\mathbf{C}$.
    - `"model"` (2): Factory reset. Restores initial priors, zeroes Dirichlet pseudo-counts, and clears all operational state.

---

## 🛡️ Semantic Safety Inspection & Risk Neutralization

A critical failure mode of LLM agents is **Self-Reported Risk Bypass**: when an LLM is asked to evaluate its own action, it frequently self-reports `risk_level="LOW"` for catastrophic commands (`rm -rf /`, `DROP TABLE users;`, `del /f /s /q C:\Windows`).

Micro-ActInf eliminates this vulnerability via an **Autonomous Semantic Command Classifier**:
1. **Tool and Argument Inspection:** Evaluates the actual payload of `run_command`, `write_to_file`, `replace_file_content`, and database queries against a rigorous regex rulebook of destructive shell operations and SQL statements.
2. **Autonomous Risk Promotion:** If a command matches destructive signatures (e.g. recursive deletions, partition formatting, forced terminations, raw SQL drop tables), the internal risk is escalated to `DESTRUCTIVE` regardless of what the LLM claimed.
3. **Action Downgrade with Actionable Directives:** Rather than an ambiguous `MODIFY` verdict, Micro-ActInf returns concrete behavioral modifications:
   - `modification_directive: "DRY_RUN_OR_DIFF_PREVIEW"`: Instructs the agent to simulate or print diffs before mutating.
   - `modification_directive: "INSPECT_ONLY"`: Downgrades write commands to read-only inspections.

---

## 🔁 FNV-1a Execution Fingerprinting & Loop Prevention

When AI agents encounter persistent bugs, they often enter an **infinite retry loop**, executing identical commands or edits that yield zero progress.

Micro-ActInf incorporates deterministic **FNV-1a 32-bit execution fingerprinting**:
$$\text{hash} = \text{FNV-1a}(\text{tool} \,\|\, \text{args})$$
- Each executed command fingerprint and its resulting `progress_delta` are recorded in the canonical 8-step ring-buffer history in the C core.
- **Hard Zero-Progress Gate:** If an incoming action has an identical fingerprint to a previous execution in history that produced $\le 0.001$ progress delta, the safety governor returns **`DENY` unconditionally**.
- The agent is prevented from executing the identical failing command and forced to reconsider its approach or ask for clarification.

---

## 🧭 The 6 Cognitive Regimes & 4 Action Policies

| Index | Cognitive Regime | Observation Trigger | Enforced Policy | Mandatory LLM Behavior |
| :---: | :--- | :--- | :---: | :--- |
| **0** | **EXPLORATION** | `general_chat` | **EPISTEMIC_EXPLORE** (0) | Clarifies technical ambiguities and requirements before writing code. |
| **1** | **CODE_GENERATION** | `code_request` | **PRAGMATIC_EXECUTE** (1) | Generates 100% production code immediately. Zero greetings, zero filler. |
| **2** | **REFACTORING** | `math_query` | **PRAGMATIC_EXECUTE** (1) | Optimizes algorithms, compresses matrix ranks, cleans technical debt. |
| **3** | **DEBUGGING** | `error_log` | **AUDIT_DIAGNOSE** (2) | Pinpoints root cause, inspects bounds/memory, emits unified diff patch. |
| **4** | **VERIFICATION** | `test_output` | **CONVERGE_CONCLUDE** (3) | Executes verification suites, reports numerical metrics, checks regression. |
| **5** | **DECISION** | `confirmation` | **CONVERGE_CONCLUDE** (3) | Locks architecture, finalizes commits, and signs off task cleanly. |

---

## 🛰️ Google Antigravity Deep Integration Guide

Google Antigravity provides native support for Model Context Protocol servers and project instructions. Micro-ActInf acts as an external **Hard Cognitive Governor** inside Antigravity sessions.

### Step 1: Configure MCP in Antigravity
Open `C:\Users\<username>\.gemini\antigravity\mcp_config.json` (or click **Settings -> MCP Servers -> Add Server** in Antigravity IDE):

```json
{
  "mcpServers": {
    "micro-actinf": {
      "command": "python",
      "args": [
        "C:/path/to/micro-actinf/mcp_server/server.py"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Step 2: Establish Inviolable Pre-Execution Rules
Create or edit `AGENTS.md` and `GEMINI.md` in your workspace root (or globally at `~/.gemini/config/GEMINI.md`):

```markdown
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

3. **HARD GOVERNANCE ACTION GATING & LEARNING (PHASE 2):**
   - **Action Safety Evaluation (Pre-Execution):** When proposing tool executions carrying operational or filesystem risk (`write_to_file`, `replace_file_content`, `run_command`), invoke:
     `call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_evaluate_action", Arguments={"proposed_tool": "<tool_name>", "action_type": "<EDIT|EXECUTE|READ>", "risk_level": "<READ|EDIT|HIGH|CRITICAL>"})`
     If the verdict is `DENY`, abort the action immediately. If `ASK_CONFIRMATION`, request explicit confirmation from the user.
   - **Credit Assignment Feedback (Post-Execution):** After tool execution completes:
     `call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_record_outcome", Arguments={"action": "<action_type>", "outcome_obs": "<outcome>", "success": <true|false>, "progress_delta": 0.25})`
     This updates prior preferences C(o) and reinforces successful cognitive trajectories.
```

### Step 3: Verify the Two-Phase Closed Loop
Run the automated Antigravity workflow simulator:
```bash
python examples/antigravity_governor_workflow.py
```
Expected output demonstrates seamless transitions, zero prompt conversational fluff, automated loop breakout, and risk gating.

---

## 🔧 Setup for Other AI Agents

### Anthropic Claude Desktop
Edit `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):
```json
{
  "mcpServers": {
    "micro-actinf": {
      "command": "python",
      "args": ["/absolute/path/to/micro-actinf/mcp_server/server.py"]
    }
  }
}
```

### Cursor / VS Code
Add to your project `.cursorrules` or `.vscode/settings.json`:
```markdown
Before editing files or proposing terminal commands, query micro-actinf MCP server.
Adhere strictly to the returned regime (PRAGMATIC_EXECUTE, AUDIT_DIAGNOSE, etc.).
```

### Offline Local LLMs (Ollama / vLLM / llama.cpp)
```bash
python examples/llm_agent_runner.py \
  --provider ollama \
  --base-url "http://localhost:11434/v1" \
  --model "qwen2.5-coder:7b" \
  "Write an AVX2 vectorized dot-product in C11."
```

---

## ⚡ C11 Core & Embedded Systems

The computational core is written in portable C11 with zero heap allocations (`malloc`/`free` strictly forbidden):

```c
#include "micro_actinf.h"

int main(void) {
    micro_actinf_t agent;
    micro_actinf_init(&agent, 6, 8, 4);

    /* Real-time observation inference step (< 2.7 microseconds) */
    micro_actinf_step(&agent, 1 /* OBS_CODE_REQUEST */);

    /* Multi-step Expected Free Energy action selection */
    uint8_t action = micro_actinf_select_action(&agent);

    /* Safety evaluation */
    actinf_verdict_t verdict = micro_actinf_evaluate_action(&agent, action, ACTINF_RISK_EDIT, 0.80f);
    if (verdict == ACTINF_VERDICT_ALLOW) {
        /* Execute action and record feedback */
        micro_actinf_record_outcome(&agent, action, 1, true, 0.25f);
    }

    return 0;
}
```

### Compile and Verify:
```bash
# Compile shared library and test suite
gcc -O3 -shared -DMICRO_ACTINF_BUILD_DLL -Iinclude src/micro_actinf.c -o libmicro_actinf.dll -lm
gcc -O3 -Iinclude tests/test_c_core.c src/micro_actinf.c -o test_c_core -lm
./test_c_core

# Run Python behavioral verification
python tests/test_governor_behavioral.py
python tests/test_persian_6_scenarios.py
```

---

## 📐 Mathematical Foundation & Complexity Calibration

### 1. Variational Bayes Belief Update
$$\mathbf{s}_{t+1} = \sigma\left( \ln \mathbf{A}_{o_t, :}^T + \ln \mathbf{s}_{\text{prior}} \right)$$
$$\mathbf{s}_{\text{prior}} = (1 - \alpha) \cdot \mathbf{B}(u_{t-1}) \mathbf{s}_t + \alpha \cdot \frac{1}{K} \mathbf{1}$$
Where:
- $\mathbf{s}_t \in \Delta^{K-1}$: Categorical probability simplex across $K$ regimes.
- $\alpha = 0.25$: Adaptive prior decay preventing Bayesian inertia and deadlocks.
- $\sigma(\cdot)$: Numerically stabilized Softmax operator.

### 2. Multi-Step Cost-Aware Expected Free Energy ($G$)
$$G(u) = \sum_{\tau=1}^H \gamma^{\tau-1} \Big[ - \big( \text{Pragmatic}(\tau) + \beta \cdot \text{Epistemic}(\tau) \big) + \text{Cost}(u) + \text{LoopPenalty}(u) \Big]$$
- **Pragmatic Value:** $\sum_{o=1}^M o_{\text{pred}}(o) C(o)$
- **Epistemic Value (Mutual Information):** $H(O_{\text{pred}}) - \sum_{s=1}^K s_{\text{pred}}(s) H(O \mid S=s)$
- **Action Costs:** $w_{\text{token}} C_{\text{token}} + w_{\text{lat}} C_{\text{lat}} + w_{\text{risk}} C_{\text{risk}}$
- **Loop Penalty:** Heuristic ring-buffer penalty applied when $\ge 3$ stuck repetitions occur.

### 3. $O(1)$ Online Conjugate Dirichlet Learning
$$\mathbf{a}_{o_t, s} \leftarrow \lambda_a \cdot \mathbf{a}_{o_t, s} + \eta_a \cdot s_t(s) \quad \forall s \in \{0, \dots, K-1\}$$
$$\mathbf{b}_{s', s, u_{t-1}} \leftarrow \lambda_b \cdot \mathbf{b}_{s', s, u_{t-1}} + \eta_b \cdot s_t(s') \cdot s_{t-1}(s) \quad \forall s, s' \in \{0, \dots, K-1\}$$

### 4. Algorithmic Complexity Calibration
- **Time Complexity with respect to History ($T$):** Strictly $O(1)$. Memory consumption and inference time do not grow as conversation turns accumulate.
- **Step Complexity with respect to Topology Dimensions:**
  - Variational State Update: $O(K + M)$
  - Dirichlet Conjugate Learning: $O(K^2 + KM)$
  - Multi-Step EFE Policy Planning: $O(A \cdot H \cdot (K^2 + KM))$
  - For compile-time bounded topology ($K \le 16, M \le 32, A \le 8, H \le 4$), execution executes in deterministic constant time ($\approx 2.89\text{ }\mu\text{s}$ combined step, $\approx 50.62\text{ }\mu\text{s}$ full decision cycle).
- **Fast Approximate Logarithm Precision:** Evaluated over $1,000,000$ points in $(0, 1]$; max absolute error $0.00762$, invariant $100.00\%$ top-1 action ranking preservation.

---

## 📊 Empirical Benchmarks & Verification

Tested on x86_64 host (GCC `-O3`) and simulated ARM Cortex-M4:

| Metric | Measured Value | Verification Suite |
| :--- | :--- | :--- |
| **Static Memory Footprint** | **20.69 KB (21,188 Bytes)** | `test_c_core` [TEST 3] (Budget $\le 36.00\text{ KB}$) |
| **Dynamic Heap Allocation (`malloc`)** | **Strictly 0 Bytes** | Static assertion & zero-heap audit |
| **Combined Step Latency (Inference + Learning)** | **2.895 µs / step** | `test_c_core` [TEST 5] (100,000 cycles) |
| **Full Decision Cycle (Multi-Step EFE)** | **50.621 µs / cycle** | `test_c_core` [TEST 6] (50,000 cycles) |
| **Throughput** | **> 340,000 decisions / sec** | Continuous real-time loop |
| **Shannon Entropy Collapse** | **$> 70\%$ collapse on evidence** | `test_c_core` [TEST 2] |
| **Log Precision & Rank Invariance** | **100.00% top-1 rank invariant** | `test_fast_log_accuracy` (1,000,000 points) |
| **ABI Memory Alignment** | **100% C/Python offset match** | `test_abi_alignment.py` |
| **Loop Breakout Guarantee** | **100% automated breakout** | `test_governor_behavioral.py` |
| **Zero-Delta Signature Blocking** | **100% stuck repeat DENIED** | `test_c_core` [TEST 18] |
| **Action Safety Gating Accuracy** | **100% correct verdicts** | `test_governor_behavioral.py` |
| **C Test Suite Total Verification** | **18 / 18 Suites PASS (100%)** | `test_c_core` |

---

## 🇮🇷 راهنمای جامع فارسی و فعال‌سازی در Google Antigravity

### این پروژه دقیقاً چه مشکلی را حل می‌کند؟
مدل‌های هوش مصنوعی پیشرفته (مانند Claude 3.7، Gemini 2.0، GPT-4.5، Cursor و Antigravity) هنگام توسعه نرم‌افزار با ۴ چالش بزرگ مواجهند:
1. **انحراف تمرکز (Context Drift):** با طولانی شدن چت، مدل هدف اصلی پروژه را گم می‌کند و وارد جزئیات غیرمرتبط می‌شود.
2. **لوپ‌های تکراری و بی‌پایان در دیباگ:** در صورت مواجهه با خطای کامپایل، مدل راه‌حل غلط قبلی را دوباره و دوباره تکرار می‌کند.
3. **هدررفت توکن با تعارفات بی‌فایده:** تا ۴۰٪ خروجی مدل صرف جملات مقدماتی و مؤدبانه (*"سلام"، "حتماً، در ادامه کد را برایتان نوشتم..."*) و انشاهای طولانی می‌شود.
4. **عدم پایداری دستورات پرامپت:** حتی اگر بنویسید *"خلاصه جواب بده"*, در کانتکست‌های بزرگ این دستورات فراموش می‌شوند.

**Micro-ActInf** یک موتور فرموله شده بر اساس **اصل حداقل انرژی آزاد کارل فریستون (Active Inference)** است که در یک فایل هسته سبک **۲۰ کیلوبایتی با زبان C11** نوشته شده و از طریق پروتکل استاندار MCP به عنوان یک **ناظر بالادستی (Cognitive Governor)** به هوش مصنوعی متصل می‌شود.

---

### راهنمای گام‌به‌گام فعال‌سازی در Google Antigravity:

#### گام ۱: افزودن سرور MCP به انتی‌گرویتی
فایل پیکربندی MCP در سیستم خود را باز کنید:
- در ویندوز: `C:\Users\<نام_کاربر>\.gemini\antigravity\mcp_config.json`
- یا داخل محیط Google Antigravity به مسیر **Settings ➔ MCP Servers ➔ Add New Server** بروید.

محتوای زیر را اضافه کنید:
```json
{
  "mcpServers": {
    "micro-actinf": {
      "command": "python",
      "args": [
        "C:/مسیر_پروژه/micro-actinf/mcp_server/server.py"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

#### گام ۲: فعال‌سازی قانون حاکمیتی قطعی در فایل GEMINI.md
برای اینکه انتی‌گرویتی در **تمامی پیام‌ها** ملزم به فراخوانی استنتاج فعال باشد، فایل `GEMINI.md` یا `AGENTS.md` پروژه را با متن زیر تنظیم کنید:

```markdown
# قانون تخطی‌ناپذیر: استنتاج فعال (MICRO-ACTINF) در هر چرخه

در هر پیام کاربر، قبل از تولید حتی ۱ کلمه پاسخ، فوراً این ۲ دستور را صدا بزن:
۱. call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_observe", Arguments={"obs_type": "<نوع_مشاهده>"})
۲. call_mcp_tool(ServerName="micro-actinf", ToolName="actinf_prescribe_policy", Arguments={})

قفل کامل خروجی بر اساس خط‌مشی دریافتی:
- اگر PRAGMATIC_EXECUTE بود: ۱۰۰٪ کد پروداکشن بدون هیچ احوالپرسی یا مقدمه.
- اگر AUDIT_DIAGNOSE بود: علت ریشه‌ای خطا و پچ دقیق بدون سخنرانی.
- اگر EPISTEMIC_EXPLORE بود: فقط سوالات شفاف‌ساز فنی.
- اگر CONVERGE_CONCLUDE بود: بنچمارک عددی و پایان کار.
```

#### گام ۳: آزمایش عملیاتی سوییچینگ شناختی (۶ سناریوی واقعی)
برای مشاهده سوئیچینگ زنده بین ۶ حالت شناختی به زبان فارسی، دستور زیر را اجرا کنید:
```bash
python tests/test_persian_6_scenarios.py
```

خروجی آزمون:
- پیام عمومی: ➔ سوییچ به `EXPLORATION` ➔ اکشن `EPISTEMIC_EXPLORE` (طرح سوال فنی)
- درخواست کد: ➔ سوییچ به `CODE_GENERATION` ➔ اکشن `PRAGMATIC_EXECUTE` (تولید مستقیم کد پروداکشن)
- خطای کرش: ➔ سوییچ به `DEBUGGING` ➔ اکشن `AUDIT_DIAGNOSE` (پچ خط‌به‌خط بدون اتلاف وقت)
- خروجی تست‌ها: ➔ سوییچ به `VERIFICATION` ➔ اکشن `CONVERGE_CONCLUDE` (سنجش عددی بنچمارک)
- معادلات ریاضی: ➔ سوییچ به `REFACTORING` ➔ اکشن `PRAGMATIC_EXECUTE` (کاهش بعد ماتریس‌ها)
- تایید نهایی: ➔ سوییچ به `DECISION` ➔ اکشن `CONVERGE_CONCLUDE` (مرج برنچ و بستن تسک)

---

## 📄 License
Released under the [MIT License](LICENSE).  
Authored with mathematical rigor and systems engineering discipline by **[naderloocodelab](https://github.com/naderloocodelab)**.
