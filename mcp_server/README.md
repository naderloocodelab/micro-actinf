# 🛰️ Micro-ActInf Model Context Protocol (MCP) Server

[![Protocol: MCP](https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(JSON--RPC%202.0)-orange.svg)](https://modelcontextprotocol.io)
[![Core: C11](https://img.shields.io/badge/Core-C11%20Canonical%20Engine%20(libmicro__actinf)-blue.svg)]()
[![Memory: 20KB](https://img.shields.io/badge/Memory-20.62%20KB%20Static%20BSS-cyan.svg)]()
[![Latency: Sub-3us](https://img.shields.io/badge/Latency-2.61%20%CE%BCs%20%2F%20step-green.svg)]()

> **The Model Context Protocol (MCP) Cognitive Governor for Autonomous AI Agents (Google Antigravity, Claude Desktop, Cursor, Windsurf).**  
> Powered directly by the **canonical C11 engine** (`libmicro_actinf.dll` / `libmicro_actinf.so`) via zero-copy `ctypes` FFI bindings.

---

## 📑 Overview
Standard AI agents suffer from context drift, repetitive debugging loops, and conversational token waste.  
The **Micro-ActInf MCP Server** acts as an active **Bayesian Governor**:
1. **Pre-Execution Observation & Policy Lock:** Enforces mathematically optimal behavior (`PRAGMATIC_EXECUTE`, `AUDIT_DIAGNOSE`, `EPISTEMIC_EXPLORE`, `CONVERGE_CONCLUDE`).
2. **Hard Action Safety Gating:** Intercepts dangerous or repetitive tool calls *before* execution (`ALLOW`, `MODIFY`, `ASK_CONFIRMATION`, `DENY`).
3. **Credit Assignment Learning:** Adapts prior preferences $C(o)$ and transitions $B(u)$ dynamically using online Dirichlet updates.

---

## 🛠️ The 6 MCP Governance Tools

| Tool Name | Parameters | Verdict / Output | Description |
| :--- | :--- | :--- | :--- |
| `actinf_observe` | `obs_type` (str), `context_attributes` (dict, opt) | `dominant_regime`, `confidence`, `shannon_entropy_nats` | Ingests observation and updates belief simplex. |
| `actinf_get_state` | *None* | Full simplex, entropy velocity, progress index, backend info | Returns full internal POMDP state. |
| `actinf_prescribe_policy` | *None* | `prescribed_action`, `directive`, `action_index` | Minimizes multi-step EFE and prescribes policy. |
| `actinf_evaluate_action` | `proposed_tool`, `action_type`, `risk_level`, `confidence_threshold` | `verdict` (`ALLOW`, `MODIFY`, `ASK_CONFIRMATION`, `DENY`) | Pre-execution safety gate preventing loops and high-risk actions. |
| `actinf_record_outcome` | `action`, `outcome_obs`, `success`, `progress_delta` | `status`, `progress_index`, `updated_preference` | Applies online Dirichlet credit assignment. |
| `actinf_reset` | *None* | `status: reset_successful` | Resets belief simplex to uniform prior. |

---

## ⚙️ Antigravity & Claude Desktop Configuration

### Google Antigravity Setup
Add to `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "micro-actinf": {
      "command": "python",
      "args": [
        "c:/Users/hassan/Desktop/نورون/micro-actinf/mcp_server/server.py"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Anthropic Claude Desktop Setup
Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):
```json
{
  "mcpServers": {
    "micro-actinf": {
      "command": "python",
      "args": [
        "/path/to/micro-actinf/mcp_server/server.py"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

---

## 🏗️ Architecture & Single Source of Truth

```
[ Antigravity / Claude / Cursor ]
                │
         (MCP JSON-RPC 2.0)
                │
                ▼
      mcp_server/server.py
                │
                ▼
      mcp_server/libactinf.py (ctypes FFI)
                │
                ▼
      libmicro_actinf.dll / .so (Canonical C11 Engine)
      ├── 20.62 KB Static Footprint
      ├── Sub-3us Latency
      └── Zero Heap Malloc
```

---

## 🇮🇷 راهنمای فارسی راه‌اندازی سرور MCP

### ویژگی‌های کلیدی سرور MCP:
۱. **موتور اصیل C11:** تمامی محاسبات ریاضی، فیلتر بیزین، آنتروپی شانون و برنامه‌ریزی افق چندمرحله‌ای مستقیماً توسط کتابخانه C کامپایل‌شده اجرا می‌شوند.
۲. **شش ابزار حاکمیتی استاندارد:** کنترل کامل چرخه حیات ایجنت از دریافت ورودی تا اعتبارسنجی ابزارها و یادگیری برخط.
۳. **پروتکل دوگانه:** پشتیبانی خودکار از کتابخانه رسمی FastMCP و فال‌بک مستقل استاندارد پایتون (بدون نیاز به نصب پکیج‌های حجیم).

### دستور اجرای آزمایشی سرور:
```bash
python mcp_server/server.py
```
سرور به صورت خودکار آماده دریافت پیام‌های JSON-RPC 2.0 روی `stdin` و ارسال پاسخ روی `stdout` خواهد بود.
