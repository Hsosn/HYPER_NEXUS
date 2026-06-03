# Hyper Nexus

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Hsosn/HYPER_NEXUS)](https://github.com/Hsosn/HYPER_NEXUS/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Hsosn/HYPER_NEXUS)](https://github.com/Hsosn/HYPER_NEXUS/issues)
[![Docker ready](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

**An autonomous AI agent platform with adaptive reasoning, 165 built-in tools, long-term semantic memory, a pure-Python 3D engine, an ML/AI engineering suite, a sandboxed virtual computer, and a real-time Web UI — all in a single Python codebase.**

> One system. One loop. Infinite capabilities.

**Made by [VESKO LABS](https://veskolabs.com)**

---

## Why Hyper Nexus?

If you've used AutoGPT, OpenHands, CrewAI, or LangChain agents, you already know the pattern: chain calls to an LLM, optionally call some tools, hope the result is good. Hyper Nexus is built around a different idea — **the agent should get measurably better the more you use it, and it should reason laterally the way humans actually solve hard problems.**

Six things you won't find together in any other open-source agent:

1. **Cross-domain creative reasoning (ADHD module)** — when a task is non-trivial, the engine fires the problem across 8 knowledge domains (biology, physics, music, economics, architecture, game theory, neuroscience, military) in parallel, then synthesises analogies back into the system prompt. No other open-source agent does this.
2. **Self-improvement on a 30-minute heartbeat** — every half hour, the agent reviews its own execution logs, mines successful tool chains, clusters failure patterns, generates heuristic fixes, and writes them to a strategy table that gets injected into future prompts. You can watch it happen in real time.
3. **Self-reflective post-response loop** — every response gets a cheap secondary LLM call that scores it for accuracy, completeness, and confidence. Low-scoring responses trigger automatic re-planning.
4. **Dual-layer long-term memory with Ebbinghaus-style forgetting** — a flat semantic store (embedding-based recall) plus a hierarchical tree (chunked, sealed, cross-linked). Idle memories decay in importance; frequently-referenced ones are promoted; duplicates are merged.
5. **100% self-hosted, MIT licensed, zero cloud calls** — runs on a laptop, a Raspberry Pi 5, or a $5/month VPS. API keys stay on your machine. The vision model (Florence-2) runs on-device. The embedding model (MiniLM) runs on-device. Nothing phones home.
6. **165 tools + 25 skill packs + 100+ integrations, in one repo** — the equivalent of stacking LangChain + OpenHands + n8n + Zapier into a single `pip install`.

**When to pick something else:** if you only need a chatbot, use the OpenAI API directly. If you need a production-grade orchestrator with a hardened eval harness, use LangGraph. Hyper Nexus is for the person who wants an agent that grows on them, that they can read end-to-end, and that they can run on a Friday night without a credit card.

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set your API key (OpenRouter, OpenAI, or any provider)
echo "OPENROUTER_API_KEY=sk-or-v1-your-key" > .env

# 3. Start the server
python run.py
```

**Or use the platform-specific startup scripts (handles Redis + Celery automatically):**

| Platform | Command |
|----------|---------|
| **Windows (CMD)** | `start.bat` |
| **Windows (PowerShell)** | `.\start.ps1` |
| **Linux / macOS** | `python run.py` |

Open **http://127.0.0.1:8765** in your browser.

> **Note:** Vision is handled entirely on-device using Microsoft Florence-2 — no external vision API key needed. The ~900 MB model downloads automatically on first image analysis.
>
> **Note:** Memory embeddings are handled on-device by default using `sentence-transformers` (all-MiniLM-L6-v2, ~80MB). No external embedding API key is required. The model downloads automatically on first memory use.

---

## Table of Contents

- [What Is Hyper Nexus?](#what-is-hyper-nexus)
- [Capabilities at a Glance](#capabilities-at-a-glance)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Detailed Module Reference](#detailed-module-reference)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## What Is Hyper Nexus?

Hyper Nexus is a **self-hosted, autonomous AI agent platform** that goes far beyond a simple chatbot. At its core is the **Nexus Framework** — a single adaptive reasoning loop that automatically selects the best strategy for every step based on task complexity.

**Who is it for?**
- **Developers** who want an AI coding assistant that can run shell commands, manage Git/GitHub, browse the web, and build full-stack applications autonomously.
- **AI/ML Engineers** who need a platform that can train PyTorch models, fine-tune LLMs with LoRA/QLoRA, run RLHF pipelines, design transformer architectures, and optimize models for deployment.
- **3D Artists & Designers** who want procedural 3D mesh generation, CSG boolean operations, skeletal rigging, animation, path-tracing rendering, and physics simulation — all in pure Python with no GPU.
- **Researchers** who need a system with deep research capabilities, web monitoring, file watching, journaling, and goal tracking with a self-improvement feedback loop.
- **Automation Engineers** who need integrations with 100+ external services (Slack, GitHub, Notion, Salesforce, etc.), MCP server support, and a sandboxed virtual computer for GUI automation.

---

## Capabilities at a Glance

| Category | Capabilities |
|----------|-------------|
| **Adaptive Reasoning** | Nexus Framework — single adaptive loop with action-observation cycles, step-by-step chain reasoning, branch exploration, search, stuck detection, loop detection, context compression, and sub-agent delegation |
| **Creative Reasoning (ADHD)** | Cross-domain analogy engine — fires tasks across 8 knowledge domains (biology, physics, music, economics, architecture, game theory, neuroscience, military) to find non-obvious solutions. Hyperfocus tracking, serendipity injection, stream bleeding, and adaptive domain utility learning |
| **Long-Term Memory** | Dual-layer: flat semantic memory (embedding-based recall with importance scoring) + hierarchical memory tree (chunked, sealed, entity-linked, cross-root relationships). Default embedding: on-device `sentence-transformers` (all-MiniLM-L6-v2). Automatic fact capture, reflective learning, consolidation, deduplication, and forgetting-curve pruning |
| **File Write** | Atomic writes with per-path locks, binary auto-detection, Windows long-path support. All operations strictly confined to `data/workspace/<project>/` |
| **Sub-Agent Delegation** | `delegate_task` (single sub-agent) and `delegate_batch` (up to 5 parallel sub-agents with retry, quality gating, and cleanup) |
| **Presentation Tool** | 24 professional themes organized by category (business, tech, medical, education, marketing, space, ocean, luxury, etc.) |
| **MCP Support** | Model Context Protocol — connect any MCP-compatible server and auto-register its tools |
| **3D Engine** | Pure Python procedural 3D — mesh creation, CSG boolean operations (union/subtract/intersect), skeletal rigging, IK solvers (FABRIK, CCD), animation curves & blending, PBR materials, path tracing global illumination, scene graph, physics simulation |
| **ML/AI Engineering** | PyTorch model training (15 `pt_*` tools), transformer architecture design, CV workbench, NLP workbench, LLM fine-tuning (LoRA, QLoRA, AdaLoRA, RLHF), GAN studio, distributed training, model optimization (pruning, quantization, ONNX export), data pipeline management |
| **Virtual Computer** | Docker-based sandboxed desktop with VNC streaming, vision-loop GUI automation, file transfer, browser and IDE pre-installed |
| **Self-Improvement** | Multi-module learning system — experience replay, meta-learning, tool affinity analysis, failure pattern recognition, strategy injection, quality feedback loops, memory consolidation, user satisfaction detection, plus real-time failure learning |
| **Real-Time Web UI** | WebSocket streaming chat, thinking visualization, complexity indicator, tool browser, memory viewer, goals tracker, settings editor, metrics dashboard, live browser preview, VM viewer, notifications, workspace browser, activity feed, MCP manager, integration manager |
| **Security** | JWT authentication, API key encryption at rest (Fernet), rate limiting (burst + sustained), strict workspace sandboxing, security headers (CSP, X-Frame-Options), role-based access control (admin/user/viewer) |
| **Environment Awareness** | Runtime OS detection, capability probing, anomaly detection (stale goals, quality decline, low disk space, missing API keys), self-awareness context injection into every reasoning loop |
| **Task Scheduling** | Celery + Redis task queue (Windows: use `--pool=solo`) with 10 scheduled tasks, natural-language cron scheduling, background agent execution, event bridge between workers and main process |

---

## Architecture Overview

```
hyper-nexus/
├── run.py                          # Entry point — starts Uvicorn with DB init (host 127.0.0.1, port 8765)
├── docker-compose.yml              # Optional: Redis 7 container (Celery broker + event bus)
├── docker/                         # Database init scripts
├── requirements.txt                # Python dependencies
├── start.bat / start.ps1           # Windows startup scripts
│
├── nexus/                          # ═══ CORE BACKEND ═══
│   ├── config.py                   # Settings: env var precedence + Fernet encryption
│   ├── celery_app.py               # Celery app with worker lifecycle hooks
│   ├── events.py                   # Async event bus (pub/sub, 500-event rolling history)
│   ├── environment.py              # Runtime awareness: OS, capabilities, anomalies
│   ├── auth.py                     # JWT auth, rate limiting, user/role management
│   ├── notifier.py                 # Notification creation + broadcasting
│   ├── heartbeat.py                # In-process scheduled tasks (10 jobs)
│   │
│   ├── api/                        # FastAPI app: REST + WebSocket
│   │   ├── server.py               # Main server (~4,300 lines, 119 routes)
│   │   ├── browser_routes.py       # Live browser preview (WebSocket streaming)
│   │   └── vm_routes.py            # Virtual computer (REST + WebSocket)
│   │
│   ├── core/
│   │   ├── llm.py                  # Multi-provider LLM client with circuit breaker
│   │   ├── llm_cache.py            # Response cache with semantic hashing
│   │   └── vision_local.py         # On-device vision (Florence-2, no external API)
│   │
│   ├── reasoning/
│   │   ├── engine.py               # Nexus Framework adaptive loop
│   │   └── adhd_module.py          # ADHD cross-domain creative reasoning (942 lines)
│   │
│   ├── memory/
│   │   ├── database.py             # SQLite async layer (~2,100 lines)
│   │   ├── memory.py               # Flat semantic memory with embedding recall
│   │   ├── memory_tree.py          # Hierarchical tree-based long-term memory
│   │   └── tree_db.py              # SQLite backing for memory tree
│   │
│   ├── tools/
│   │   ├── registry.py             # Tool registry: middleware, cache, rate-limit (~1,900 lines)
│   │   └── builtin/                # 32 modules providing 165 tools
│   │
│   ├── tasks/
│   │   ├── agent_tasks.py          # Background agent execution
│   │   ├── scheduler_tasks.py      # Scheduled/cron tasks with NL scheduling
│   │   ├── event_bridge.py         # Redis pubsub: worker→main event relay
│   │   └── task_state.py           # Task checkpoint/resume
│   │
│   ├── self_improve/               # Self-improvement system
│   │   ├── __init__.py             # Main improvement pipeline (run_improvement_pipeline)
│   │   ├── predict.py              # Tool outcome prediction and pattern extraction
│   │   └── realtime.py             # Real-time failure learning and fix generation
│   │
│   ├── virtual_computer/
│   │   ├── container.py            # Docker lifecycle, VNC, vision loop
│   │   └── setup-vm.sh             # VM provisioning script
│   │
│   └── watchers/
│       ├── file_watcher.py         # File system change detection
│       └── web_monitor.py          # Web page monitoring
│
├── nexus3d/                        # ═══ 3D ENGINE (pure Python) ═══
│   ├── mesh/                       # Mesh representation, OBJ I/O, primitives
│   ├── csg/                        # CSG boolean ops (BSP trees)
│   ├── rigging/                    # Skeletal rigging, skin weights, IK solvers
│   ├── animation/                  # Animation curves, tracks, clips, blending
│   ├── materials/                  # PBR materials, procedural textures
│   ├── rendering/                  # Path tracer + headless rasterizer
│   ├── scene/                      # Scene graph with hierarchical transforms
│   ├── math3d/                     # Vectors, quaternions, matrices, IK
│   ├── cinematic/                  # Camera: DoF, motion blur, shake
│   ├── api/                        # FastAPI server for 3D operations
│   └── utils/
│
├── nexus3d_skills/                 # ═══ 12 AGENT-TRIGGERABLE 3D SKILLS ═══
│
├── nexus_ml_skills/                # ═══ 13 AGENT-TRIGGERABLE ML/AI SKILLS ═══
│
└── webui/                          # ═══ FRONTEND (vanilla JS SPA) ═══
    ├── index.html                  # Main entry point
    ├── css/                        # sci-fi theme + panel styles
    └── js/                         # modules: ws, state, panels, utils
```

**Key architectural features:**

- **Zero-extras startup:** `python run.py` boots straight into SQLite + asyncio heartbeat + the static SPA. Docker/Redis/Celery are optional optimizations.
- **Real-time WebSocket streaming:** Every `emit()` call streams events to all connected browsers. The WebUI updates in real time — thoughts, tool calls, tool results, errors, self-improvement events.
- **Event-driven architecture:** The async event bus (`nexus/events.py`) decouples all subsystems. The reasoning engine, tool system, watchers, self-improvement, and WebSocket relay all communicate through it.
- **Session-scoped engines:** Each chat session gets its own `ReasoningEngine` instance cached in `server.py`, with its own memory, context, and tool boost state.
- **Strict workspace confinement:** All file operations (read/write/list/delete) are blocked from escaping `data/workspace/`. The system prompt instructs the agent to nest every project in a named subfolder (e.g. `data/workspace/my_api/`).

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11+ | Required for all backend code |
| **LLM API Key** | — | OpenRouter, OpenAI, Together AI, NVIDIA NIM, or any OpenAI-compatible endpoint |

**Optional dependencies:**

| Dependency | Purpose |
|-----------|---------|
| **Docker** | Redis (Celery broker), and the optional virtual computer container |
| **Playwright** (`playwright install chromium`) | Browser automation and live browser preview |
| **FFmpeg** (system binary) | Media processing — convert, trim, GIF creation |
| **PyTorch** (included in requirements.txt) | ML/AI training, transformers, neural architecture design |

**Vision dependencies (auto-detected at startup):**

Hyper Nexus uses **Microsoft Florence-2** as its default vision engine — entirely on-device, no external API calls. All dependencies are included in `requirements.txt`:

| Package | Role |
|---------|------|
| `transformers` | HuggingFace model loading and inference |
| `Pillow` | Image loading and processing |
| `einops` | Tensor reshaping (Florence-2 architecture) |
| `timm` | PyTorch Image Models (vision backbone) |
| `accelerate` | Device placement and optimization |
| `sentencepiece` | Tokenizer support |
| `protobuf` | Model config serialization |

On first use of image analysis, the Florence-2 model (~900 MB) is downloaded from HuggingFace and cached locally. All subsequent analysis runs entirely on-device.

---

## Installation

### 1. Clone & Enter the Project

```bash
git clone https://github.com/your-org/hyper-nexus.git
cd hyper-nexus
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs: FastAPI, Uvicorn, aiosqlite, Celery, Redis, websockets, httpx, BeautifulSoup, python-docx, python-pptx, cryptography, authlib, plus:
- **PyTorch ecosystem** — `torch`, `torchvision`, `torchaudio`, `scikit-learn`, `pandas`
- **NLP/Transformers** — `transformers`, `tokenizers`, `einops`, `timm`
- **Reinforcement Learning** — `gymnasium`, `stable-baselines3`
- **Model Optimization** — `onnx`, `onnxruntime`
- **Local Vision Engine** — Florence-2 (via transformers, einops, timm, Pillow)

**Optional — browser automation:**
```bash
playwright install chromium
```

**Optional — FFmpeg:**
| OS | Command |
|----|---------|
| macOS | `brew install ffmpeg` |
| Linux | `sudo apt install ffmpeg` |
| Windows | `choco install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org) |

**Optional — Docker (for Redis, virtual computer):**
```bash
docker compose up -d
```

### 3. Configure Your LLM API Key

Create a `.env` file in the project root:

```bash
# OpenRouter (recommended — access to all major models)
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**Alternative providers:**
```bash
OPENAI_API_KEY=sk-proj-your-key-here
TOGETHER_API_KEY=your-key-here
NVIDIA_API_KEY=your-key-here
CUSTOM_API_KEY=your-key-here
```

You can also configure the API key through the Web UI **Settings** panel. Keys entered through the UI are encrypted at rest using Fernet symmetric encryption. API keys from environment variables take highest precedence and are **never** written to disk.

### 4. Start the Server

```bash
python run.py
```

Or use the included platform-specific scripts:
```bash
# Windows (Command Prompt)
start.bat

# Windows (PowerShell)
.\start.ps1
```

The startup process:
1. Verifies local vision (Florence-2) dependencies
2. Initializes SQLite database with schema auto-creation
3. Starts the asyncio heartbeat loop for periodic tasks
4. Starts the Nexus server on **http://127.0.0.1:8765**

Open your browser and navigate to **http://127.0.0.1:8765**.

---

## Detailed Module Reference

---

## Core System — `nexus/`

### Reasoning Engine

**File:** `nexus/reasoning/engine.py` (2,805 lines)

The **Nexus Framework** is the platform's adaptive reasoning core — a single loop that automatically selects the right reasoning strategy per step based on structural signals in the user's message.

**How it works:**

1. **Input Processing** — Receives user message + session ID. Retrieves long-term memories and recent conversation history.

2. **Complexity Assessment** (`_assess_complexity`) — Analyzes the message for structural signals:
   - Multi-task connectors ("and then", "after that", "first...then...finally")
   - Numbered steps and bullet patterns
   - High-complexity intent phrases ("research and write", "build a complete", "end-to-end")
   - Action verb count (architect, orchestrate, synthesize, deploy)
   - Message length and question depth
   - Returns `"low"`, `"medium"`, or `"high"`

3. **Tool Prefiltering** (`_prefilter_tools`) — Filters 165 tools down to task-relevant ones using keyword scoring. For high-complexity tasks, force-includes delegation tools (`delegate_task`, `delegate_batch`). Results cached with 30s TTL.

4. **Planning Injection** — For complex tasks, injects a planning instruction block into the system prompt with step-by-step execution protocol and verification checkpoints.

5. **ADHD Creative Injection** — For medium+ complexity tasks, fires the ADHD cross-domain reasoning module to inject creative analogies and tool suggestions into the system prompt.

6. **Core Reasoning Loop** — Iterative `thought → action → observation` cycle with:
   - **Stuck Detection** — Same tool+args repeated 3 times (`loop_detection_threshold`)? Forces a different approach
   - **Loop Detection** — Too many iterations? Forces completion with summary
   - **Hallucination Detection** — Regex patterns catch fabricated tool results, future-tense descriptions, and fake references
   - **Context Compression** — Adaptively truncates old history when approaching token limits
   - **Self-Awareness Injection** — System state, memory stats, active goals, environment context
   - **Tool Call Leak Stripping** — Multi-pass stripping removes DSML/XML markup, function-call text, standalone tool names, bracketed references, backticked names, JSON blocks, Python code leaks, and YAML-style tool descriptions from visible assistant content

7. **Sub-Agent Delegation** — Can spawn sub-agents for parallel task execution via `delegate_task` and `delegate_batch`. For high-complexity tasks, delegate tools are force-included.

8. **Memory Integration** — After each cycle, important information is saved to memory. Factual memories are auto-captured from user statements. Reflections may be generated from conversation context.

**Reasoning strategies (applied per-step, not per-task):**

| Strategy | Applied When |
|----------|-------------|
| **Action-Observation Loop** | All tasks involving tool use |
| **Step-by-Step Chain** | Analysis, explanation, comparison |
| **Branch Exploration** | Creative tasks, brainstorming, debugging |
| **Search & Simulation** | Complex multi-step planning, deep research |

### ADHD Cross-Domain Reasoning Module

**File:** `nexus/reasoning/adhd_module.py` (942 lines)

A creative reasoning booster that mimics the ADHD brain's superpower: hyper-connecting seemingly unrelated knowledge domains.

**How it works:**

1. **Problem Type Extraction** — Strips the task to its structural type: search, optimization, coordination, flow, memory, pruning, parallelism, or deadlock
2. **Cross-Domain Fire** — Fires the problem type across 8 knowledge domains simultaneously (biology, physics, music, economics, architecture, game theory, neuroscience, military), each domain having 8 pre-written reasoning patterns
3. **Stream Building** — Spawns N parallel reasoning streams seeded from the highest-relevance domain+pattern matches
4. **Stream Bleeding** — Cross-pollinates the strongest stream into the weakest one, blending actual reasoning content
5. **Hyperfocus Detection** — Locks onto any domain that exceeds the hyperfocus threshold, with escalating confidence and burnout prevention
6. **Serendipity Injection** — Optionally injects a wild-card domain with preference for high-fatigue (recently abandoned) domains to escape local optima
7. **Synthesis** — Collapses everything into a prompt block with analogies, hyperfocus hint, serendipity pattern, and tool suggestions

**Adaptive learning features:**
- **Domain utility tracking** via exponential moving average (decay=0.5)
- **Domain fatigue** — temporary dip after use, simulating interest-based fading
- **Hyperfocus streak** — consecutive fires in the same domain escalate confidence
- **Racing thoughts** — random domain derail to simulate tangential thinking (15% chance per fire)
- **Novelty tracker** — tracks fires since each domain was last used
- **Cache** — 32-entry LRU cache with 30s TTL keyed by task+complexity

**When it fires:** Configurable via `enable_adhd_reasoning` (default True) and `adhd_complexity_min` (default "medium"). The engine refreshes ADHD context on every reasoning loop iteration for fresh analogies.

### LLM Provider Abstraction

**Files:** `nexus/core/llm.py`, `nexus/core/llm_cache.py`

A clean abstraction layer over multiple LLM providers:

- **Provider Support:** OpenRouter (default), OpenAI, Together AI, NVIDIA NIM, Groq, custom OpenAI-compatible endpoints (Ollama, LM Studio, Azure OpenAI)
- **Circuit Breaker:** Automatically detects failures and stops calling a provider after configurable thresholds. Periodic health checks for recovery detection.
- **Cost Tracking:** Per-request token counting and USD cost estimation per model, with cumulative cost caps per task (default $2.00)
- **Model Routing:** Separate model configurations for reasoning (`default_model`), memory (`memory_model`), and vision (Florence-2 local)
- **Rate Limiting:** Client-side rate limiting to stay within provider API limits
- **Retry Logic:** Exponential backoff with jitter on transient failures
- **Response Cache:** Caches deterministic completions (temperature=0) with configurable TTL (default 15s) and max size (128 entries) using semantic hashing

### Memory System

**Files:** `nexus/memory/database.py` (1,764 lines), `nexus/memory/memory.py`, `nexus/memory/memory_tree.py`, `nexus/memory/tree_db.py`

Dual-layer long-term memory architecture:

**Layer 1 — Flat Semantic Memory (`memory.py`):**
- **Embedding-Based Storage:** Memories embedded using configurable embedding model (default: on-device `sentence-transformers` `all-MiniLM-L6-v2`, small 384-dim CPU-friendly model) with cosine similarity search
- **Memory Types:** `factual`, `semantic`, `episodic`, `procedural`, `profile`, `preference`
- **Automatic Fact Capture:** Scans user input for self-referential statements ("I am", "I like", "I work at") and stores as factual memories
- **Importance Scoring:** 0.0–1.0 scale; higher-importance memories preferentially retained and recalled
- **Cross-Session Recall:** Retrieves K most relevant memories for context injection in every conversation (configurable `memory_retrieval_k`, default 6)
- **Reflective Learning:** Generates reflections from conversation history stored as procedural memories
- **Consolidation:** Periodic deduplication, pruning of stale entries (>90 days, <0.2 importance), promotion of frequently-accessed memories

**Layer 2 — Hierarchical Memory Tree (`memory_tree.py`):**
- Content chunked into nodes organized under roots (documents/conversations)
- Node states: `buffered` (new) → `sealed` (after age threshold, typically 24h)
- **Daily digests** summarize each day's activity
- **Entity extraction and cross-linking** connects related content across roots
- **Topic trees** group related content across different sources
- Background jobs process summarization and sealing periodically

**Database Layer (`database.py`):**
- SQLite via `aiosqlite` (async) with single-writer connection pattern
- Schema auto-creation on `init()` — tables: sessions, messages, memories, goals, tasks, tool_executions, tool_profiles, notifications, file_watches, web_monitors, nl_schedules, integrations, triggers, memory_tree_*, quality_scores, task_checkpoints, and more
- All queries use `$N` positional parameter syntax
- Full-text search via SQLite LIKE

### Tool System

**File:** `nexus/tools/registry.py` (1,612 lines) + 32 modules in `nexus/tools/builtin/` providing **165 tools**

**Tool Registry — Production-Grade Execution Engine:**

- **Registration:** `@tool` decorator with name, description, JSON Schema parameters, risk level, category, cache settings, rate limits, dependencies, timeout, and version tags
- **Middleware Pipeline:** Pre/post/error middleware with priority ordering and abort capability
- **Circuit Breaker:** Auto-disables tools after consecutive failures, auto-recovers after cooldown
- **Rate Limiting:** Sliding-window rate limiter per tool
- **Parameter Alias Normalization:** Auto-maps aliases ("file" → "path", "data" → "content") case-insensitively
- **JSON Schema Validation:** Full parameter validation before execution
- **LRU Result Cache:** Per-tool TTL, O(1) eviction via `OrderedDict`
- **Timeout Enforcement:** Per-tool timeout with asyncio cancellation
- **Profiling:** Per-tool profiles track total calls, success/failure counts, latency percentiles (p50/p90/p99), per-parameter-hash success rates, error categorization, and effectiveness score
- **Semantic Discovery:** TF-IDF cosine similarity engine for finding tools by natural language query
- **Composition:** Sequential ("pipe"), parallel, and fan-out strategies
- **Event Emission:** Every execution emits events for real-time WebUI updates

**All registered tools (165):**

| Module | Tools |
|--------|-------|
| `file_tools.py` | `file_read`, `file_write`, `file_list`, `file_search`, `file_delete`, `project_read_context`, `project_write_context` |
| `code_tools.py` | `python_exec` |
| `shell_session.py` | `shell_run`, `shell_command`, `shell_attach`, `shell_sessions`, `shell_reset` |
| `git_tools.py` | `git_init`, `git_clone`, `git_status`, `git_diff`, `git_add`, `git_commit`, `git_log`, `git_branch`, `git_push`, `git_pull` |
| `github_tools.py` | `github_list_repos`, `github_list_issues`, `github_list_prs`, `github_create_issue`, `github_create_pr`, `github_commits`, `github_read_file`, `github_get_repo` |
| `web_tools.py` | `web_search`, `fetch_url` |
| `deep_research.py` | `deep_research` |
| `memory_tools.py` | `remember`, `recall_memories`, `forget_memories`, `memory_set_permanent`, `tree_ingest`, `tree_browse`, `tree_status`, `recent_events` |
| `goal_tools.py` | `add_goal`, `list_goals`, `update_goal_progress` |
| `journal_tools.py` | `journal_write`, `journal_read`, `journal_mood_summary` |
| `monitor_tools.py` | `watch_path`, `list_watches`, `remove_watch`, `monitor_url`, `improvement_log` |
| `system_tools.py` | `system_status`, `ip_info`, `http_request`, `text_utils`, `unit_convert`, `current_time`, `date_calc`, `random_choice`, `generate_uuid`, `calculator` |
| `delegate_tools.py` | `delegate_task`, `delegate_batch` |
| `mcp_tools.py` | `mcp_connect_server`, `mcp_disconnect_server`, `mcp_list_servers`, `mcp_call_tool`, `mcp_auto_connect`, `mcp_remove_server` |
| `vision_tools.py` | `image_understand` |
| `image_gen_tools.py` | `generate_image` |
| `browser_tools.py` | `browser_open`, `browser_click`, `browser_fill_form`, `browser_screenshot` |
| `email_tools.py` | `email_send`, `email_inbox`, `email_reply`, `email_search` |
| `presentation_tools.py` | `create_ppt`, `ppt_add_slide`, `list_ppt_templates` |
| `docx_tools.py` | `docx`, `list_docx_templates` |
| `ffmpeg_tools.py` | `ffmpeg_convert`, `ffmpeg_extract_audio`, `ffmpeg_create_gif`, `ffmpeg_merge_video`, `ffmpeg_trim` |
| `fullstack_tools.py` | `fullstack_scaffold`, `fullstack_status`, `fullstack_template` |
| `pytorch_tools.py` | `pt_model_create`, `pt_model_train`, `pt_model_predict`, `pt_model_evaluate`, `pt_model_save`, `pt_model_load`, `pt_model_info`, `pt_model_convert`, `pt_data_preprocess`, `pt_data_augment`, `pt_image_classification`, `pt_text_classification`, `pt_transfer_learning`, `pt_hyperparameter_tune`, `pt_training_monitor`, `pt_model_visualize`, `pt_ml_info`, `pt_status` |
| `ml_ai_skill_tools.py` | `ml_cv_workbench`, `ml_data_pipeline`, `ml_deep_learning_trainer`, `ml_distributed_training`, `ml_gan_studio`, `ml_llm_trainer`, `ml_model_optimizer`, `ml_neural_architect`, `ml_nlp_workbench`, `ml_peft_finetuning`, `ml_rl_lab`, `ml_rlhf_lab`, `ml_transformer_architect` |
| `nexus3d_tools.py` | `nexus3d_info`, `nexus3d_create_mesh`, `nexus3d_transform_mesh`, `nexus3d_csg_boolean`, `nexus3d_create_armature`, `nexus3d_solve_ik`, `nexus3d_animate_procedural`, `nexus3d_physics_simulate`, `nexus3d_raycast`, `nexus3d_material_library`, `nexus3d_cinematic_dof` |
| `virtual_computer_tools.py` | `vm_start`, `vm_stop`, `vm_restart`, `vm_destroy`, `vm_status`, `vm_execute`, `vm_screenshot`, `vm_vision_loop`, `vm_mouse_click`, `vm_mouse_move`, `vm_type_text`, `vm_press_key`, `vm_upload`, `vm_download`, `vm_install_package` |
| `integration_tools.py` | `list_available_integrations`, `list_connected_integrations`, `configure_integration`, `call_integration_api`, `integration_health_check`, `integration_webhook_info` |
| `tasks_tools.py` | `schedule_task`, `list_schedules`, `list_nl_schedules`, `remove_schedule`, `remove_nl_schedule`, `schedule`, `cancel_background_task`, `list_background_tasks` |

**Integration connectors (`integration_tools.py`, 112 services):**

**Communication:** Slack, Discord, Telegram, Twilio, SendGrid, Mailgun, Microsoft Teams
**Project Management:** GitHub, GitLab, Bitbucket, Linear, Jira, Asana, Monday, Trello, ClickUp, Notion, Basecamp, Confluence, Wrike
**CRM & Sales:** Salesforce, HubSpot, Pipedrive, Close, Zoho
**Cloud & DevOps:** AWS, AWS S3, Azure, Google Cloud, Cloudflare, DigitalOcean, Vercel, Netlify, Railway, Render, Fly.io, Heroku, Backblaze, Wasabi, Terraform, Ansible, Kubernetes, Docker, Docker Hub, GHCR, Harbor, Portainer, Jenkins, CircleCI, Azure DevOps, Cloudways, GoDaddy, Hostinger, Namecheap, Porkbun
**Monitoring:** Datadog, Sentry, PagerDuty, New Relic, Grafana, UptimeRobot, Pingdom, Statuspage, Let's Encrypt, ZeroSSL
**Storage:** Dropbox, Google Drive, Google Sheets, Firebase, Supabase
**Google:** Calendar, Drive, Sheets, Docs, Slides, Forms, Meet, Analytics, Ads, Search Console, YouTube, Google Play
**Microsoft:** Teams, Outlook, OneDrive, SharePoint, Azure Storage
**Design:** Figma, Canva, Adobe Creative Cloud
**Finance:** Stripe, PayPal, Lemon Squeezy, BigCommerce, Shopify, WooCommerce
**AI/ML:** OpenAI, Anthropic, Cohere, HuggingFace, Stability AI, Replicate, Google AI
**Marketing:** Amplitude, Mixpanel, Segment, PostHog, Heap, Hotjar, Plausible
**Data:** Airtable, PlanetScale, MongoDB Atlas
**Other:** IFTTT, Zapier, Make, n8n, App Store, YouTube, Webhook

### API & WebSocket Server

**Files:** `nexus/api/server.py` (1,957 lines), `nexus/api/browser_routes.py`, `nexus/api/vm_routes.py`

**`server.py`** — FastAPI application with:

- **REST Endpoints:** Chat, settings, tools, skills, memories, goals, integrations, users, system status — all with Pydantic validation. 144 routes across `server.py`, `browser_routes.py`, and `vm_routes.py`.
- **WebSocket Chat (`/ws`):** Real-time streaming with event-driven responses — the agent streams thoughts, tool calls, observations, and errors as they happen
- **Static File Mount:** Serves the WebUI SPA from `webui/` at the root path
- **Authentication Middleware:** JWT token validation with API key fallback
- **Rate Limiting:** Per-IP burst window (10s) + sustained window (minute), separate limits per endpoint type
- **CORS:** Configurable origins for cross-origin requests
- **Security Headers:** CSP, X-Frame-Options, X-Content-Type-Options on all responses
- **Graceful Shutdown:** DB pool close, heartbeat stop, LLM client shutdown

**`browser_routes.py`** — Live browser preview via WebSocket screenshot streaming
**`vm_routes.py`** — Virtual computer: REST lifecycle + VNC screen streaming over WebSocket

### Authentication & Security

**File:** `nexus/auth.py`

- **JWT Tokens:** Stateless HMAC-SHA256 tokens with 24-hour expiry
- **API Keys:** Fallback authentication via `X-API-Key` header (format: `nk-...`)
- **Password Hashing:** HMAC-SHA256 with per-instance salt
- **User Management:** Create, update, delete users with role-based access (admin/user/viewer)
- **WebSocket Auth:** Token validation via WebSocket query parameter
- **Encryption at Rest:** API keys encrypted with Fernet (AES-128-CBC), key stored separately in `data/.enc_key`
- **Environment Variable Priority:** Env vars take highest precedence, never written to `settings.json`
- **Workspace Sandboxing:** All file operations strictly confined to `data/workspace/<project>/` — absolute paths outside the workspace are rejected at the tool layer

### Self-Improvement System

**Directory:** `nexus/self_improve/`

A comprehensive learning system with a main `run_improvement_pipeline()` entry point plus real-time hooks. Pipeline runs on a configurable cycle (default: every 30 minutes via the heartbeat scheduler).

**Core files:**
- `__init__.py` — Main pipeline orchestration (`run_improvement_pipeline`, `_quality_feedback_loop`)
- `predict.py` — Tool outcome prediction, pattern extraction, success/failure predictors
- `realtime.py` — Real-time learning: instant fix generation from failure traces

**Capabilities (across the pipeline + real-time module):**

| Capability | What It Does |
|------------|-------------|
| **Experience Replay** | Stores high-scoring task executions as reusable trajectories |
| **Meta-Learning Engine** | Learns which reasoning approaches work per task type |
| **Tool Affinity Analysis** | Co-occurrence matrix, successful vs. failed tool chains, anti-patterns |
| **Failure Pattern Recognition** | NLP clustering of failures, generates countermeasures |
| **Strategy Injection** | Lifecycle-managed learned strategies injected into system prompt |
| **Quality Feedback Loop** | Monitors quality trends, triggers interventions on decline |
| **Memory Consolidation** | Deduplication, usage tracking, promotion, pruning |
| **Learning Rate & Decay** | Time-weighted confidence scores with evidence adjustment |
| **Quality Assessment** | Self-supervised quality scoring of responses |
| **Self-Supervised Learning** | Outcome prediction, pattern extraction, negative mining |
| **Real-Time Failure Learning** | Instant heuristic fix generation on tool failures |
| **User Satisfaction Detection** | Implicit feedback from conversation signals |
| **Prediction Engine** | Tool success prediction and failure risk scoring |

**Guardrails:** Only writes to database (never modifies code/configuration on disk), uses cheaper `memory_model` for LLM calls, self-learned memory cap (50 max), strategy content capped at 200 characters.

### Event Bus & Notifications

**Files:** `nexus/events.py`, `nexus/notifier.py`

- Async pub/sub event bus with 500-event rolling history
- Automatic dead subscriber cleanup
- Every `emit()` call streams to all connected WebSocket clients
- Notification system with DB persistence, read/unread status, and severity levels (info/warn/error)

### Environment Awareness

**File:** `nexus/environment.py`

Comprehensive runtime monitoring:
- OS detection, Python version, CPU count, uptime, memory/disk usage
- Capability detection: which API keys configured, which tools registered
- Filesystem awareness: workspace layout, directory contents
- Network context: Docker detection, deployed vs. local heuristic
- Anomaly detection: stale goals (7+ days), quality decline, memory bloat, low disk, missing API keys
- Self-awareness context string injected into system prompt

### Heartbeat & Task Queue

**Files:** `nexus/heartbeat.py`, `nexus/celery_app.py`, `nexus/tasks/`

The `heartbeat.py` module runs 10 scheduled tasks as in-process asyncio jobs (no Celery worker required). Celery + Redis is an optional optimization for distributed deployments.

| Task | Interval | Purpose |
|------|----------|---------|
| `check_data_integrity` | 5 min | SQLite health check |
| `process_scheduled_tasks` | 30 s | Execute due scheduled tasks |
| `process_nl_schedules` | 30 s | NL-defined schedule processing |
| `process_incomplete_goals` | 2 min | Check stalled goals |
| `file_watcher_tick` | 30 s | Poll watched directories |
| `web_monitor_tick` | 60 s | Monitor watched web pages |
| `self_improvement_run` | 30 min | Self-improvement pipeline |
| `environment_check` | 5 min | Environment anomaly detection |
| `health_summary` | 30 s | Health status events |
| `process_active_triggers` | 60 s | Process event triggers |
| `memory_curve_pruning` | 30 min | Forgetting-curve pruning |

**Event Bridge:** When Celery is enabled, Redis pubsub relays events from Celery workers to the main process, keeping the Web UI updated during background task execution.

### Virtual Computer

**Files:** `nexus/virtual_computer/container.py`, `nexus/virtual_computer/setup-vm.sh`

Docker-based sandboxed desktop environment:
- Container lifecycle: create, start, stop, restart, destroy via REST API
- VNC streaming to Web UI via WebSocket
- Vision-loop GUI automation: agent "sees" the VM screen and issues click/type/scroll commands
- File transfer: upload from host, download from container
- Pre-installed: Firefox, VS Code, Git, Python, Node.js
- Use cases: GUI testing, isolated web browsing, software installation experiments

### Watchers

**Files:** `nexus/watchers/file_watcher.py`, `nexus/watchers/web_monitor.py`

- **File Watcher:** Monitors directories for content changes (create/modify/delete) using content hashing
- **Web Monitor:** Periodically fetches URLs and compares content hashes for change detection

### Local Vision Engine

**File:** `nexus/core/vision_local.py`

On-device vision using **Microsoft Florence-2** (230M params) via HuggingFace Transformers:

- **Fully Local:** ~900 MB download once, all subsequent analysis offline
- **Automatic Task Detection:** Determines best Florence-2 task from prompt keywords: OCR, detailed caption, VQA, standard description
- **Dual API:** `analyze_image(file_path, prompt)` and `analyze_image_base64(b64_data, prompt)`
- **WebUI Integration:** Uploaded images auto-analyzed before being sent to LLM
- **Lazy Loading:** Model loaded on first use, cached globally
- **CPU-Optimized:** Runs on float32, GPU auto-detected if CUDA available

### Server Entry Point

**File:** `run.py`

Startup sequence:
1. Vision dependency check (Florence-2 packages)
2. SQLite database initialization with schema auto-creation
3. Heartbeat scheduler start (10 in-process tasks)
4. Uvicorn server on `127.0.0.1:8765` (host=`127.0.0.1`, port=`8765`, `reload=False`, `log_level="info"`) with WebSocket support
5. Graceful shutdown: DB pool close, heartbeat stop, LLM client shutdown

---

## 3D Engine — `nexus3d/`

A complete pure Python 3D engine with zero external GPU dependencies.

| Module | Description |
|--------|-------------|
| **Mesh** | Core `Mesh` class with vertices, faces, edges, normals, texcoords, OBJ I/O, subdivision, simplification |
| **Primitives** | Factory functions: cube, sphere (UV + icosphere), cylinder, cone, torus, plane, circle, arrow, grid, Suzanne monkey head |
| **CSG** | Constructive Solid Geometry with BSP trees — union, subtract, intersect. Architectural primitives (walls, windows, doors, stairs) |
| **Rigging** | `Bone` and `Armature` classes, skin weights (heat-map + nearest-vertex), humanoid skeleton factory |
| **Animation** | Curves (linear, bezier, stepped), tracks, clips, blending, mixing. Export to glTF, BVH, FBX |
| **Materials** | PBR material system: procedural textures (noise, checker, brick, wood, marble, Voronoi, gradient), UV mapping |
| **Path Tracer** | Monte Carlo path tracer with global illumination, BRDFs (Lambert, Phong, GGX), subsurface scattering, DoF, tone mapping |
| **Renderer** | Headless renderer with PyRender (OSMesa) support and software rasterizer fallback |
| **Scene Graph** | `SceneNode` hierarchy, local/world transforms, bounding box computation |
| **Math** | Vectors (2D/3D/4D), matrices (3x3/4x4), quaternions, IK solvers (FABRIK, CCD, 2-bone analytic), collision detection, physics bodies |
| **Camera** | DoF, exposure, motion blur, camera shake, spline-based camera paths |
| **API** | FastAPI server exposing all 3D operations as REST endpoints |
| **CLI** | Command-line interface for common 3D operations |

## 3D Skill Plugins — `nexus3d_skills/`

12 agent-triggerable skills:

| Skill | Description |
|-------|-------------|
| **Animation Director** | Character animation direction — motion sequencing, timing, easing, retargeting |
| **Architectural Toolkit** | Building generation with rooms, floors, windows, doors, staircases, roofs, facades |
| **Character Creator** | Humanoid/creature character generation with adjustable proportions |
| **Cinema Camera** | Cinematic camera rigging — dolly, crane, tracking, orbit, rule-of-thirds composition |
| **CSG Architect** | Architectural boolean carving for windows/doors, wall generation, room carving |
| **CSG Toolkit** | General CSG modeling — complex boolean operations, shape carving, model merging |
| **Material Lab** | PBR parameter tuning, procedural texture generation, material presets |
| **Motion Pipeline** | Motion capture processing — BVH import, retargeting, blending, looping |
| **Physics Lab** | Rigid bodies, constraints, collisions, gravity, simulation stepping |
| **Product Renderer** | Product visualization — turntable animation, studio lighting, reflections |
| **Scene Composer** | Scene layout — object placement, grouping, lighting setup, environment staging |
| **Studio Renderer** | Multi-pass rendering, denoising, compositing, output formatting |

## ML/AI Engineering Suite — `nexus_ml_skills/`

13 specialized skill modules:

| Skill | Description |
|-------|-------------|
| **Computer Vision Workbench** | Image classification, object detection (YOLO, Faster R-CNN), semantic segmentation (U-Net) |
| **Data Pipeline** | Data loading, preprocessing, augmentation (flip, rotate, crop, mixup), dataset splitting |
| **Deep Learning Trainer** | Full PyTorch training loop with optimizers, schedulers, early stopping, checkpointing |
| **Distributed Training** | Multi-GPU/multi-node training via PyTorch DDP with NCCL backend |
| **GAN Studio** | DCGAN, WGAN-GP, StyleGAN-like architectures with G/D loss tracking |
| **LLM Trainer** | Causal/masked LM training, dataset preparation, perplexity evaluation |
| **Model Optimizer** | Pruning (structured/unstructured), quantization, ONNX export, inference optimization |
| **Neural Architect** | ConvNet, transformer, MLP architecture design with FLOPs/parameter counting |
| **NLP Workbench** | Text classification, NER, sentiment analysis, language modeling |
| **RL Lab** | Reinforcement learning with Gymnasium environments and Stable-Baselines3 |
| **RLHF Pipeline** | Reward modeling, PPO training, preference data management |
| **Transformers Builder** | Custom transformer architecture design with attention variants |
| **LoRA Studio** | Parameter-efficient fine-tuning with LoRA, QLoRA, AdaLoRA |

---

## Configuration Reference

The configuration system (`nexus/config.py`) uses three-layer precedence:
1. **Environment variables** (highest)
2. **`settings.json`** (persisted to `data/settings.json`, editable via WebUI)
3. **Hardcoded defaults**

Secrets are encrypted at rest using Fernet when stored in `settings.json`. Environment-variable keys are never persisted.

Key configuration categories:

| Category | Settings |
|----------|---------|
| **Provider** | `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `NVIDIA_API_KEY`, `CUSTOM_API_KEY`, `CUSTOM_BASE_URL` |
| **Model** | `default_model`, `memory_model` |
| **Generation** | `temperature` (default 0.7), `max_tokens`, `top_p` |
| **Agent Identity** | `agent_name`, `personality`, `traits`, `communication_style` |
| **Reasoning** | `max_nexus_iterations` (default 9999), `adaptive_iterations` (default True), `loop_detection_threshold` (default 3), `loop_detection_fast` (default True) |
| **ADHD** | `enable_adhd_reasoning` (default True), `adhd_complexity_min` (default "medium"), `adhd_max_analogies` (default 3) |
| **Memory** | `enable_long_term_memory`, `memory_retrieval_k` (default 6), `embedding_model`, `enable_memory_tree` |
| **Speed** | `cache_llm_responses`, `parallel_tool_execution`, `fast_response_mode` |
| **Task Execution** | `max_tool_calls_per_task` (default 50), `max_cost_per_task_usd` (default 2.0), `max_task_duration_seconds` (default 1800) |
| **Security** | `enable_auth`, `cors_origins` |
| **Celery** | `enable_celery`, `redis_url` |
| **Self-Improvement** | `enable_self_improvement` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI + Uvicorn |
| **Database** | SQLite (via aiosqlite, async) |
| **Task Queue** | Celery + Redis (optional); in-process asyncio heartbeat by default |
| **WebSocket** | FastAPI WebSocket + custom event bus |
| **Frontend** | Vanilla JavaScript SPA (no framework) — 23 JS modules |
| **LLM Providers** | OpenRouter, OpenAI, Together AI, NVIDIA NIM, Groq, custom endpoints |
| **Local Vision** | Microsoft Florence-2 (HuggingFace Transformers) |
| **ML Training** | PyTorch, scikit-learn, transformers, ONNX |
| **3D Engine** | Pure Python (no GPU dependency) |
| **VM** | Docker + VNC + Flask inside container |
| **Browser Automation** | Playwright |
| **Templating** | python-pptx, python-docx |
| **Encryption** | Fernet (AES-128-CBC) |
| **Auth** | JWT (HMAC-SHA256) |

---

## License

Hyper Nexus is released under the **MIT License** — see [LICENSE](LICENSE) for full text. You are free to use, modify, and distribute this software, provided the original copyright and license notice are preserved.
