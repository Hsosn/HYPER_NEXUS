# Hyper Nexus

**An autonomous AI agent platform with adaptive reasoning, 40+ built-in tools, long-term semantic memory, a pure-Python 3D engine, an ML/AI engineering suite, a sandboxed virtual computer, and a real-time Web UI — all in a single Python codebase.**

> One system. One loop. Infinite capabilities.

**Made by [VESKO LABS](https://veskolabs.com)**

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

---

## What Is Hyper Nexus?

Hyper Nexus is a **self-hosted, autonomous AI agent platform** that goes far beyond a simple chatbot. At its core is the **Nexus Framework** — a single adaptive reasoning loop that automatically selects the best strategy for every step based on task complexity.

**Who is it for?**
- **Developers** who want an AI coding assistant that can run shell commands, manage Git/GitHub, browse the web, and build full-stack applications autonomously.
- **AI/ML Engineers** who need a platform that can train PyTorch models, fine-tune LLMs with LoRA/QLoRA, run RLHF pipelines, design transformer architectures, and optimize models for deployment.
- **3D Artists & Designers** who want procedural 3D mesh generation, CSG boolean operations, skeletal rigging, animation, path-tracing rendering, and physics simulation — all in pure Python with no GPU.
- **Researchers** who need a system with deep research capabilities, web monitoring, file watching, journaling, and goal tracking with a self-improvement feedback loop.
- **Automation Engineers** who need integrations with 80+ external services (Slack, GitHub, Notion, Salesforce, etc.), MCP server support, and a sandboxed virtual computer for GUI automation.

---

## Capabilities at a Glance

| Category | Capabilities |
|----------|-------------|
| **Adaptive Reasoning** | Nexus Framework — single adaptive loop with action-observation cycles, step-by-step chain reasoning, branch exploration, search, stuck detection, loop detection, context compression, and sub-agent delegation |
| **Creative Reasoning (ADHD)** | Cross-domain analogy engine — fires tasks across 8 knowledge domains (biology, physics, music, economics, architecture, game theory, neuroscience, military) to find non-obvious solutions. Hyperfocus tracking, serendipity injection, stream bleeding, and adaptive domain utility learning |
| **Long-Term Memory** | Dual-layer: flat semantic memory (embedding-based recall with importance scoring) + hierarchical memory tree (chunked, sealed, entity-linked, cross-root relationships). Default embedding: on-device `sentence-transformers` (all-MiniLM-L6-v2). Automatic fact capture, reflective learning, consolidation, deduplication, and forgetting-curve pruning |
| **File Write** | Atomic writes with per-path locks, binary auto-detection, Windows long-path support |
| **Sub-Agent Delegation** | `delegate_task` (single sub-agent) and `delegate_batch` (up to 5 parallel sub-agents with retry, quality gating, and cleanup) |
| **Presentation Tool** | 24 professional themes organized by category (business, tech, medical, education, marketing, space, ocean, luxury, etc.) |
| **MCP Support** | Model Context Protocol — connect any MCP-compatible server and auto-register its tools |
| **3D Engine** | Pure Python procedural 3D — mesh creation, CSG boolean operations (union/subtract/intersect), skeletal rigging, IK solvers (FABRIK, CCD), animation curves & blending, PBR materials, path tracing global illumination, scene graph, physics simulation |
| **ML/AI Engineering** | PyTorch model training, transformer architecture design, CV workbench, NLP workbench, LLM fine-tuning (LoRA, QLoRA, AdaLoRA, RLHF), GAN studio, distributed training, model optimization (pruning, quantization, ONNX export), data pipeline management |
| **Virtual Computer** | Docker-based sandboxed desktop with VNC streaming, vision-loop GUI automation, file transfer, browser and IDE pre-installed |
| **Self-Improvement** | Multi-module learning system — experience replay, meta-learning, tool affinity analysis, failure pattern recognition, strategy injection, quality feedback loops, memory consolidation, user satisfaction detection, plus real-time failure learning |
| **Real-Time Web UI** | WebSocket streaming chat, thinking visualization, complexity indicator, tool browser, memory viewer, goals tracker, settings editor, metrics dashboard, live browser preview, VM viewer, notifications, workspace browser, activity feed, MCP manager, integration manager |
| **Security** | JWT authentication, API key encryption at rest (Fernet), rate limiting (burst + sustained), workspace sandboxing, security headers (CSP, X-Frame-Options), role-based access control (admin/user/viewer) |
| **Environment Awareness** | Runtime OS detection, capability probing, anomaly detection (stale goals, quality decline, low disk space, missing API keys), self-awareness context injection into every reasoning loop |
| **Task Scheduling** | Celery + Redis task queue (Windows: use `--pool=solo`), natural-language cron scheduling, background agent execution, event bridge between workers and main process |

---

## Architecture Overview

```
hyper-nexus/
├── run.py                          # Entry point — starts Uvicorn with DB init
├── docker-compose.yml              # Optional: PostgreSQL 16 + Redis 7 containers
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
│   │
│   ├── api/
│   │   ├── server.py               # FastAPI app: REST + WebSocket chat (4302 lines)
│   │   ├── browser_routes.py       # Live browser preview (WebSocket streaming)
│   │   └── vm_routes.py            # Virtual computer (REST + WebSocket)
│   │
│   ├── core/
│   │   ├── llm.py                  # Multi-provider LLM client with circuit breaker
│   │   ├── llm_cache.py            # Response cache with semantic hashing
│   │   └── vision_local.py         # On-device vision (Florence-2, no external API)
│   │
│   ├── reasoning/
│   │   ├── engine.py               # Nexus Framework adaptive loop (20,329 lines)
│   │   └── adhd_module.py          # ADHD cross-domain creative reasoning (945 lines)
│   │
│   ├── memory/
│   │   ├── database.py             # SQLite async layer (aiosqlite, 2134 lines)
│   │   ├── memory.py               # Flat semantic memory with embedding recall
│   │   ├── memory_tree.py          # Hierarchical tree-based long-term memory
│   │   └── tree_db.py              # SQLite backing for memory tree
│   │
│   ├── tools/
│   │   ├── registry.py             # Tool registry: middleware, cache, rate-limit (2300+ lines)
│   │   └── builtin/                # 40+ tools across 30+ modules
│   │
│   ├── tasks/
│   │   ├── agent_tasks.py          # Background agent execution
│   │   ├── scheduler_tasks.py      # Scheduled/cron tasks with NL scheduling
│   │   ├── event_bridge.py         # Redis pubsub: worker→main event relay
│   │   └── task_state.py           # Task checkpoint/resume
│   │
│   ├── self_improve/               # Self-improvement system
│   │   ├── __init__.py             # Main 15-step improvement pipeline
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
├── ml_ai_skills/                   # ═══ 13 AGENT-TRIGGERABLE ML/AI SKILLS ═══
│
└── webui/                          # ═══ FRONTEND (vanilla JS SPA) ═══
    ├── index.html                  # Main entry point
    ├── css/                        # neon-dark theme + panel styles
    └── js/                         # 17 modules: ws, state, panels, utils
```

**Key architectural features:**

- **Zero-extras startup:** `python run.py` — SQLite (not PostgreSQL), embedded asyncio heartbeat (no Celery worker required), static SPA served by FastAPI. Docker/Redis/Celery are optional optimizations.
- **Real-time WebSocket streaming:** Every `emit()` call streams events to all connected browsers. The WebUI updates in real time — thoughts, tool calls, tool results, errors, self-improvement events.
- **Event-driven architecture:** The async event bus (`nexus/events.py`) decouples all subsystems. The reasoning engine, tool system, watchers, self-improvement, and WebSocket relay all communicate through it.
- **Session-scoped engines:** Each chat session gets its own `ReasoningEngine` instance cached in `server.py`, with its own memory, context, and tool boost state.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11+ | Required for all backend code |
| **LLM API Key** | — | OpenRouter, OpenAI, Together AI, NVIDIA NIM, or any OpenAI-compatible endpoint |

**Optional dependencies:**

| Dependency | Purpose |
|-----------|---------|
| **Docker** | PostgreSQL, Redis, and virtual computer containers |
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

**Optional — Docker (for Redis, PostgreSQL, virtual computer):**
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

**File:** `nexus/reasoning/engine.py` (20,329 lines)

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

3. **Tool Prefiltering** (`_prefilter_tools`) — Filters 40+ tools down to task-relevant ones using keyword scoring. For high-complexity tasks, force-includes delegation tools (`delegate_task`, `delegate_batch`). Results cached with 30s TTL.

4. **Planning Injection** — For complex tasks, injects a planning instruction block into the system prompt with step-by-step execution protocol and verification checkpoints.

5. **ADHD Creative Injection** — For medium+ complexity tasks, fires the ADHD cross-domain reasoning module to inject creative analogies and tool suggestions into the system prompt.

6. **Core Reasoning Loop** — Iterative `thought → action → observation` cycle with:
   - **Stuck Detection** — Same tool+args repeated? Forces a different approach
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

**File:** `nexus/reasoning/adhd_module.py` (945 lines)

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
- **Cost Tracking:** Per-request token counting and USD cost estimation per model, with cumulative cost caps per task
- **Model Routing:** Separate model configurations for reasoning (`default_model`), memory (`memory_model`), and vision (Florence-2 local)
- **Rate Limiting:** Client-side rate limiting to stay within provider API limits
- **Retry Logic:** Exponential backoff with jitter on transient failures
- **Response Cache:** Caches deterministic completions (temperature=0) with configurable TTL (default 15s) and max size (128 entries) using semantic hashing

### Memory System

**Files:** `nexus/memory/database.py` (2134 lines), `nexus/memory/memory.py`, `nexus/memory/memory_tree.py`, `nexus/memory/tree_db.py`

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
- Full-text search via PostgreSQL ILIKE (legacy) and SQLite LIKE

### Tool System

**File:** `nexus/tools/registry.py` (2300+ lines) + 30+ modules in `nexus/tools/builtin/` providing 40+ tools

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

**Core tools (always available):**

| Module | Tools | Purpose |
|--------|-------|---------|
| `basic_tools.py` | `calculate`, `get_time`, `get_date`, `system_info`, `echo`, `random_number` | Utility operations |
| `file_tools.py` | `file_read`, `file_write`, `file_list`, `file_search`, `file_delete`, `file_move`, `file_copy` | Full filesystem interaction with atomic writes, binary detection, chunked streaming, Windows long-path support |
| `web_tools.py` | `web_search`, `fetch_url`, `read_url` | DuckDuckGo search and page content extraction |
| `shell_session.py` | `create_session`, `run_command`, `read_session_output`, `list_sessions`, `close_session` | Persistent shell sessions with workspace sandboxing |
| `git_tools.py` | `git_status`, `git_diff`, `git_log`, `git_commit`, `git_checkout`, `git_branch`, `git_push`, `git_pull` | Git repository management |
| `github_tools.py` | 8 GitHub API tools | Issues, PRs, repos, workflows |
| `email_tools.py` | `email_send`, `email_read`, `email_list`, `email_search` | SMTP send + IMAP read/list/search |
| `research_tools.py` | `deep_research` | Iterative multi-query web research with synthesis |
| `memory_tools.py` | `memory_store`, `memory_search`, `memory_recall` | Long-term memory management |
| `goal_tools.py` | `goal_create`, `goal_list`, `goal_update`, `goal_complete`, `goal_delete` | Goal lifecycle management |
| `system_tools.py` | `get_process_info`, `get_environment_info`, `get_system_info` | System monitoring |
| `journal_tools.py` | 6 journal tools | Journal CRUD with mood tracking |
| `integration_tools.py` | 80+ connectors | OAuth-based external service connections |
| `mcp_tools.py` | `mcp_register_server`, `mcp_unregister_server`, `mcp_list_servers`, `mcp_call_tool` | MCP server lifecycle and tool invocation |
| `delegate_tools.py` | `delegate_task`, `delegate_batch` | Sub-agent spawn for parallel execution |
| `code_tools.py` | `execute_code`, `execute_python`, `execute_node`, `execute_bash` | Isolated code execution |
| `monitor_tools.py` | 7 monitoring tools | File/web watch management, self-improvement log |
| `vision_tools.py` | `image_understand` | Florence-2 local image analysis |
| `vision_loop.py` | `start_vision_loop`, `stop_vision_loop` | Continuous vision analysis loop |
| `image_gen_tools.py` | `generate_image`, `generate_image_variation` | AI image generation |

**Optional tools (require additional dependencies):**

| Module | Dependencies | Tools | Purpose |
|--------|-------------|-------|---------|
| `browser_tools.py` | Playwright | `browser_navigate`, `browser_click`, `browser_type`, `browser_screenshot`, etc. | Full browser automation |
| `fullstack_tools.py` | None | `create_fullstack_app` | Generate React/Vue + FastAPI/Express + DB apps |
| `presentation_tools.py` | python-pptx | `create_ppt`, `ppt_add_slide` (timeout=300s) | PowerPoint creation with charts, tables, themes |
| `docx_tools.py` | python-docx | `create_docx`, `docx_add_paragraph`, `docx_add_table` | Word document creation |
| `ffmpeg_tools.py` | FFmpeg binary | `ffmpeg_convert`, `ffmpeg_extract_audio`, `ffmpeg_create_gif`, etc. | Media processing |
| `nexus3d_tools.py` | nexus3d | 6 3D tools | Mesh creation, CSG, rendering, animation |
| `pytorch_tools.py` | PyTorch | `train_model`, `design_architecture`, `fine_tune`, `evaluate_model`, `export_model` | ML model training |
| `virtual_computer_tools.py` | Docker | 10 VM tools | Container lifecycle + GUI automation |
| `ml_ai_skill_tools.py` | Various ML deps | `load_ml_skill`, `execute_ml_skill` | ML/AI skill module bridge |
| `custom_loader.py` | None | `load_custom_tool`, `load_custom_skill` | User-defined tool/script loading |

**Integration connectors (`integration_tools.py`):**

**Communication:** Slack, Discord, Telegram, Twilio, SendGrid, Mailgun
**Project Management:** GitHub, GitLab, Linear, Jira, Asana, Monday.com, Trello, ClickUp, Notion, Basecamp, Redmine
**CRM & Sales:** Salesforce, HubSpot, Zendesk, Freshdesk, Intercom, Pipedrive, Close, Help Scout
**Cloud & DevOps:** AWS (S3, EC2, Lambda, DynamoDB, SQS, SNS), Google Cloud, Cloudflare, DigitalOcean, Vercel, Netlify, Railway, Render, Pulumi, Terraform Cloud
**Monitoring:** Datadog, Sentry, PagerDuty, New Relic, Grafana, Prometheus, UptimeRobot, Better Uptime
**Google:** Calendar, Drive, Gmail, Sheets, Docs, Slides, Forms, Meet, Analytics, Ads, Search Console, YouTube
**Microsoft:** Teams, Outlook, OneDrive, SharePoint, Azure DevOps, Azure Storage, Azure Functions
**Design:** Figma, Canva, Adobe Creative Cloud
**Finance:** Stripe, PayPal, QuickBooks, Xero, FreshBooks, Chargebee
**Marketing:** Mailchimp, HubSpot Marketing, Google Analytics, Facebook Ads, Twitter/X Ads, LinkedIn Ads
**Data:** Airtable, Supabase, MongoDB Atlas, Snowflake, BigQuery
**Other:** Dropbox, Box, Evernote, Spotify, Medium, WordPress, Shopify, Wix, Reddit

### API & WebSocket Server

**Files:** `nexus/api/server.py` (4302 lines), `nexus/api/browser_routes.py`, `nexus/api/vm_routes.py`

**`server.py`** — FastAPI application with:

- **REST Endpoints:** Chat, settings, tools, skills, memories, goals, integrations, users, system status — all with Pydantic validation
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
- **Workspace Sandboxing:** Shell sessions cannot escape `data/workspace/`

### Self-Improvement System

**Directory:** `nexus/self_improve/`

A comprehensive learning system that runs on a configurable cycle (default: every 30 minutes via the heartbeat scheduler):

| # | Module | What It Does |
|---|--------|-------------|
| 1 | **Experience Replay** | Stores high-scoring task executions as reusable trajectories |
| 2 | **Meta-Learning Engine** | Learns which reasoning approaches work per task type |
| 3 | **Tool Affinity Analysis** | Co-occurrence matrix, successful vs. failed tool chains, anti-patterns |
| 4 | **Failure Pattern Recognition** | NLP clustering of failures, generates countermeasures |
| 5 | **Strategy Injection** | Lifecycle-managed learned strategies injected into system prompt |
| 6 | **Quality Feedback Loop** | Monitors quality trends, triggers interventions on decline |
| 7 | **Memory Consolidation** | Deduplication, usage tracking, promotion, pruning |
| 8 | **Learning Rate & Decay** | Time-weighted confidence scores with evidence adjustment |
| 9 | **Quality Assessment** | Self-supervised quality scoring of responses |
| 10 | **Self-Supervised Learning** | Outcome prediction, pattern extraction, negative mining |
| 11 | **Improvement Pipeline** | End-to-end: assess → identify → generate → test → validate → deploy |
| 12 | **Real-Time Failure Learning** | Instant heuristic fix generation on tool failures |
| 13 | **User Satisfaction Detection** | Implicit feedback from conversation signals |
| 14 | **Prediction Engine** | Tool success prediction and failure risk scoring |

**Core files:**
- `__init__.py` — Main pipeline (15-step analysis cycle)
- `predict.py` — Tool outcome prediction, pattern extraction, success/failure predictors
- `realtime.py` — Real-time learning: instant fix generation from failure traces

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

### Task Queue (Celery + Redis)

**Files:** `nexus/celery_app.py`, `nexus/tasks/`

Celery with Redis as broker and result backend:

| Task | Interval | Purpose |
|------|----------|---------|
| `check_data_integrity` | 5 min | SQLite health check |
| `process_scheduled_tasks` | 30s | Execute due scheduled tasks |
| `process_nl_schedules` | 30s | NL-defined schedule processing |
| `process_incomplete_goals` | 2 min | Check stalled goals |
| `file_watcher_tick` | 30s | Poll watched directories |
| `web_monitor_tick` | 60s | Monitor watched web pages |
| `self_improvement_run` | 30 min | Self-improvement pipeline |
| `health_summary` | 30s | Health status events |
| `memory_curve_pruning` | 30 min | Forgetting-curve pruning |

**Event Bridge:** Redis pubsub relays events from Celery workers to the main process, keeping the Web UI updated during background task execution.

### Virtual Computer

**Files:** `nexus/virtual_computer/container.py`, `nexus/virtual_computer/setup-vm.sh`

Docker-based sandboxed desktop environment:
- Container lifecycle: create, start, stop, delete via REST API
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
3. Heartbeat/Celery beat scheduler start
4. Uvicorn server on `127.0.0.1:8765` with WebSocket support
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

## ML/AI Engineering Suite — `ml_ai_skills/`

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
| **Generation** | `temperature` (default 0.3), `max_tokens`, `top_p` |
| **Agent Identity** | `agent_name`, `personality`, `traits`, `communication_style` |
| **Reasoning** | `max_nexus_iterations` (default 9999), `adaptive_iterations` (default True), `loop_detection_threshold` |
| **ADHD** | `enable_adhd_reasoning` (default True), `adhd_complexity_min` (default "medium"), `adhd_max_analogies` (default 3) |
| **Memory** | `enable_long_term_memory`, `memory_retrieval_k` (default 6), `embedding_model`, `enable_memory_tree` |
| **Speed** | `cache_llm_responses`, `parallel_tool_execution`, `fast_response_mode` |
| **Task Execution** | `max_tool_calls_per_task`, `max_cost_per_task_usd`, `max_task_duration_seconds` (1800) |
| **Security** | `enable_auth`, `cors_origins` |
| **Celery** | `enable_celery`, `redis_url` |
| **Self-Improvement** | `enable_self_improvement` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Framework** | FastAPI + Uvicorn |
| **Database** | SQLite (via aiosqlite, async) — optional: PostgreSQL |
| **Task Queue** | Celery + Redis (optional) |
| **WebSocket** | FastAPI WebSocket + custom event bus |
| **Frontend** | Vanilla JavaScript SPA (no framework) |
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

All rights reserved. Hyper Nexus is proprietary software. Unauthorized copying, distribution, or use is prohibited.
