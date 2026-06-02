"""

FastAPI server for Nexus - WebSocket streaming + REST endpoints + static WebUI.



Security fixes (v6.1):

- JWT authentication on all API/WebSocket endpoints

- Per-IP rate limiting middleware

- CORS configuration

- API key support for programmatic access



v30: Single-agent ReasoningEngine for all tasks.

     Session-scoped engine cache.

v28: Security hardening — encrypted settings at rest, sandbox MRO fix, atomic rate limiting.

"""

from __future__ import annotations



import asyncio

import hashlib

import json

import time

import uuid

from pathlib import Path

from typing import Any



from fastapi import FastAPI, File, Form, HTTPException, Header, UploadFile, WebSocket, WebSocketDisconnect, Depends, Request

from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel



# ── v29: Session context variable ──────────────────────────────────────────

# Allows tool implementations (memory tools, etc.) to access the current

# session_id without needing to pass it through the entire call chain.

# Set per-request in the WebSocket chat handler and REST chat endpoint.

import contextvars

_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(

    "nexus_session_id", default=""

)

# ── Chat interruption tracking ─────────────────────────────────────────

# Tracks the currently running chat task per session. When a new message

# arrives for a session that already has a running task, the old task is

# cancelled so the new message is processed immediately.

_current_chat_tasks: dict[str, asyncio.Task] = {}

_chat_tasks_lock = asyncio.Lock()



# ── Nexus Framework: unified adaptive reasoning (no mode selection needed) ──────



from .. import config, heartbeat

# All agent logic flows through ReasoningEngine's Nexus Framework Reasoning loop

from ..core import llm

from ..events import BUS, emit

from ..memory import db

from ..reasoning import ReasoningEngine

from ..tools import REGISTRY

from ..tools.builtin import custom_loader



# ── FIX: Session-scoped ReasoningEngine cache ──────────────────────────────────

# Avoids creating a brand new ReasoningEngine + MemoryManager on every single

# request, which loses the in-memory vector index and forces a full rebuild

# from DB on each call.  TTL of 30 min keeps things warm.

# ── Session cache with periodic cleanup ──────────────────────────────────

_session_cache: dict[str, tuple[ReasoningEngine, float]] = {}

_session_cache_lock = asyncio.Lock()
_SESSION_CACHE_TTL = 1800  # 30 minutes

_SESSION_CACHE_MAX_SIZE = 100  # Prevent unbounded growth

_last_cache_cleanup = time.time()  # Track when we last did a full sweep

_CACHE_CLEANUP_INTERVAL = 300  # Run full expired sweep every 5 minutes





def _cleanup_expired_sessions() -> int:

    """Remove all expired entries from the session cache.



    Called both on overflow AND on a timer, so sessions don't linger

    indefinitely when the cache never hits max size.

    Returns the number of entries evicted.

    """

    global _last_cache_cleanup

    now = time.time()

    expired = [k for k, (_, ts) in _session_cache.items()

              if now - ts >= _SESSION_CACHE_TTL]

    for k in expired:

        del _session_cache[k]

    _last_cache_cleanup = now

    return len(expired)





async def _get_engine(session_id: str) -> ReasoningEngine:

    """Return a cached ReasoningEngine for the given session_id, or create one."""

    now = time.time()

    async with _session_cache_lock:
        # Evict expired entries on overflow OR on timer

        if len(_session_cache) > _SESSION_CACHE_MAX_SIZE:

            _cleanup_expired_sessions()

        elif now - _last_cache_cleanup >= _CACHE_CLEANUP_INTERVAL:

            _cleanup_expired_sessions()



        if session_id in _session_cache:

            engine, ts = _session_cache[session_id]

            if now - ts < _SESSION_CACHE_TTL:

                _session_cache[session_id] = (engine, now)

                return engine

            else:

                del _session_cache[session_id]

        engine = ReasoningEngine(session_id)

        _session_cache[session_id] = (engine, now)

        return engine



async def _invalidate_engine_cache() -> int:

    """Clear all cached engines. Returns the number evicted."""

    async with _session_cache_lock:

        count = len(_session_cache)

        _session_cache.clear()

        return count



# ── Auth (v6.1) ────────────────────────────────────────────────────────────────

from ..auth import require_auth, optional_auth, check_rate_limit, create_token, authenticate, init_auth, get_rate_limit_config, reset_rate_limit







app = FastAPI(title="Hyper Nexus Agent", version="8.0.0")



# ── CORS ────────────────────────────────────────────────────────────────────────

app.add_middleware(

    CORSMiddleware,

    allow_origins=config.get("cors_origins", ["http://localhost:8765", "http://127.0.0.1:8765"]),

    allow_credentials=True,

    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],

    allow_headers=["Authorization", "Content-Type", "X-API-Key"],

)



# ── Security headers middleware ──────────────────────────────────────────────

@app.middleware("http")

async def security_headers_middleware(request: Request, call_next):

    """Add security headers to all HTTP responses."""

    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"

    response.headers["X-Frame-Options"] = "DENY"

    response.headers["X-XSS-Protection"] = "1; mode=block"

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # CSP allows self + inline styles (needed for current WebUI)

    response.headers["Content-Security-Policy"] = (

        "default-src 'self'; "

        "script-src 'self' 'unsafe-inline'; "

        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "

        "font-src https://fonts.gstatic.com; "

        "img-src 'self' data: blob:; "

        "connect-src 'self' ws: wss:; "

        "frame-ancestors 'none'"

    )

    return response











# ── Rate limiting middleware ────────────────────────────────────────────────────

# Paths that bypass rate limiting entirely (static assets, health checks, WebSocket upgrade)

_RATE_LIMIT_EXEMPT_PREFIXES = ("/static/", "/favicon", "/api/status")

_RATE_LIMIT_READ_PREFIXES = ("/api/sessions", "/api/tools", "/api/skills", "/api/memories", "/api/goals", "/api/tasks", "/api/metrics", "/api/notifications", "/api/workspace", "/api/events", "/api/quality", "/api/improvements", "/api/environment", "/api/integrations", "/api/browser")

_RATE_LIMIT_WRITE_PREFIXES = ("/api/chat", "/api/upload", "/api/custom_tools", "/api/settings", "/api/skills/upload", "/api/skills/custom")



@app.middleware("http")

async def rate_limit_middleware(request: Request, call_next):

    """Per-IP rate limiting with smart categorization."""

    path = request.url.path



    # Skip rate limit for static files, favicon, and status endpoint

    if any(path.startswith(p) for p in _RATE_LIMIT_EXEMPT_PREFIXES):

        return await call_next(request)



    # Skip WebSocket upgrade requests (detected by Upgrade header, not just known paths)

    if path == "/ws" or path == "/api/browser/stream" or request.headers.get("upgrade", "").lower() == "websocket":

        return await call_next(request)



    client_ip = request.client.host if request.client else "unknown"

    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

    base_key = f"ip:{ip_hash}"



    # Categorize request for appropriate limits

    is_write = any(path.startswith(p) for p in _RATE_LIMIT_WRITE_PREFIXES)

    is_read = any(path.startswith(p) for p in _RATE_LIMIT_READ_PREFIXES)



    if is_write:

        # Write/API calls get a tighter per-endpoint limit

        key = f"{base_key}:write"

        allowed, remaining = check_rate_limit(key, max_requests=120, window_seconds=60)

    elif is_read:

        # Read-only endpoints share a generous bucket

        key = f"{base_key}:read"

        allowed, remaining = check_rate_limit(key, max_requests=300, window_seconds=60)

    else:

        # Everything else (auth, misc)

        key = base_key

        allowed, remaining = check_rate_limit(key)



    if not allowed:

        retry_after = 10

        return JSONResponse(

            status_code=429,

            content={"detail": "Rate limit exceeded. Please wait and try again.", "retry_after": retry_after},

            headers={"Retry-After": str(retry_after), "X-RateLimit-Remaining": "0"},

        )



    response = await call_next(request)

    response.headers["X-RateLimit-Remaining"] = str(remaining)

    return response





@app.on_event("startup")

async def on_startup():

    """Startup logic — runs when NOT using the lifespan context in run.py.



    When run.py's lifespan() is active, FastAPI may still call this in some

    versions.  We guard db.init() and heartbeat.start() with idempotency checks

    so they are safe to call multiple times.

    """

    await init_auth()  # Set up authentication

    # db.init() and heartbeat.start() are called by run.py lifespan — they are

    # idempotent so this is safe if both fire.

    try:

        await db.init()

        await heartbeat.start()

    except Exception:

        pass  # Already initialised by lifespan







    await emit("startup", message=f"{config.get('agent_name')} online")

    # #3 warning: alert if auth is disabled

    if not config.get("enable_auth", False):

        print(

            "\n[!] SECURITY WARNING: enable_auth is FALSE.\n"

            "    The entire API is accessible without authentication.\n"

            "    If this server is exposed to the network, set enable_auth=true\n"

            "    in Settings or set NEXUS_AUTH_KEY env var.\n"

        )

    # Cleanup duplicate memories on startup

    try:

        from ..memory import cleanup_duplicates, prune_stale_memories

        deleted = await cleanup_duplicates()

        if deleted > 0:

            print(f"[startup] Cleaned up {deleted} duplicate memories")

        pruned = await prune_stale_memories()

        if pruned > 0:

            print(f"[startup] Pruned {pruned} stale memories")

        # Load custom tools from DB (deferred from module-level import)

        try:

            await custom_loader.load_all_from_db()

            print("[startup] Custom tools loaded from database")

        except Exception as e:

            print(f"[startup] Custom tools load failed: {e}")

        # Load custom skills from DB

        try:

            await custom_loader.load_all_custom_skills_from_db()

            await custom_loader.refresh_custom_skills_cache()

            print("[startup] Custom skills loaded from database")

        except Exception as e:

            print(f"[startup] Custom skills load failed: {e}")

    except Exception as e:

        print(f"[startup] Memory cleanup failed: {e}")



    # Initialize MCP Manager — load servers from DB and auto-connect

    try:

        from ..tools.builtin.mcp_manager import MCP_MANAGER

        await MCP_MANAGER.initialize()

        # v22: Sync REGISTRY with MCP state (ensures all MCP tools are registered)

        sync_result = await MCP_MANAGER.sync_registry()

        _sync_info = ""

        if sync_result.get("unregistered"):

            _sync_info += f", cleaned {len(sync_result['unregistered'])} stale"

        if sync_result.get("registered"):

            _sync_info += f", registered {len(sync_result['registered'])} new"

        print(f"[startup] MCP Manager initialized (REGISTRY sync: {_sync_info or 'clean'})")

    except Exception as e:

        print(f"[startup] MCP Manager init failed: {e}")



    # Start browser event mirror (safe async context here)

    try:

        if _browser_mirror_pending:

            asyncio.create_task(start_event_mirror())

            print("[startup] Browser event mirror started")

    except Exception as e:

        print(f"[startup] Browser mirror start failed: {e}")





@app.on_event("shutdown")

async def on_shutdown():

    """Shutdown logic — idempotent with run.py lifespan."""

    # Stop browser event mirror

    try:

        if _browser_mirror_pending:

            from .browser_routes import stop_event_mirror

            await stop_event_mirror()

    except Exception:

        pass

    # Cleanup BrowserManager (Playwright) — this is now the ONLY Playwright instance

    # since browser_tools.py was refactored to share BrowserManager's browser.

    try:

        from .browser_routes import get_manager

        await get_manager().cleanup()

    except Exception:

        pass

    try:

        await heartbeat.stop()

    except Exception:

        pass

    try:

        await db.close()

    except Exception:

        pass

    try:

        await llm.shutdown_client()

    except Exception:

        pass





# ---------------- WebUI static files ----------------



WEBUI_DIR = config.WEBUI_DIR





@app.get("/")

async def root():

    index = WEBUI_DIR / "index.html"

    if not index.exists():

        return JSONResponse({"error": "WebUI not built"}, status_code=500)

    return FileResponse(index)





@app.get("/favicon.svg")

async def favicon():

    favicon_path = WEBUI_DIR / "favicon.svg"

    if favicon_path.exists():

        return FileResponse(favicon_path, media_type="image/svg+xml")

    return JSONResponse({"error": "Not Found"}, status_code=404)





if WEBUI_DIR.exists():

    app.mount("/static", StaticFiles(directory=WEBUI_DIR), name="static")



# ── Register browser live routes (dedicated WebSocket + REST for live preview) ──

try:

    from .browser_routes import router as browser_router, start_event_mirror

    app.include_router(browser_router)

    _browser_mirror_pending = True  # Will be started in on_startup()

except ImportError:

    _browser_mirror_pending = False  # browser_routes not available



# ── Register VM routes (REST + WebSocket for virtual computer) ──

try:

    from .vm_routes import router as vm_router

    app.include_router(vm_router)

    print("[startup] VM routes registered at /api/vm")

except ImportError:

    print("[startup] VM routes not available")





# ---------------- Models ----------------



class LoginRequest(BaseModel):

    username: str

    password: str



class ChatRequest(BaseModel):

    session_id: str

    message: str

    images: list[str] | None = None  # Base64 encoded images



class SettingsUpdate(BaseModel):

    updates: dict[str, Any]



class GoalCreate(BaseModel):

    title: str

    description: str = ""

    priority: int = 5



class CustomToolCreate(BaseModel):

    model_config = {"protected_namespaces": ()}

    name: str

    description: str

    tool_schema: dict

    code: str



class CreateUserRequest(BaseModel):

    username: str

    password: str

    role: str = "user"





# ── Skills (mapped to actual tool names) ────────────────────────────────

SKILLS_REGISTRY = [

    {"name": "generate_image", "description": "Generate AI images from text prompts", "category": "ai_media", "icon": "IMG", "enabled": True},

    {"name": "web_search", "description": "Search the web for real-time information", "category": "research", "icon": "WEB", "enabled": True},

    {"name": "fetch_url", "description": "Extract and read content from web pages", "category": "research", "icon": "READ", "enabled": True},

    {"name": "pdf", "description": "Create and manipulate PDF documents", "category": "documents", "icon": "PDF", "enabled": True},

    {"name": "docx", "description": "Create and edit Microsoft Word documents", "category": "documents", "icon": "DOC", "enabled": True},

    {"name": "xlsx", "description": "Create and manage Excel spreadsheets", "category": "documents", "icon": "XLS", "enabled": True},

    {"name": "ppt", "description": "Create PowerPoint presentations", "category": "documents", "icon": "PPT", "enabled": True},

    {"name": "charts", "description": "Generate charts and visualizations", "category": "documents", "icon": "CHT", "enabled": True},

    {"name": "fullstack_dev", "description": "Full-stack web development", "category": "development", "icon": "DEV", "enabled": True},

    {"name": "agent_browser", "description": "Browser automation and scraping", "category": "development", "icon": "URL", "enabled": True},


    {"name": "ffmpeg", "description": "FFmpeg media processing - convert, trim, merge, extract audio, create GIF", "category": "media", "icon": "VID", "enabled": True},

    {"name": "ffmpeg_convert", "description": "Convert media between formats (video/audio)", "category": "media", "icon": "VID", "enabled": True},

    {"name": "ffmpeg_extract_audio", "description": "Extract audio track from video", "category": "media", "icon": "AUD", "enabled": True},

    {"name": "ffmpeg_create_gif", "description": "Create optimized animated GIF from video", "category": "media", "icon": "GIF", "enabled": True},

    {"name": "ffmpeg_merge_video", "description": "Merge/join multiple video files", "category": "media", "icon": "VID", "enabled": True},

    {"name": "ffmpeg_trim", "description": "Trim video to specific time range", "category": "media", "icon": "VID", "enabled": True},

    {"name": "LLM", "description": "Language model chat completions", "category": "ai_media", "icon": "AI", "enabled": True},

    {"name": "video_generation", "description": "Generate AI videos from text", "category": "ai_media", "icon": "VID", "enabled": True},

    {"name": "video_understand", "description": "Analyze video content", "category": "ai_media", "icon": "PLY", "enabled": True},

    {"name": "finance", "description": "Financial data analysis", "category": "finance", "icon": "FIN", "enabled": True},

    {"name": "skill_creator", "description": "Create custom skills", "category": "development", "icon": "SKL", "enabled": True},

    {"name": "skill_vetter", "description": "Security vetting for skills", "category": "development", "icon": "SEC", "enabled": True},

    {"name": "image_edit", "description": "Edit images with AI", "category": "ai_media", "icon": "EDT", "enabled": True},

    {"name": "image_understand", "description": "Analyze image content", "category": "ai_media", "icon": "IMG", "enabled": True},

    # ── Nexus3D Engine Skills ──────────────────────────────────────────

    {"name": "nexus3d-architectural-toolkit", "description": "Generate 3D architectural elements: rooms, buildings, stairs, windows, doors, furniture, and complete scenes", "category": "3d_engine", "icon": "3DX", "enabled": True},

    {"name": "nexus3d-character-creator", "description": "Create fully rigged 3D humanoid characters with customizable proportions, auto skinning, pose libraries, and blend shapes", "category": "3d_engine", "icon": "CHR", "enabled": True},

    {"name": "nexus3d-scene-composer", "description": "Compose complex 3D scenes with multiple objects, lighting, and camera setups", "category": "3d_engine", "icon": "SCN", "enabled": True},

    {"name": "nexus3d-animation-director", "description": "Create and manage character animations, transitions, blending, and procedural motion", "category": "3d_engine", "icon": "ANI", "enabled": True},

    {"name": "nexus3d-motion-pipeline", "description": "Process and refine motion capture data, retarget animations, and generate motion graphs", "category": "3d_engine", "icon": "MOT", "enabled": True},

    {"name": "nexus3d-product-renderer", "description": "Render product visualization with studio lighting, turntable animations, and material previews", "category": "3d_engine", "icon": "RDR", "enabled": True},

    {"name": "nexus3d-physics-lab", "description": "Run physics simulations with rigid bodies, collisions, gravity, and constraints", "category": "3d_engine", "icon": "PHY", "enabled": True},

    {"name": "nexus3d-csg-architect", "description": "Design architectural elements using CSG (Constructive Solid Geometry) boolean operations", "category": "3d_engine", "icon": "CSG", "enabled": True},

    {"name": "nexus3d-studio-renderer", "description": "High-quality rendering with PBR materials, multi-light setups, and environment mapping", "category": "3d_engine", "icon": "REN", "enabled": True},

    {"name": "nexus3d-cinema-camera", "description": "Cinematic camera systems with DOF, motion blur, camera shake presets, and lens profiles", "category": "3d_engine", "icon": "CAM", "enabled": True},

    {"name": "nexus3d-csg-toolkit", "description": "CSG operations toolkit: boolean union/subtract/intersect on arbitrary 3D shapes", "category": "3d_engine", "icon": "BTN", "enabled": True},

    {"name": "nexus3d-material-lab", "description": "PBR material creation with procedural textures, layered materials, and UV mapping", "category": "3d_engine", "icon": "MAT", "enabled": True},

    # ── ML/AI Advanced Skills ──────────────────────────────────────────

    {"name": "llm-trainer", "description": "Advanced LLM training: GPT Transformers, LoRA, QLoRA, RLHF, PPO, fine-tuning with DeepSpeed/FSDP/DDP", "category": "ai_media", "icon": "LLM", "enabled": True},

    {"name": "transformer-architect", "description": "Design and build advanced Transformer architectures: GPT, BERT, T5, custom attention, RoPE, Flash Attention", "category": "ai_media", "icon": "TRF", "enabled": True},

    {"name": "distributed-training", "description": "Production distributed training: DDP, FSDP, DeepSpeed ZeRO, gradient checkpointing, torch.compile, model/tensor parallelism", "category": "ai_media", "icon": "DST", "enabled": True},

    {"name": "peft-finetuning", "description": "Parameter-efficient fine-tuning: LoRA, QLoRA, Adapters, Prefix Tuning, Prompt Tuning, BitFit from scratch", "category": "ai_media", "icon": "PFT", "enabled": True},

    {"name": "rlhf-lab", "description": "RLHF alignment lab: SFT, Reward Models, PPO, DPO, KTO, ORPO, SimPO, Constitutional AI, safety evaluation", "category": "ai_media", "icon": "RLF", "enabled": True},

]



# Load enabled/disabled state from settings

def _get_skills_with_state():

    """Return skills list with enabled state from settings."""

    skills_state = config.get("skills_enabled_state", {})

    result = []

    for s in SKILLS_REGISTRY:

        skill = dict(s)

        if skill["name"] in skills_state:

            skill["enabled"] = skills_state[skill["name"]]

        result.append(skill)

    return result





# ---------------- Auth endpoints (v6.1) ----------------



@app.post("/api/auth/login")

async def api_login(req: LoginRequest, request: Request):

    """Authenticate and return a JWT token."""

    client_ip = request.client.host if request.client else "unknown"

    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

    allowed, _ = check_rate_limit(f"login:{ip_hash}", max_requests=10, window_seconds=60)

    if not allowed:

        raise HTTPException(429, "Too many login attempts. Try again in 60 seconds.")



    token = authenticate(req.username, req.password)

    if not token:

        raise HTTPException(401, "Invalid username or password")

    return {"token": token, "username": req.username}





@app.get("/api/auth/me")

async def api_auth_me(auth: dict = Depends(require_auth)):

    """Return current authenticated user info."""

    return {"session_id": auth.get("sid"), "role": auth.get("role")}





# ── User Management (admin only) ──────────────────────────────────────────────

@app.get("/api/users")

async def api_list_users(auth: dict = Depends(require_auth)):

    """List all registered users (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..auth import list_users

    return list_users()



@app.post("/api/users")

async def api_create_user(req: CreateUserRequest, auth: dict = Depends(require_auth)):

    """Create a new user (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..auth import create_user

    user = await create_user(req.username, req.password, req.role)

    if not user:

        raise HTTPException(400, "User creation failed (username may already exist)")

    await emit("user_created", username=req.username, role=req.role)

    return {"ok": True, "user": user}



@app.patch("/api/users/{username}")

async def api_update_user(username: str, payload: dict[str, Any], auth: dict = Depends(require_auth)):

    """Update a user's role or password (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..auth import update_user

    user = await update_user(username, role=payload.get("role"), password=payload.get("password"))

    if not user:

        raise HTTPException(404, "User not found")

    await emit("user_updated", username=username)

    return {"ok": True, "user": user}



@app.delete("/api/users/{username}")

async def api_delete_user(username: str, auth: dict = Depends(require_auth)):

    """Delete a user (admin only, cannot delete last admin)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..auth import delete_user

    ok = await delete_user(username)

    if not ok:

        raise HTTPException(400, "Cannot delete user (last admin or not found)")

    await emit("user_deleted", username=username)

    return {"ok": True}



@app.post("/api/users/{username}/regenerate-key")

async def api_regenerate_key(username: str, auth: dict = Depends(require_auth)):

    """Generate a new API key for a user (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..auth import regenerate_api_key

    new_key = await regenerate_api_key(username)

    if not new_key:

        raise HTTPException(404, "User not found")

    return {"ok": True, "new_api_key": new_key}





# ---------------- REST endpoints ----------------



@app.get("/api/status")

async def api_status(auth: dict = Depends(optional_auth)):

    return {

        "agent_name": config.get("agent_name"),

        "version": "8.0.0",

        "tools": len(REGISTRY.all()),

        "memory_enabled": config.get("enable_long_term_memory", True),

        "auth_enabled": config.get("enable_auth", False),

        "usage": {

            "prompt_tokens": llm.GLOBAL_USAGE.prompt_tokens,

            "completion_tokens": llm.GLOBAL_USAGE.completion_tokens,

            "total_tokens": llm.GLOBAL_USAGE.total_tokens,

            "cost_usd": llm.GLOBAL_USAGE.cost_usd,

        },

    }





@app.get("/api/rate-limit")

async def api_rate_limit_info(auth: dict = Depends(require_auth)):

    """Return current rate limit configuration and status (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    return get_rate_limit_config()





@app.post("/api/rate-limit/reset")

async def api_rate_limit_reset(auth: dict = Depends(require_auth)):

    """Reset all rate limit buckets (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    # #8 fix: use the public reset_all function instead of mutating private internals

    from ..auth import reset_all_rate_limits

    count = reset_all_rate_limits()

    return {"ok": True, "reset_buckets": count}





@app.get("/api/llm/status")

async def api_llm_status(auth: dict = Depends(require_auth)):

    """Return LLM circuit breaker status and global usage stats (admin only)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..core.llm import get_circuit_breaker_status, GLOBAL_USAGE

    return {

        "circuit_breaker": get_circuit_breaker_status(),

        "global_usage": {

            "prompt_tokens": GLOBAL_USAGE.prompt_tokens,

            "completion_tokens": GLOBAL_USAGE.completion_tokens,

            "total_tokens": GLOBAL_USAGE.total_tokens,

            "cost_usd": round(GLOBAL_USAGE.cost_usd, 6),

        },

    }





@app.post("/api/llm/circuit-breaker/reset")

async def api_llm_circuit_breaker_reset(

    model: str | None = None, auth: dict = Depends(require_auth)

):

    """Reset circuit breaker for one model or all models (admin only).

    #7 fix: accepts optional ?model= query param to reset a single model."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    from ..core.llm import reset_circuit_breaker

    count = reset_circuit_breaker(model)

    if model:

        return {"ok": True, "reset_model": model, "reset": count}

    return {"ok": True, "reset_models": count}





@app.get("/api/settings")

async def api_get_settings(auth: dict = Depends(require_auth)):

    s = dict(config.SETTINGS)

    # FIX: Mask all secret keys in a single pass (was double-masking openrouter_api_key)

    _API_KEY_NAMES = ("openrouter_api_key", "openai_api_key", "nvidia_api_key", "custom_api_key", "together_api_key")

    _SECRET_NAMES = ("auth_hmac_key", "auth_salt", "smtp_password", "github_token", "default_admin_password")

    for secret_key in _API_KEY_NAMES + _SECRET_NAMES:

        val = s.get(secret_key, "")

        if val and len(val) > 8:

            s[secret_key] = "•" * (len(val) - 4) + val[-4:]

        elif val:

            # Short values (placeholders or empty): just mask entirely

            s[secret_key] = "••••"

    # Mark env-overridden keys so UI knows they can't be changed via settings

    s["_env_overridden"] = list(config._env_loaded_keys)

    return s





@app.post("/api/settings")

async def api_update_settings(req: SettingsUpdate, auth: dict = Depends(require_auth)):

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required to update settings")

    updates = dict(req.updates)



    # FIX: Properly handle API key fields during save.

    # - Never overwrite secrets that should only come from env (auth keys, salts)

    # - DO allow API keys to be saved (openrouter, openai, custom) since users

    #   legitimately need to set these via the UI

    _NEVER_OVERWRITE = {"auth_hmac_key", "auth_salt"}

    _API_KEY_NAMES = ("openrouter_api_key", "openai_api_key", "nvidia_api_key", "custom_api_key", "together_api_key")



    for key in list(updates.keys()):

        if key in _NEVER_OVERWRITE:

            updates.pop(key)

            continue



        # Skip API key updates where the submitted value is the masked version

        # (user didn't change the field — it still shows bullets from GET)

        if key in _API_KEY_NAMES:

            submitted = updates.get(key, "") or ""

            if not submitted or submitted.count("•") > len(submitted) // 2:

                # Value is empty or mostly masked — don't overwrite the real key

                updates.pop(key)

                continue



    config.set_many(updates)

    await emit("settings_updated", updates=list(updates.keys()))

    return {"ok": True}





@app.get("/api/models")

async def api_models(auth: dict = Depends(require_auth)):

    models = await llm.list_models()

    return {"models": models}





@app.post("/api/test-connection")

async def api_test_connection(auth: dict = Depends(require_auth)):

    """Test the current API key by calling the /models endpoint.

    Returns success/failure with details to help debug 401 errors."""

    provider = config.get("provider", "openrouter")

    api_key_name = {

        "together": "together_api_key",

        "openai": "openai_api_key",

        "nvidia": "nvidia_api_key",

        "custom": "custom_api_key",

        "groq": "groq_api_key",

    }.get(provider, "openrouter_api_key")



    api_key = config.get(api_key_name, "")

    key_preview = ""

    if api_key and len(api_key) > 8:

        key_preview = api_key[:4] + "..." + api_key[-4:]

    elif api_key:

        key_preview = "(short key set)"

    else:

        key_preview = "(empty — no key configured!)"



    try:

        models = await llm.list_models()

        return {

            "ok": True,

            "provider": provider,

            "key_field": api_key_name,

            "key_preview": key_preview,

            "model_count": len(models),

        }

    except Exception as e:

        return {

            "ok": False,

            "provider": provider,

            "key_field": api_key_name,

            "key_preview": key_preview,

            "error": "Connection test failed. Check server logs for details.",

        }





@app.get("/api/sessions")

async def api_sessions(auth: dict = Depends(require_auth)):

    return await db.list_sessions()





@app.post("/api/sessions")

async def api_new_session(name: str = "New Session", auth: dict = Depends(require_auth)):

    sid = str(uuid.uuid4())

    await db.create_session(sid, name)

    return {"id": sid, "name": name}





@app.get("/api/sessions/{session_id}/messages")

async def api_session_messages(session_id: str, auth: dict = Depends(require_auth)):

    return await db.get_messages(session_id, limit=500)





@app.delete("/api/sessions/{session_id}")

async def api_delete_session(session_id: str, auth: dict = Depends(require_auth)):

    await db.delete_session(session_id)

    # Clear the cached ReasoningEngine for this session

    async with _session_cache_lock:

        _session_cache.pop(session_id, None)

    return {"ok": True}





@app.delete("/api/sessions")

async def api_delete_all_sessions(auth: dict = Depends(require_auth)):

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    async with db._connect() as conn:

        await conn.execute("DELETE FROM messages")

        await conn.execute("DELETE FROM sessions")

    # Clear all cached engines

    async with _session_cache_lock:

        _session_cache.clear()

    return {"ok": True}





@app.post("/api/chat")

async def api_chat(req: ChatRequest, auth: dict = Depends(require_auth)):

    # v29: Set session context so tools can access session_id

    _current_session_id.set(req.session_id)

    await db.create_session(req.session_id, "Chat")

    engine = await _get_engine(req.session_id)


    # Nexus Framework: unified adaptive reasoning — no mode selection needed
    # Direct ReasoningEngine call — fast, single-agent Nexus Framework    # Pass images for vision support

    try:
        answer = await asyncio.wait_for(engine.respond(req.message, images=req.images if req.images else None), timeout=600.0)
    except TimeoutError:
        raise HTTPException(504, "Request timed out after 600 seconds. The LLM provider took too long to respond — try a faster model or check your API provider's status.")
    return {"answer": answer}





@app.get("/api/tools")

async def api_tools(auth: dict = Depends(require_auth)):

    return [

        {

            "name": t.name,

            "description": t.description,

            "risk": t.risk,

            "category": t.category,

            "parameters": t.parameters_schema,

        }

        for t in REGISTRY.all()

    ]





@app.get("/api/custom_tools")

async def api_custom_tools(auth: dict = Depends(require_auth)):

    return await db.list_custom_tools()





@app.post("/api/custom_tools")

async def api_create_custom_tool(req: CustomToolCreate, auth: dict = Depends(require_auth)):

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required to create custom tools")

    status = custom_loader.load_custom_tool(req.name, req.description, req.tool_schema, req.code)

    if status != "ok":

        raise HTTPException(400, status)

    await db.add_custom_tool(req.name, req.description, req.tool_schema, req.code)

    await emit("custom_tool_added", name=req.name)

    return {"ok": True}





@app.delete("/api/custom_tools/{tool_id}")

async def api_delete_custom_tool(tool_id: int, auth: dict = Depends(require_auth)):

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    row = next((t for t in await db.list_custom_tools() if t["id"] == tool_id), None)

    if row:

        REGISTRY.unregister(row["name"])

    await db.delete_custom_tool(tool_id)

    return {"ok": True}





@app.get("/api/memories")

async def api_memories(kind: str | None = None, auth: dict = Depends(require_auth)):

    return await db.all_memories(kind=kind, limit=500)





@app.post("/api/memories")

async def api_add_memory(

    content: str,

    kind: str = "factual",

    importance: float = 0.7,

    tags: str = "[]",

    auth: dict = Depends(require_auth),

):

    import json as _json

    parsed_tags = _json.loads(tags) if isinstance(tags, str) else tags

    mem_id = await db.add_memory(content=content, kind=kind, importance=importance, tags=parsed_tags)

    return {"ok": True, "id": mem_id}





@app.delete("/api/memories/{mem_id}")

async def api_delete_memory(mem_id: int, auth: dict = Depends(require_auth)):

    await db.delete_memory(mem_id)

    return {"ok": True}





@app.post("/api/memories/cleanup")

async def api_cleanup_memories(auth: dict = Depends(require_auth)):

    from ..memory import cleanup_duplicates

    deleted = await cleanup_duplicates()

    return {"ok": True, "deleted": deleted}





@app.get("/api/goals")

async def api_goals(status: str | None = None, auth: dict = Depends(require_auth)):

    return await db.list_goals(status=status)





@app.post("/api/goals")

async def api_create_goal(g: GoalCreate, auth: dict = Depends(require_auth)):

    gid = await db.add_goal(g.title, g.description, g.priority)

    await emit("goal_created",

               goal_id=gid,

               title=g.title,

               description=g.description,

               priority=g.priority,

               message=f"New goal created: '{g.title}' (priority {g.priority}/10).")

    return {"id": gid}





@app.patch("/api/goals/{goal_id}")

async def api_update_goal(goal_id: int, payload: dict[str, Any], auth: dict = Depends(require_auth)):

    _GOAL_COLUMNS = {"title", "description", "priority", "status", "progress", "updated_at"}

    safe_payload = {k: v for k, v in payload.items() if k in _GOAL_COLUMNS}

    if not safe_payload:

        raise HTTPException(400, "No valid fields to update")

    await db.update_goal(goal_id, **safe_payload)

    await emit("goal_updated", goal_id=goal_id, updates=list(safe_payload.keys()))

    return {"ok": True}





@app.delete("/api/goals/{goal_id}")

async def api_delete_goal(goal_id: int, auth: dict = Depends(require_auth)):

    await db.delete_goal(goal_id)

    await emit("goal_deleted", goal_id=goal_id)

    return {"ok": True}





@app.get("/api/tasks")

async def api_tasks(auth: dict = Depends(require_auth)):

    return await db.list_tasks()





@app.get("/api/tool_executions")

async def api_tool_executions(auth: dict = Depends(require_auth)):

    return await db.recent_tool_executions()





@app.get("/api/reflections")

async def api_reflections(auth: dict = Depends(require_auth)):

    return await db.recent_reflections()









# ── File Watches ────────────────────────────────────────────────────────────

@app.get("/api/watches")

async def api_list_watches(auth: dict = Depends(require_auth)):

    return await db.list_file_watches(enabled_only=False)



@app.post("/api/watches")

async def api_add_watch(path: str, label: str = "", auth: dict = Depends(require_auth)):

    wid = await db.add_file_watch(path, label)

    return {"id": wid, "ok": True}



@app.delete("/api/watches/{watch_id}")

async def api_delete_watch(watch_id: int, auth: dict = Depends(require_auth)):

    await db.delete_file_watch(watch_id)

    return {"ok": True}





# ── Web Monitors ────────────────────────────────────────────────────────────

@app.get("/api/monitors")

async def api_list_monitors(auth: dict = Depends(require_auth)):

    return await db.list_web_monitors(enabled_only=False)



@app.post("/api/monitors")

async def api_add_monitor(url: str, label: str = "", interval_seconds: int = 3600, auth: dict = Depends(require_auth)):

    mid = await db.add_web_monitor(url, label, interval_seconds)

    return {"id": mid, "ok": True}



@app.delete("/api/monitors/{monitor_id}")

async def api_delete_monitor(monitor_id: int, auth: dict = Depends(require_auth)):

    await db.delete_web_monitor(monitor_id)

    return {"ok": True}





# ── NL Schedules ────────────────────────────────────────────────────────────

@app.get("/api/schedules")

async def api_list_schedules(auth: dict = Depends(require_auth)):

    return await db.list_nl_schedules(enabled_only=False)



@app.delete("/api/schedules/{schedule_id}")

async def api_delete_schedule(schedule_id: int, auth: dict = Depends(require_auth)):

    await db.delete_nl_schedule(schedule_id)

    return {"ok": True}





# ── Improvement Log ─────────────────────────────────────────────────────────

@app.post("/api/improvements/trigger")
async def api_trigger_self_improvement(auth: dict = Depends(require_auth)):
    """Manually trigger the self-improvement pipeline."""
    from ..tasks.scheduler_tasks import _do_self_improvement_run
    try:
        await _do_self_improvement_run()
        return {"ok": True, "message": "Self-improvement pipeline triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/improvements")

async def api_improvements(resolved: bool = False, auth: dict = Depends(require_auth)):

    return await db.list_improvement_log(resolved=resolved, limit=100)





# ── Journal ────────────────────────────────────────────────────────────────

@app.get("/api/journal")

async def api_journal(limit: int = 50, mood: str | None = None, auth: dict = Depends(require_auth)):

    """Get journal entries, optionally filtered by mood."""

    return await db.list_journal_entries(limit=min(limit, 200), mood=mood)





@app.post("/api/journal")

async def api_add_journal(

    content: str,

    mood: str = "",

    tags: str = "[]",

    people: str = "[]",

    auth: dict = Depends(require_auth),

):

    import json as _json

    parsed_tags = _json.loads(tags) if isinstance(tags, str) else tags

    parsed_people = _json.loads(people) if isinstance(people, str) else people

    entry_id = await db.add_journal_entry(

        content=content, mood=mood, tags=parsed_tags,

        people=parsed_people, permanent=True,

    )

    return {"ok": True, "id": entry_id}





@app.delete("/api/journal/{entry_id}")

async def api_delete_journal(entry_id: int, auth: dict = Depends(require_auth)):

    await db.delete_journal_entry(entry_id)

    return {"ok": True}





@app.get("/api/journal/mood-summary")

async def api_mood_summary(days: int = 7, auth: dict = Depends(require_auth)):

    return await db.get_mood_summary(days=min(days, 365))





@app.post("/api/memories/{mem_id}/permanent")

async def api_set_memory_permanent(mem_id: int, auth: dict = Depends(require_auth)):

    """Set a memory as permanent (exempt from pruning) (#22)."""

    async with db._connect() as conn:

        await conn.execute("UPDATE memories SET permanent=1 WHERE id=?", (mem_id,))

    return {"ok": True}





@app.delete("/api/memories/{mem_id}/permanent")

async def api_unset_memory_permanent(mem_id: int, auth: dict = Depends(require_auth)):

    """Remove permanence flag from a memory."""

    async with db._connect() as conn:

        await conn.execute("UPDATE memories SET permanent=0 WHERE id=?", (mem_id,))

    return {"ok": True}





# ── Archived Memories (long-term storage) ────────────────────────────────

@app.get("/api/memories/archived")

async def api_archived_memories(auth: dict = Depends(require_auth), limit: int = 50):

    """Get archived memories."""

    return await db.get_archived_memories(limit=limit)





@app.post("/api/memories/{mem_id}/archive")

async def api_archive_memory(mem_id: int, auth: dict = Depends(require_auth)):

    """Archive a memory to long-term storage."""

    await db.archive_memory(mem_id)

    return {"ok": True}


# ── Memory Tree API Routes ──────────────────────────────────────────────────


@app.get("/api/memory/tree/stats")

async def api_tree_stats(auth: dict = Depends(require_auth)):
    """Get memory tree statistics: roots, nodes, chunks, entities, pending jobs."""
    from ..memory.memory_tree import MemoryTree
    tree = MemoryTree()
    stats = await tree.get_stats()
    return stats


@app.post("/api/memory/tree/ingest")

async def api_tree_ingest(
    content: str,
    source: str = "",
    tags: str = "[]",
    auth: dict = Depends(require_auth),
):
    """Ingest content into the memory tree."""
    import json as _json
    from ..memory.memory_tree import MemoryTree
    parsed_tags = _json.loads(tags) if isinstance(tags, str) else tags
    tree = MemoryTree()
    result = await tree.ingest(content=content, source=source, tags=parsed_tags)
    return result


@app.get("/api/memory/tree/browse")

async def api_tree_browse(
    action: str = "roots",
    query: str = "",
    root_id: str = "",
    days_ago: int = 0,
    limit: int = 20,
    auth: dict = Depends(require_auth),
):
    """Browse the memory tree: search, daily_digest, topic, roots, root."""
    from ..memory.memory_tree import MemoryTree
    tree = MemoryTree()
    if action == "roots":
        from ..memory.tree_db import tree_list_roots
        roots = await tree_list_roots(limit=limit)
        return {"roots": roots}
    elif action == "search":
        nodes = await tree.recall(query=query, limit=limit)
        return {"nodes": nodes}
    elif action == "daily_digest":
        digest = await tree.get_daily_digest(days_ago=days_ago)
        return digest
    elif action == "topic":
        nodes = await tree.get_topic_tree(query, limit=limit)
        return {"nodes": nodes, "topic": query}
    elif action == "root":
        nodes = await tree.recall_by_root(root_id)
        return {"nodes": nodes, "root_id": root_id}
    return {"error": f"Unknown action: {action}"}








# ── Workspace ────────────────────────────────────────────────────────────────

@app.get("/api/workspace/files")

async def api_workspace_files(auth: dict = Depends(require_auth)):

    from pathlib import Path as _Path

    from .. import config as _cfg

    workspace = _Path(_cfg.BASE_DIR) / "data" / "workspace"

    workspace.mkdir(parents=True, exist_ok=True)

    items = []

    for entry in sorted(workspace.rglob("*")):

        try:

            is_dir = entry.is_dir()

            rel = str(entry.relative_to(workspace))

            size = entry.stat().st_size if not is_dir else 0

            items.append({

                "name": entry.name,

                "path": rel,

                "is_dir": is_dir,

                "size": size,

            })

        except Exception:

            pass

    return items



@app.get("/api/workspace/file")

async def api_workspace_file(path: str, auth: dict = Depends(require_auth)):

    from pathlib import Path as _Path

    from .. import config as _cfg

    workspace = _Path(_cfg.BASE_DIR) / "data" / "workspace"

    target = (workspace / path).resolve()

    if workspace.resolve() not in target.parents and target != workspace.resolve():

        raise HTTPException(403, "Path outside workspace")

    if not target.exists() or target.is_dir():

        raise HTTPException(404, "File not found")

    try:

        return target.read_text(encoding="utf-8", errors="replace")[:50000]

    except Exception as e:

        raise HTTPException(500, str(e))



@app.get("/api/workspace/events")

async def api_workspace_events(limit: int = 50, auth: dict = Depends(require_auth)):

    return await db.recent_workspace_events(limit=limit)





# ── Environment Awareness (v7) ─────────────────────────────────────────────────

@app.get("/api/environment")

async def api_environment(auth: dict = Depends(require_auth)):

    """Get comprehensive environment state snapshot."""

    from ..environment import ENVIRONMENT

    return await ENVIRONMENT.get_state()





# ── Task Quality Scores (v7) ──────────────────────────────────────────────────

@app.get("/api/quality/scores")

async def api_quality_scores(limit: int = 50, auth: dict = Depends(require_auth)):

    """Get recent task quality scores from self-reflection."""

    return await db.recent_task_scores(limit=limit)





@app.get("/api/quality/average")

async def api_quality_average(auth: dict = Depends(require_auth)):

    """Get average quality score over the last 7 days."""

    return {"avg_score": await db.avg_quality_score()}





# ── Tool Intelligence (v7) ────────────────────────────────────────────────────

@app.get("/api/tools/intelligence")

async def api_tool_intelligence(auth: dict = Depends(require_auth)):

    """Get tool performance profiles with strategic ranking."""

    return await REGISTRY.strategic_tool_ranking()





@app.get("/api/tools/profiles")

async def api_tool_profiles(auth: dict = Depends(require_auth)):

    """Get all tool performance profiles."""

    return await db.get_tool_profiles()





# ── Reasoning Preferences (learned from meta-learning) ────────────────────────

@app.get("/api/reasoning/preferences")

async def api_reasoning_preferences(auth: dict = Depends(require_auth)):

    """Get learned reasoning strategy preferences."""

    return await db.get_all_reasoning_preferences()





# ── Tool Co-occurrence Analysis ─────────────────────────────────────────────

@app.get("/api/tools/cooccurrence")

async def api_tool_cooccurrence(auth: dict = Depends(require_auth)):

    """Get tool co-occurrence patterns."""

    return await db.get_tool_cooccurrence()





# ── Golden Traces (best examples) ─────────────────────────────────────────────

@app.get("/api/tools/golden-traces")

async def api_golden_traces(auth: dict = Depends(require_auth), task_type: str = ""):

    """Get golden traces for task type."""

    return await db.get_golden_traces(task_type=task_type)





@app.post("/api/tools/golden-traces")

async def api_store_golden_trace(auth: dict = Depends(require_auth), req: dict = {}):

    """Store a golden trace (best execution example)."""

    return await db.store_golden_trace(

        session_id=req.get("session_id", ""),

        task_type=req.get("task_type", ""),

        tools_used=req.get("tools_used", []),

        final_answer=req.get("final_answer", ""),

        quality_score=req.get("quality_score", 5.0),

    )





@app.get("/api/metrics")

async def api_metrics(auth: dict = Depends(require_auth)):

    from ..environment import ENVIRONMENT

    from ..core.llm import GLOBAL_USAGE

    env = await ENVIRONMENT.get_state()

    return {

        "version": "8.0.0",

        "agent_name": config.get("agent_name", "AgentNexus"),

        "model": config.get("default_model"),

        "reasoning_strategy": "nexus",

        "memory_enabled": config.get("enable_long_term_memory"),

        "auth_enabled": config.get("enable_auth", False),

        "usage": {

            "prompt_tokens": GLOBAL_USAGE.prompt_tokens,

            "completion_tokens": GLOBAL_USAGE.completion_tokens,

            "total_tokens": GLOBAL_USAGE.total_tokens,

            "cost_usd": round(GLOBAL_USAGE.cost_usd, 6),

        },

        "stats": {

            "memories":            len(await db.all_memories()),

            "reflections":         len(await db.recent_reflections(limit=1000)),

            "sessions":            len(await db.list_sessions()),

            "goals":               len(await db.list_goals()),

            "tasks":               len(await db.list_tasks()),

            "tool_executions":     len(await db.recent_tool_executions(limit=1000)),

            "watches":             len(await db.list_file_watches(enabled_only=False)),

            "monitors":            len(await db.list_web_monitors(enabled_only=False)),

            "schedules":           len(await db.list_nl_schedules(enabled_only=False)),

            "improvements":        len(await db.list_improvement_log(resolved=False)),

        },

        "quality": env.get("quality", {}),

        "tools_health": env.get("tools", {}),

        "anomalies": env.get("anomalies", []),

    }





@app.get("/api/events/history")

async def api_event_history(limit: int = 200, auth: dict = Depends(require_auth)):

    return BUS.history(limit=limit)





# ── Virtual Computer routes are registered via vm_router (see above). ────────

# Inline VM route definitions have been removed to avoid duplicate route conflicts.





# ── Email OAuth 2.0 (Gmail / Outlook) ────────────────────────────────────────────────
@app.get("/api/integrations/email/oauth/start")
async def api_email_oauth_start(provider: str = "gmail", auth: dict = Depends(require_auth)):
    """Start email OAuth 2.0 flow — returns the provider authorization URL.

    Provider must be "gmail" or "outlook". The user is redirected to
    Google/Microsoft to grant access.
    """
    if provider not in ("gmail", "outlook"):
        raise HTTPException(400, "provider must be 'gmail' or 'outlook'")
    # Check integration config first, fall back to global config
    intg_cfg = {}
    try:
        from ..memory import db as _db
        intg_cfg = await _db.get_integration_config("email") or {}
    except Exception:
        pass
    client_id = intg_cfg.get("oauth_client_id") or config.get("google_oauth_client_id" if provider == "gmail" else "microsoft_oauth_client_id", "")
    if not client_id:
        return {"ok": False, "error": f"OAuth not configured. Set the {provider} client ID and secret in the Email integration settings first.", "authorization_url": None}
    redirect_uri = f"{config.get('base_url', 'http://localhost:8765')}/api/integrations/email/oauth/callback"
    import uuid
    state = f"email_{provider}_{uuid.uuid4().hex[:16]}"
    from ..tools.builtin.email_oauth import build_auth_url
    auth_url = build_auth_url(provider, client_id, redirect_uri, state)
    # Store state temporarily for verification on callback
    _pending_oauth_states[state] = {"provider": provider, "redirect_uri": redirect_uri, "ts": time.time()}
    return {"ok": True, "authorization_url": auth_url, "provider": provider}


# In-memory store for pending OAuth states (volatile — survives only until callback)
_pending_oauth_states: dict[str, dict] = {}


@app.get("/api/integrations/email/oauth/callback")
async def api_email_oauth_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    """Handle OAuth 2.0 callback from Google Gmail or Microsoft Outlook.

    Exchanges the authorization code for tokens and stores them in the email
    integration config. Returns an HTML page that can close a popup window.
    """
    if error:
        return HTMLResponse(
            f"<html><body><h2>Authorization denied</h2><p>{error_description or error}</p>"
            "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
            status_code=400,
        )
    if not code or not state:
        return HTMLResponse(
            "<html><body><h2>Missing authorization code</h2>"
            "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
            status_code=400,
        )
    # Look up pending state
    pending = _pending_oauth_states.pop(state, None)
    if not pending:
        return HTMLResponse(
            "<html><body><h2>Invalid or expired state</h2>"
            "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
            status_code=400,
        )
    provider: str = pending["provider"]
    redirect_uri: str = pending["redirect_uri"]
    # Check integration config first, fall back to global config
    intg_cfg = {}
    try:
        from ..memory import db as _db
        intg_cfg = await _db.get_integration_config("email") or {}
    except Exception:
        pass
    if provider == "gmail":
        client_id = intg_cfg.get("oauth_client_id") or config.get("google_oauth_client_id", "")
        client_secret = intg_cfg.get("oauth_client_secret") or config.get("google_oauth_client_secret", "")
    elif provider == "outlook":
        client_id = intg_cfg.get("oauth_client_id") or config.get("microsoft_oauth_client_id", "")
        client_secret = intg_cfg.get("oauth_client_secret") or config.get("microsoft_oauth_client_secret", "")
    else:
        return HTMLResponse("<html><body><h2>Unknown provider</h2></body></html>", status_code=400)
    if not client_id or not client_secret:
        return HTMLResponse(
            "<html><body><h2>OAuth not configured</h2><p>Client ID or secret missing.</p>"
            "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
            status_code=400,
        )
    try:
        from ..tools.builtin.email_oauth import exchange_code, refresh_token as _rt
        token_data = await exchange_code(provider, code, client_id, client_secret, redirect_uri)
        try:
            from ..memory import db as _db
            existing = await _db.get_integration_config("email") or {}
            existing["provider"] = provider
            existing["oauth_token"] = token_data
            existing["smtp_host"] = "smtp.gmail.com" if provider == "gmail" else "smtp.office365.com"
            existing["smtp_port"] = 587
            existing["imap_host"] = "imap.gmail.com" if provider == "gmail" else "outlook.office365.com"
            existing["imap_port"] = 993
            existing["smtp_user"] = token_data.get("email", "")
            existing["email_address"] = token_data.get("email", "")
            # Save to email integration config
            await _db.upsert_integration(
                intg_id="email",
                name="Email (SMTP)",
                category="communication",
                description="Send and receive emails via OAuth",
                connected=True,
                config=existing,
            )
            success = True
        except Exception as e:
            return HTMLResponse(
                f"<html><body><h2>Token received but save failed</h2><p>{e}</p>"
                "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
                status_code=500,
            )
    except Exception as e:
        return HTMLResponse(
            f"<html><body><h2>Token exchange failed</h2><p>{e}</p>"
            "<script>if(window.opener)window.close();else window.location.href='/';</script></body></html>",
            status_code=400,
        )
    # Return success HTML that notifies parent window
    provider_name = "Google Gmail" if provider == "gmail" else "Microsoft Outlook"
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Connected</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex;
         align-items: center; justify-content: center; height: 100vh;
         margin: 0; background: #0a0a0f; color: #e2e8f0; }}
  .card {{ text-align: center; padding: 40px; }}
  .check {{ width: 48px; height: 48px; border-radius: 50%;
            background: #10b981; display: inline-flex; align-items: center;
            justify-content: center; font-size: 24px; color: white;
            margin-bottom: 16px; }}
  h2 {{ margin: 0 0 8px; }}
  p {{ color: #94a3b8; }}
</style></head>
<body>
<div class="card">
  <div class="check">✓</div>
  <h2>{provider_name} Connected!</h2>
  <p>You can close this window and return to Nexus.</p>
</div>
<script>
  if (window.opener) {{
    window.opener.postMessage({{ type: 'email_oauth_success', provider: '{provider}' }}, '*');
    setTimeout(() => window.close(), 500);
  }}
</script>
</body></html>"""
    return HTMLResponse(html)


# ── OAuth 2.0 Callback (legacy, for integration_tools HUB) ────────────────────────

@app.get("/api/oauth/callback")

async def api_oauth_callback(code: str = "", state: str = "", service_id: str = "google_drive", auth: dict = Depends(optional_auth)):

    """Handle OAuth 2.0 redirect callback.

    

    For localhost/self-hosted installs, add a redirect_uri like:

      http://localhost:8765/api/oauth/callback?service_id=google_drive

    

    The OAuth provider redirects here with ?code=... after user authorization.

    This endpoint exchanges the code for tokens and stores them.

    """

    from ..tools.builtin.integration_tools import HUB

    if not code:

        return JSONResponse({"error": "No authorization code provided"}, status_code=400)

    

    result = await HUB.complete_oauth_flow(service_id, code)

    if result and "error" not in result:

        return JSONResponse({

            "ok": True,

            "message": f"OAuth tokens stored for {service_id}",

            "token_type": result.get("token_type", "Bearer"),

            "expires_in": result.get("expires_in"),

        })

    return JSONResponse({"error": result.get("error", "Token exchange failed") if result else "No active OAuth flow"}, status_code=400)





@app.get("/api/oauth/start")

async def api_oauth_start(service_id: str = "google_drive", redirect_base: str = "", auth: dict = Depends(require_auth)):

    """Start an OAuth 2.0 flow and return the authorization URL.

    

    For localhost, use redirect_base=http://localhost:8765/api/oauth/callback

    """

    from ..tools.builtin.integration_tools import HUB

    redirect_uri = f"{redirect_base or 'http://localhost:8765'}/api/oauth/callback?service_id={service_id}"

    oauth_state = HUB.start_oauth_flow(service_id, redirect_uri=redirect_uri)

    if not oauth_state:

        return JSONResponse({"error": f"OAuth not supported for {service_id}"}, status_code=400)

    return {"authorization_url": oauth_state.authorization_url, "service_id": service_id}





# ---------------- WebSocket: live event stream + chat (with auth) ----------------



@app.get("/api/notifications")

async def api_notifications(unread_only: bool = False, auth: dict = Depends(require_auth)):

    return await db.list_notifications(unread_only=unread_only)



@app.post("/api/notifications/read")

async def api_mark_read(auth: dict = Depends(require_auth)):

    await db.mark_notifications_read()

    return {"ok": True}



@app.get("/api/notifications/count")

async def api_notification_count(auth: dict = Depends(require_auth)):

    return {"unread": await db.unread_notification_count()}





# ── Workspace diff ────────────────────────────────────────────────────────────

@app.get("/api/workspace/diff")

async def api_workspace_diff(path: str, before: str = "", after: str = "", auth: dict = Depends(require_auth)):

    import difflib

    if not before and not after:

        raise HTTPException(400, "Provide before and/or after text")

    diff = list(difflib.unified_diff(

        (before or "").splitlines(keepends=True),

        (after  or "").splitlines(keepends=True),

        fromfile=f"a/{path}",

        tofile=f"b/{path}",

        n=3,

    ))

    return {"diff": "".join(diff)[:20000], "path": path}





# ── File Upload (user → workspace) ─────────────────────────────────────────

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB



@app.post("/api/upload")

async def api_upload_file(

    file: UploadFile = File(...),

    subfolder: str = Form(""),

    auth: dict = Depends(require_auth),

):

    import re

    import asyncio as _asyncio



    workspace = Path(config.BASE_DIR) / "data" / "workspace"

    target_dir = workspace

    if subfolder:

        safe = re.sub(r"[^a-zA-Z0-9_\-/\. ]", "", subfolder).strip("/")

        target_dir = (workspace / safe).resolve()

        if workspace.resolve() not in target_dir.parents and target_dir != workspace.resolve():

            raise HTTPException(400, "Invalid subfolder")



    try:

        target_dir.mkdir(parents=True, exist_ok=True)

        raw_name = (file.filename or "upload").replace("\\", "/").split("/")[-1]

        fname = re.sub(r"[^a-zA-Z0-9_\-\. ]", "_", raw_name).strip("._") or "upload"

        if not fname:

            fname = "upload"

        dest  = target_dir / fname



        stem, suffix = Path(fname).stem, Path(fname).suffix

        counter = 0

        while dest.exists():

            counter += 1

            dest = target_dir / f"{stem}_{counter}{suffix}"



        contents = await file.read()

        if len(contents) > MAX_UPLOAD_BYTES:

            raise HTTPException(413, f"File too large: {len(contents)//1024//1024} MB (max 200 MB)")



        await _asyncio.to_thread(dest.write_bytes, contents)



        rel = dest.relative_to(workspace).as_posix()

        await emit("file_uploaded", path=rel, size=len(contents), name=dest.name)

        return {"ok": True, "path": rel, "size": len(contents), "name": dest.name}



    except HTTPException:

        raise

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise HTTPException(500, "Upload failed. Check server logs for details.") from exc





@app.get("/api/workspace/download")

async def api_workspace_download(path: str, auth: dict = Depends(require_auth)):

    from fastapi.responses import FileResponse as _FR

    workspace = Path(config.BASE_DIR) / "data" / "workspace"

    safe_path = path.replace("\\", "/")

    target = (workspace / safe_path).resolve()

    if workspace.resolve() not in target.parents and target != workspace.resolve():

        raise HTTPException(403, "Path outside workspace")

    if not target.exists() or target.is_dir():

        raise HTTPException(404, "File not found")

    return _FR(target, filename=target.name)





@app.get("/api/skills")

async def api_skills(auth: dict = Depends(optional_auth)):

    """Return all available skills with their enabled state, including custom skills."""

    builtin_skills = _get_skills_with_state()

    # Append custom skills from database

    try:

        custom_skills = await db.list_custom_skills()

        for cs in custom_skills:

            builtin_skills.append({

                "name": cs.get("name", ""),

                "description": cs.get("description", ""),

                "category": cs.get("category", "custom"),

                "icon": cs.get("icon", "PLG"),

                "enabled": bool(cs.get("enabled", 1)),

                "is_custom": True,

                "custom_skill_id": cs.get("id"),

                "source_type": cs.get("source_type", "upload"),

            })

    except Exception:

        pass

    return builtin_skills





@app.post("/api/skills/{skill_name}/toggle")

async def api_toggle_skill(skill_name: str, auth: dict = Depends(require_auth)):

    """Toggle a skill's enabled state (builtin or custom)."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    # Check if it's a builtin skill

    skills_state = config.get("skills_enabled_state", {})

    skill = next((s for s in SKILLS_REGISTRY if s["name"] == skill_name), None)

    if skill:

        current = skills_state.get(skill_name, skill["enabled"])

        skills_state[skill_name] = not current

        config.set_many({"skills_enabled_state": skills_state})

        return {"ok": True, "enabled": not current}

    # Check if it's a custom skill

    try:

        custom = await db.get_custom_skill_by_name(skill_name)

        if custom:

            new_enabled = not bool(custom.get("enabled", 1))

            await db.toggle_custom_skill(custom["id"], new_enabled)

            await custom_loader.refresh_custom_skills_cache()

            return {"ok": True, "enabled": new_enabled}

    except Exception:

        pass

    raise HTTPException(404, f"Skill '{skill_name}' not found")





# ── Custom Skill Upload Endpoints ─────────────────────────────────────────────



@app.post("/api/skills/upload")

async def api_upload_skill(

    file: UploadFile = File(...),

    auth: dict = Depends(require_auth),

):

    """Upload a custom skill as SKILL.MD or .zip file.



    Accepts:

    - SKILL.md / SKILL.MD — single skill markdown file

    - .zip — archive containing one or more SKILL.md files and companion code

    """

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required to upload skills")



    if not file.filename:

        raise HTTPException(400, "No filename provided")



    content = await file.read()

    if len(content) > 10 * 1024 * 1024:  # 10MB limit

        raise HTTPException(400, "File too large (max 10MB)")



    fname_lower = file.filename.lower()



    if fname_lower.endswith(".zip"):

        result = await custom_loader.process_skill_zip_upload(file.filename, content)

    elif fname_lower.endswith((".md", ".skill.md", ".skill")):

        result = await custom_loader.process_skill_md_upload(file.filename, content)

    else:

        raise HTTPException(400, "Unsupported file type. Upload SKILL.md or .zip files only.")



    if "error" in result:

        raise HTTPException(400, result["error"])



    # Refresh cache so agent is aware of new skill

    await custom_loader.refresh_custom_skills_cache()



    # Invalidate prompt cache so agent picks up new skill

    from ..reasoning.engine import _prompt_caches, _prompt_cache_compressed_caches

    _prompt_caches.clear()

    _prompt_cache_compressed_caches.clear()



    await emit("custom_skill_added", name=result.get("name", "unknown"))

    return result





@app.get("/api/skills/custom")

async def api_list_custom_skills(auth: dict = Depends(optional_auth)):

    """List all custom skills with full details."""

    try:

        return await db.list_custom_skills()

    except Exception:

        return []





@app.get("/api/skills/custom/{skill_id}")

async def api_get_custom_skill(skill_id: int, auth: dict = Depends(optional_auth)):

    """Get a single custom skill's details."""

    skill = await db.get_custom_skill(skill_id)

    if not skill:

        raise HTTPException(404, "Custom skill not found")

    return skill





@app.delete("/api/skills/custom/{skill_id}")

async def api_delete_custom_skill(skill_id: int, auth: dict = Depends(require_auth)):

    """Delete a custom skill by ID."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    skill = await db.get_custom_skill(skill_id)

    if not skill:

        raise HTTPException(404, "Custom skill not found")

    # Remove from tool registry

    skill_name = skill.get("name", "")

    try:

        REGISTRY.unregister(skill_name)

    except Exception:

        pass

    # Remove from database

    await db.delete_custom_skill(skill_id)

    # Remove files from disk

    await custom_loader.delete_custom_skill_files(skill_name)

    # Refresh cache

    await custom_loader.refresh_custom_skills_cache()

    # Invalidate prompt cache

    from ..reasoning.engine import _prompt_caches, _prompt_cache_compressed_caches

    _prompt_caches.clear()

    _prompt_cache_compressed_caches.clear()

    await emit("custom_skill_removed", name=skill_name)

    return {"ok": True}





# ── Task Complexity Analysis ──────────────────────────────────────────────────

class TaskAnalysisRequest(BaseModel):

    message: str





@app.post("/api/analyze-task")

async def api_analyze_task(req: TaskAnalysisRequest, auth: dict = Depends(require_auth)):

    """Analyze task complexity for the Nexus Framework.



    Returns complexity assessment. The Nexus Framework automatically

    adapts its reasoning strategy — no manual mode selection needed.

    """

    # Simple complexity heuristic based on message length and keywords

    low = req.message.lower()

    complexity = "simple"

    if len(req.message) > 200 or any(kw in low for kw in ["complex", "multi-step", "research", "comprehensive"]):

        complexity = "complex"

    elif len(req.message) > 80 or any(kw in low for kw in ["analyze", "compare", "explain", "plan"]):

        complexity = "moderate"

    return {

        "complexity": complexity,

        "reasoning_strategy": "nexus",

        "confidence": 0.9,

        "explanation": f"Nexus Framework adaptive reasoning — automatically optimizes strategy for {complexity} tasks",

        "needs_suggestion": False,

        "enable_multi_agent": False,

    }





# ── Integrations ─────────────────────────────────────────────────────────────

class IntegrationToggle(BaseModel):

    connected: bool



class IntegrationConfigUpdate(BaseModel):

    config: dict[str, Any]





@app.get("/api/integrations")

async def api_integrations(auth: dict = Depends(require_auth)):

    """Return all integrations with their connection state from the database."""

    return await db.list_connected_integrations(connected_only=False)





@app.post("/api/integrations/{intg_id}/toggle")

async def api_toggle_integration(intg_id: str, req: IntegrationToggle, auth: dict = Depends(require_auth)):

    """Toggle an integration's connected/disconnected state."""

    new_state = await db.toggle_integration(intg_id, req.connected)

    await emit("integration_toggled", integration_id=intg_id, connected=new_state)

    return {"ok": True, "id": intg_id, "connected": new_state}





@app.post("/api/integrations/{intg_id}/config")

async def api_update_integration_config(intg_id: str, req: IntegrationConfigUpdate, auth: dict = Depends(require_auth)):

    """Update an integration's configuration (credentials, tokens, etc.).



    Also ensures the integration row exists via upsert so toggling works

    even if the user saved config before ever toggling connected state.

    """

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")



    # Lookup integration metadata from known definitions for name/category/description

    _KNOWN_INTEGRATIONS = {

        'slack': ('Slack', 'communication', 'Send messages, channels, threads, and files'),

        'discord': ('Discord', 'communication', 'Channels, DMs, server management, and webhooks'),

        'email': ('Email (SMTP)', 'communication', 'Send and read emails, manage inbox and labels'),

        'github': ('GitHub', 'development', 'Repos, issues, PRs, commits, and code review'),

        'gitlab': ('GitLab', 'development', 'Projects, merge requests, pipelines, and registry'),

        'vercel': ('Vercel', 'development', 'Deployments, previews, and project configuration'),

        'notion': ('Notion', 'productivity', 'Pages, databases, wikis, and content management'),

        'jira': ('Jira', 'productivity', 'Issues, sprints, boards, and project tracking'),

        'linear': ('Linear', 'productivity', 'Issues, cycles, projects, and roadmap planning'),

        'trello': ('Trello', 'productivity', 'Boards, lists, cards, and automation workflows'),

        'asana': ('Asana', 'productivity', 'Tasks, projects, timelines, and team management'),

        'google_drive': ('Google Drive', 'storage', 'Files, folders, sharing, and document access'),

        'dropbox': ('Dropbox', 'storage', 'Cloud storage, file sync, and team collaboration'),

        'aws_s3': ('AWS S3', 'storage', 'Object storage, buckets, and cloud data management'),

        'airtable': ('Airtable', 'data', 'Spreadsheets, databases, and workflow automation'),

        'google_sheets': ('Google Sheets', 'data', 'Spreadsheets, formulas, charts, and data processing'),

        'hubspot': ('HubSpot', 'crm', 'Contacts, deals, tickets, and marketing automation'),

        'salesforce': ('Salesforce', 'crm', 'Leads, opportunities, accounts, and sales workflows'),

        'openai': ('OpenAI', 'ai', 'GPT models, embeddings, DALL-E, and function calling'),

        'zapier': ('Zapier', 'ai', 'Automate workflows across 5000+ apps and services'),

        'grafana': ('Grafana', 'monitoring', 'Dashboards, metrics, alerts, and data visualization'),

        'pagerduty': ('PagerDuty', 'monitoring', 'Incident management, on-call schedules, and escalation'),

        'posthog': ('PostHog', 'analytics', 'Open-source product analytics, session recording, and feature flags'),

        'amplitude': ('Amplitude', 'analytics', 'Product analytics, cohorts, and behavioral data'),

        'mixpanel': ('Mixpanel', 'analytics', 'Product analytics, funnels, and engagement tracking'),

        'heap': ('Heap', 'analytics', 'Digital experience analytics and session replay'),

        'hotjar': ('Hotjar', 'analytics', 'Heatmaps, session recordings, and user feedback'),

        'terraform': ('Terraform', 'infrastructure', 'Infrastructure as Code, provisioning, and state management'),

        'ansible': ('Ansible', 'infrastructure', 'Configuration management, automation, and CI/CD'),

        'kubernetes': ('Kubernetes', 'infrastructure', 'Container orchestration, clusters, and deployments'),

        'docker': ('Docker', 'infrastructure', 'Container registry, builds, and image management'),

        'harbor': ('Harbor', 'infrastructure', 'Container registry, vulnerability scanning, and replication'),

        'portainer': ('Portainer', 'infrastructure', 'Container management UI and cluster visualization'),

        'cloudflare': ('Cloudflare', 'domain', 'DNS management, SSL/TLS, CDN, and edge computing'),

        'route53': ('Route53', 'domain', 'AWS DNS, domain registration, and traffic flow'),

        'godaddy': ('GoDaddy', 'domain', 'Domain registration, hosting, and website building'),

        'namecheap': ('Namecheap', 'domain', 'Domain registration, SSL certificates, and hosting'),

        'letsencrypt': ('Let\'s Encrypt', 'domain', 'Free SSL/TLS certificates and automated renewal'),

        'zerossl': ('ZeroSSL', 'domain', 'Free SSL certificates, ACME automation, and certificate management'),

        'aws': ('AWS', 'cloud', 'EC2, S3, Lambda, and cloud infrastructure'),

        'gcp': ('GCP', 'cloud', 'Compute, Cloud Functions, Cloud Run, and cloud services'),

        'azure': ('Azure', 'cloud', 'Virtual machines, containers, functions, and cloud services'),

        'vercel': ('Vercel', 'hosting', 'Serverless deployments, preview URLs, and CI/CD'),

        'netlify': ('Netlify', 'hosting', 'Static hosting, forms, and edge functions'),

        'supabase': ('Supabase', 'database', 'PostgreSQL, auth, and realtime subscriptions'),

        'firebase': ('Firebase', 'database', 'Firestore, auth, hosting, and cloud functions'),

        'dockerhub': ('Docker Hub', 'registry', 'Public container images and automated builds'),

        'ghcr': ('GHCR', 'registry', 'GitHub Container Registry for private images'),

        'appstore': ('App Store', 'distribution', 'iOS app distribution and TestFlight'),

        'googleplay': ('Google Play', 'distribution', 'Android app distribution and internal testing'),

        'youtube': ('YouTube', 'media', 'Upload, search, and manage YouTube videos'),

         'youtube': ('YouTube', 'media', 'Upload, search, and manage YouTube videos'),

         'sentry': ('Sentry', 'monitoring', 'Error tracking, performance monitoring, and release health'),

         'paypal': ('PayPal', 'payments', 'Payment processing'),

         'lemon_squeezy': ('Lemon Squeezy', 'payments', 'Digital products'),

         'shopify': ('Shopify', 'ecommerce', 'Online store'),

         'woocommerce': ('WooCommerce', 'ecommerce', 'WordPress commerce'),

         'bigcommerce': ('BigCommerce', 'ecommerce', 'Online store'),

         'twilio': ('Twilio', 'communication', 'SMS, voice, video'),

         'sendgrid': ('SendGrid', 'communication', 'Email delivery'),

         'mailgun': ('Mailgun', 'communication', 'Email API'),

         'postmark': ('Postmark', 'communication', 'Email delivery'),

         'telegram': ('Telegram', 'communication', 'Bot messaging, groups, channels'),

         'microsoft_teams': ('MS Teams', 'communication', 'Chat, meetings, channels'),

         'bitbucket': ('Bitbucket', 'development', 'Repos, pipelines, pull requests'),

         'circleci': ('CircleCI', 'development', 'Continuous integration and delivery'),

         'jenkins': ('Jenkins', 'development', 'Automation server, CI/CD pipelines'),

         'confluence': ('Confluence', 'productivity', 'Documentation, wikis, collaboration'),

         'clickup': ('ClickUp', 'productivity', 'Tasks, docs, goals, chat'),

         'monday': ('Monday.com', 'productivity', 'Work management, boards, automations'),

         'basecamp': ('Basecamp', 'productivity', 'Project management, messaging'),

         'backblaze': ('Backblaze B2', 'storage', 'Cloud object storage'),

         'wasabi': ('Wasabi', 'storage', 'Hot cloud storage'),

         'aws': ('AWS', 'cloud', 'EC2, S3, Lambda, and cloud infrastructure'),

         'gcp': ('GCP', 'cloud', 'Compute, Cloud Functions, Cloud Run, and cloud services'),

         'azure': ('Azure', 'cloud', 'Virtual machines, containers, functions, and cloud services'),

         'vercel': ('Vercel', 'hosting', 'Serverless deployments, preview URLs, and CI/CD'),

         'netlify': ('Netlify', 'hosting', 'Static hosting, forms, and edge functions'),

         'supabase': ('Supabase', 'database', 'PostgreSQL, auth, and realtime subscriptions'),

         'firebase': ('Firebase', 'database', 'Firestore, auth, hosting, and cloud functions'),

         'planetscale': ('PlanetScale', 'database', 'Serverless MySQL'),

         'dockerhub': ('Docker Hub', 'registry', 'Public container images and automated builds'),

         'ghcr': ('GHCR', 'registry', 'GitHub Container Registry for private images'),

         'appstore': ('App Store', 'distribution', 'iOS app distribution and TestFlight'),

         'googleplay': ('Google Play', 'distribution', 'Android app distribution and internal testing'),

         'sentry': ('Sentry', 'monitoring', 'Error tracking, performance monitoring, and release health'),
         'anthropic': ('Anthropic', 'ai', 'Claude API access and model inference'),
         'cohere': ('Cohere', 'ai', 'NLP and embedding API access'),
         'google_ai': ('Google AI', 'ai', 'Gemini and Vertex AI model access'),
         'huggingface': ('Hugging Face', 'ai', 'Model inference and dataset access'),
         'replicate': ('Replicate', 'ai', 'Model API access and inference'),
         'stabilityai': ('Stability AI', 'ai', 'Image generation model API'),
         'plausible': ('Plausible', 'analytics', 'Privacy-friendly web analytics'),
         'segment': ('Segment', 'analytics', 'Customer data platform and events'),
         'ifttt': ('IFTTT', 'development', 'Webhook-based automation triggers'),
         'make': ('Make', 'development', 'Visual automation workflows (Integromat)'),
         'close': ('Close', 'crm', 'Sales CRM and communication platform'),
         'pipedrive': ('Pipedrive', 'crm', 'Sales pipeline and CRM platform'),
         'zoho': ('Zoho', 'crm', 'CRM and business software suite'),
         'porkbun': ('Porkbun', 'domain', 'Domain registration and DNS management'),
         'cloudways': ('Cloudways', 'hosting', 'Managed cloud hosting platform'),
         'digitalocean': ('DigitalOcean', 'hosting', 'Cloud infrastructure and hosting'),
         'flyio': ('Fly.io', 'hosting', 'Edge application hosting platform'),
         'heroku': ('Heroku', 'hosting', 'Cloud platform as a service'),
         'hostinger': ('Hostinger', 'hosting', 'Web hosting and VPS platform'),
         'railway': ('Railway', 'hosting', 'App hosting and deployment platform'),
         'render': ('Render', 'hosting', 'Cloud application hosting'),
         'datadog': ('Datadog', 'monitoring', 'Infrastructure and APM monitoring'),
         'newrelic': ('New Relic', 'monitoring', 'Application performance monitoring'),
         'pingdom': ('Pingdom', 'monitoring', 'Website uptime and performance monitoring'),
         'statuspage': ('StatusPage', 'monitoring', 'Service status and incident management'),
         'uptimerobot': ('UptimeRobot', 'monitoring', 'Website and server monitoring'),


    }



    meta = _KNOWN_INTEGRATIONS.get(intg_id, (intg_id, 'custom', 'Custom API integration'))

    connected = bool(req.config)  # Mark as connected if config has data



    # Upsert to ensure row exists

    await db.upsert_integration(

        intg_id=intg_id,

        name=meta[0],

        category=meta[1],

        description=meta[2],

        connected=connected,

        config=req.config,

    )

    await emit("integration_config_updated", integration_id=intg_id)

    return {"ok": True, "connected": connected}





@app.get("/api/integrations/{intg_id}")

async def api_get_integration(intg_id: str, auth: dict = Depends(require_auth)):

    """Get a single integration's details."""

    row = await db.get_integration(intg_id)

    if not row:

        raise HTTPException(404, f"Integration '{intg_id}' not found")

    return row





@app.get("/api/integrations/connected/ids")

async def api_connected_integration_ids(auth: dict = Depends(require_auth)):

    """Return just the list of connected integration IDs (for lightweight context injection)."""

    rows = await db.list_connected_integrations(connected_only=True)

    return {"ids": [r["id"] for r in rows]}





@app.get("/api/integrations/context")

async def api_integration_context(auth: dict = Depends(require_auth)):

    """Return integration context text for injection into the agent system prompt."""

    rows = await db.list_connected_integrations(connected_only=True)

    if not rows:

        return {"context": ""}

    lines = [f"  - {r['name']} ({r['id']}): {r.get('description', '')}" for r in rows]

    context = (

        "Available connected integrations:\n"

        + "\n".join(lines)

        + "\nYou can use these integrations to perform actions on behalf of the user. "

        "Reference them when relevant."

    )

    return {"context": context}





@app.get("/api/integrations/custom/list")

async def api_list_custom_integrations(auth: dict = Depends(require_auth)):

    """Return all custom API integrations."""

    rows = await db.list_connected_integrations(connected_only=False)

    custom = [r for r in rows if r.get("id", "").startswith("custom_api_")]

    return custom





@app.delete("/api/integrations/{intg_id}")

async def api_delete_integration(intg_id: str, auth: dict = Depends(require_auth)):

    """Delete an integration entirely from the database."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    await db.delete_integration(intg_id)

    await emit("integration_deleted", integration_id=intg_id)

    return {"ok": True}





# ── Webhook Receiver (for external integrations) ──────────────────────

@app.post("/api/integrations/webhook/{service_id}/{webhook_id}")

async def api_webhook_receiver(

    service_id: str,

    webhook_id: str,

    req: dict = {},

    x_hub_signature: str | None = Header(None, alias="X-Hub-Signature-256"),

):

    """Receive webhook events from external services."""

    from ..tools.builtin.integration_tools import WEBHOOK_MANAGER

    

    # Verify signature if secret is set

    if x_hub_signature:

        verified = WEBHOOK_MANAGER.verify_signature(service_id, webhook_id, req, x_hub_signature)

        if not verified:

            raise HTTPException(401, "Invalid signature")

    

    # Process the webhook event

    event_type = req.get("type", "unknown")

    await WEBHOOK_MANAGER._handle_webhook_event(service_id, webhook_id, event_type, req)

    

    return {"ok": True}





# ── MCP Servers ────────────────────────────────────────────────────────────────

from ..tools.builtin.mcp_manager import MCP_MANAGER



class MCPServerCreate(BaseModel):

    name: str

    transport: str = "stdio"

    command: str = ""

    args: list[str] = []

    env: dict[str, str] = {}

    url: str = ""

    headers: dict[str, str] = {}

    description: str = ""

    category: str = "custom"

    auto_connect: bool = False

    config: dict[str, Any] = {}



class MCPServerUpdate(BaseModel):

    name: str | None = None

    transport: str | None = None

    command: str | None = None

    args: list[str] | None = None

    env: dict[str, str] | None = None

    url: str | None = None

    headers: dict[str, str] | None = None

    description: str | None = None

    category: str | None = None

    auto_connect: bool | None = None

    config: dict[str, Any] | None = None





@app.get("/api/mcp/servers")

async def api_list_mcp_servers(auth: dict = Depends(require_auth)):

    """List all MCP servers."""

    return MCP_MANAGER.list_servers()





@app.get("/api/mcp/servers/connected")

async def api_list_connected_mcp_servers(auth: dict = Depends(require_auth)):

    """List connected MCP servers."""

    return MCP_MANAGER.list_connected_servers()





@app.post("/api/mcp/servers")

async def api_create_mcp_server(req: MCPServerCreate, auth: dict = Depends(require_auth)):

    """Create a new MCP server configuration."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    server = await MCP_MANAGER.add_server(

        name=req.name, transport=req.transport, command=req.command,

        args=req.args, env=req.env, url=req.url, headers=req.headers,

        description=req.description, category=req.category,

        auto_connect=req.auto_connect, config=req.config,

    )

    return {"ok": True, "server": server.to_dict()}





@app.post("/api/mcp/servers/{mcp_id}/connect")

async def api_connect_mcp_server(mcp_id: str, auth: dict = Depends(require_auth)):

    """Connect to an MCP server."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    try:

        result = await MCP_MANAGER.connect(mcp_id)

        return result

    except Exception as e:

        raise HTTPException(500, str(e))





@app.post("/api/mcp/servers/{mcp_id}/disconnect")

async def api_disconnect_mcp_server(mcp_id: str, auth: dict = Depends(require_auth)):

    """Disconnect from an MCP server."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    try:

        result = await MCP_MANAGER.disconnect(mcp_id)

        return result

    except Exception as e:

        raise HTTPException(500, str(e))





@app.post("/api/mcp/servers/{mcp_id}/test")

async def api_test_mcp_server(mcp_id: str, auth: dict = Depends(require_auth)):

    """Test connection to an MCP server."""

    return await MCP_MANAGER.test_connection(mcp_id)





@app.get("/api/mcp/servers/{mcp_id}")

async def api_get_mcp_server(mcp_id: str, auth: dict = Depends(require_auth)):

    """Get a single MCP server's details."""

    row = await db.get_mcp_server(mcp_id)

    if not row:

        raise HTTPException(404, f"MCP server '{mcp_id}' not found")

    return row





@app.put("/api/mcp/servers/{mcp_id}")

async def api_update_mcp_server(mcp_id: str, req: MCPServerUpdate, auth: dict = Depends(require_auth)):

    """Update an MCP server's configuration."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    existing = await db.get_mcp_server(mcp_id)

    if not existing:

        raise HTTPException(404, f"MCP server '{mcp_id}' not found")

    updates = req.model_dump(exclude_none=True)

    merged = {

        "name": updates.get("name", existing["name"]),

        "transport": updates.get("transport", existing["transport"]),

        "command": updates.get("command", existing.get("command", "")),

        "args": updates.get("args", json.loads(existing.get("args", "[]"))),

        "env": updates.get("env", json.loads(existing.get("env", "{}"))),

        "url": updates.get("url", existing.get("url", "")),

        "headers": updates.get("headers", json.loads(existing.get("headers", "{}"))),

        "description": updates.get("description", existing.get("description", "")),

        "category": updates.get("category", existing.get("category", "custom")),

        "auto_connect": updates.get("auto_connect", bool(existing.get("auto_connect", 0))),

        "config": updates.get("config", json.loads(existing.get("config", "{}"))),

    }

    await db.upsert_mcp_server(mcp_id=mcp_id, **merged)

    return {"ok": True}





@app.delete("/api/mcp/servers/{mcp_id}")

async def api_delete_mcp_server(mcp_id: str, auth: dict = Depends(require_auth)):

    """Delete an MCP server entirely."""

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    await MCP_MANAGER.remove_server(mcp_id)

    return {"ok": True}





@app.get("/api/mcp/tools")

async def api_list_mcp_tools(auth: dict = Depends(require_auth)):

    """List all tools from all connected MCP servers.



    v22: Also includes which tools are registered in the REGISTRY.

    """

    mcp_tools = MCP_MANAGER.get_all_tools()

    registry_map = MCP_MANAGER.get_all_registry_mcp_tools()

    return {

        "tools": mcp_tools,

        "registry_registered": registry_map,

        "total_mcp_tools": len(mcp_tools),

        "total_in_registry": sum(len(v) for v in registry_map.values()),

    }





@app.get("/api/mcp/context")

async def api_mcp_context(auth: dict = Depends(require_auth)):

    """Return MCP context text for agent system prompt injection."""

    return {"context": MCP_MANAGER.get_tools_for_prompt()}





@app.post("/api/mcp/sync-registry")

async def api_mcp_sync_registry(auth: dict = Depends(require_auth)):

    """v22: Synchronize MCP tools with the tool REGISTRY.



    Ensures all connected MCP server tools are properly registered

    in the REGISTRY and stale entries are cleaned up.

    """

    if auth.get("role") != "admin":

        raise HTTPException(403, "Admin access required")

    result = await MCP_MANAGER.sync_registry()

    return {"ok": True, **result}





# ── Triggers (Backend Persistence) ─────────────────────────────────────────────

class TriggerCreate(BaseModel):

    id: str = ""

    name: str

    type: str

    integration_id: str = ""

    config: dict[str, Any] = {}

    enabled: bool = True




@app.get("/api/triggers")

async def api_list_triggers(auth: dict = Depends(require_auth)):

    """List all triggers."""

    return await db.list_triggers()





@app.get("/api/triggers/enabled")

async def api_list_enabled_triggers(auth: dict = Depends(require_auth)):

    """List only enabled triggers."""

    return await db.list_triggers(enabled_only=True)





@app.post("/api/triggers")

async def api_create_trigger(req: TriggerCreate, auth: dict = Depends(require_auth)):

    """Create or update a trigger."""

    trigger_id = req.id or str(uuid.uuid4())

    await db.upsert_trigger(

        trigger_id=trigger_id, name=req.name,

        trigger_type=req.type, integration_id=req.integration_id,

        config=req.config, enabled=req.enabled,

    )

    await emit("trigger_updated", trigger_id=trigger_id)

    return {"ok": True, "id": trigger_id}





@app.post("/api/triggers/{trigger_id}/toggle")

async def api_toggle_trigger(trigger_id: str, req: dict, auth: dict = Depends(require_auth)):

    """Toggle a trigger's enabled state."""

    enabled = req.get("enabled", True)

    await db.toggle_trigger(trigger_id, enabled)

    await emit("trigger_updated", trigger_id=trigger_id, enabled=enabled)

    return {"ok": True, "id": trigger_id, "enabled": enabled}





@app.delete("/api/triggers/{trigger_id}")

async def api_delete_trigger(trigger_id: str, auth: dict = Depends(require_auth)):

    """Delete a trigger."""

    await db.delete_trigger(trigger_id)

    await emit("trigger_updated", trigger_id=trigger_id)

    return {"ok": True}





@app.post("/api/triggers/{trigger_id}/config")

async def api_configure_trigger(trigger_id: str, req: dict, auth: dict = Depends(require_auth)):

    """Configure a specific trigger (set config and/or enabled state)."""

    config = req.get("config", {})

    enabled = req.get("enabled")

    # Look up existing trigger to preserve name/type if available

    existing = None

    try:

        triggers = await db.list_triggers()

        existing = next((t for t in triggers if t.get("id") == trigger_id), None)

    except Exception:

        pass

    name = existing.get("name", trigger_id) if existing else trigger_id

    trigger_type = existing.get("type", "custom") if existing else "custom"

    integration_id = existing.get("integration_id", "") if existing else ""

    if enabled is None and existing:

        enabled = existing.get("enabled", True)

    elif enabled is None:

        enabled = True

    await db.upsert_trigger(

        trigger_id=trigger_id, name=name,

        trigger_type=trigger_type, integration_id=integration_id,

        config=config, enabled=enabled,

    )

    await emit("trigger_updated", trigger_id=trigger_id, enabled=enabled)

    return {"ok": True, "id": trigger_id, "enabled": enabled}





@app.websocket("/ws")

async def ws_endpoint(ws: WebSocket):

    # CRITICAL FIX: Validate auth BEFORE accepting the WebSocket connection.

    # Previously, ws.accept() was called first, allowing unauthenticated clients

    # to briefly connect and receive EventBus events.

    token = ws.query_params.get("token", "")

    auth_payload = None



    if token:

        from ..auth import verify_token

        auth_payload = verify_token(token)



    if not auth_payload and config.get("enable_auth", False):

        # Can't read messages before accept in FastAPI, so we accept,

        # validate, and immediately close if auth fails.

        await ws.accept()

        try:

            first_msg = await asyncio.wait_for(ws.receive_text(), timeout=10)

            payload = json.loads(first_msg)

            if payload.get("type") == "auth":

                auth_payload = authenticate(payload.get("username", ""), payload.get("password", ""))

                if auth_payload:

                    auth_payload = {"sid": payload.get("username"), "role": "user",

                                    "exp": time.time() + 999999}

                if not auth_payload:

                    from ..auth import verify_token as _vt

                    auth_payload = _vt(payload.get("token", ""))

        except (asyncio.TimeoutError, json.JSONDecodeError):

            await ws.close(code=4001, reason="Authentication required")

            return



        if not auth_payload:

            await ws.close(code=4001, reason="Authentication required")

            return

    else:

        # No auth required or already authenticated via token — safe to accept

        await ws.accept()



    if auth_payload:

        await ws.send_json({"kind": "auth_ok", "data": {"session_id": auth_payload.get("sid")}})



    q = BUS.subscribe()

    async def sender():
        try:
            while True:
                event = await q.get()
                # Removed debug print that was flooding stdout on every event
                await ws.send_json(event.to_dict())
        except Exception as exc:
            logger.warning("WebSocket sender died: %s", exc)



    send_task = asyncio.create_task(sender())



    try:

        # WebSocket message size limit and idle timeout

        _WS_MAX_MSG_SIZE = 50 * 1024 * 1024  # 50 MB max per message (supports file uploads)

        _WS_IDLE_TIMEOUT = 600  # 10 minutes idle timeout



        while True:

            try:

                msg = await asyncio.wait_for(ws.receive_text(), timeout=_WS_IDLE_TIMEOUT)

            except asyncio.TimeoutError:
                try:
                    await ws.close(code=1000, reason="Idle timeout")
                except Exception:
                    pass  # Client may have already disconnected
                break



            if len(msg) > _WS_MAX_MSG_SIZE:

                await ws.send_json({"kind": "error", "data": {"error": "Message too large (max 1MB)"}})

                continue



            try:

                payload = json.loads(msg)

            except json.JSONDecodeError:

                await ws.send_json({"kind": "error", "data": {"error": "Invalid JSON"}})

                continue

            if payload.get("type") == "chat":
                session_id = payload.get("session_id") or str(uuid.uuid4())
                await db.create_session(session_id, "Chat")
                user_input = payload.get("message", "")
                user_images = payload.get("images") or []

                async with _chat_tasks_lock:
                    old_task = _current_chat_tasks.get(session_id)
                    if old_task and not old_task.done():
                        old_task.cancel()
                        await ws.send_json({"kind": "interrupted", "data": {"session_id": session_id}})

                async def run_chat(sid=session_id, text=user_input, images=user_images):
                    try:
                        _current_session_id.set(sid)

                        from pathlib import Path as _P
                        _workspace = _P(config.BASE_DIR) / "data" / "workspace"
                        _workspace.mkdir(parents=True, exist_ok=True)

                        def _snapshot(ws: "_P") -> dict[str, float]:
                            snap: dict[str, float] = {}
                            for entry in ws.rglob("*"):
                                if entry.is_file():
                                    try:
                                        snap[str(entry.relative_to(ws))] = entry.stat().st_mtime
                                    except Exception:
                                        pass
                            return snap

                        _before = await asyncio.to_thread(_snapshot, _workspace)
                        await emit("task_complexity",
                               complexity="moderate",
                               reasoning_strategy="nexus",
                               confidence=0.9,
                               explanation="Nexus Framework adaptive reasoning",
                               needs_suggestion=False,
                               session_id=sid)

                        engine = await _get_engine(sid)
                        answer = await asyncio.wait_for(engine.respond(text, images=images), timeout=1800.0)
                        _after = await asyncio.to_thread(_snapshot, _workspace)
                        _new_or_modified: list[dict] = []
                        for rel_path, mtime in _after.items():
                            if rel_path not in _before or _before[rel_path] != mtime:
                                try:
                                    size = (_workspace / rel_path).stat().st_size
                                except Exception:
                                    size = 0
                                _new_or_modified.append({
                                    "name": _P(rel_path).name,
                                    "path": rel_path,
                                    "size": size,
                                })
                        await emit("final_answer", content=answer, session_id=sid, workspace_files=_new_or_modified)

                    except asyncio.CancelledError:
                        pass
                    except TimeoutError:
                        import traceback
                        traceback.print_exc()
                        await emit("error", source="chat", error="Request timed out after 1800 seconds.", session_id=sid)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        await emit("error", source="chat", error=str(e), session_id=sid)
                    finally:
                        async with _chat_tasks_lock:
                            if _current_chat_tasks.get(sid) is asyncio.current_task():
                                del _current_chat_tasks[sid]

                new_task = asyncio.create_task(run_chat())
                async with _chat_tasks_lock:
                    _current_chat_tasks[session_id] = new_task

            elif payload.get("type") == "ping":
                await ws.send_json({"kind": "pong", "data": {}})

            elif payload.get("type") == "heartbeat":
                await ws.send_json({"kind": "heartbeat_ack", "data": {"ts": time.time()}})

    except WebSocketDisconnect:

        pass

    finally:

        BUS.unsubscribe(q)

        send_task.cancel()

