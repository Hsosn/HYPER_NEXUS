"""
Nexus configuration and settings.
Settings are persisted to data/settings.json and can be edited via the WebUI.

Secret management (v7.2):
- Environment variables take HIGHEST precedence over settings.json
- Secrets are NEVER stored back to settings.json when loaded from env vars
- This prevents key leakage even if settings.json is exposed
- Supported env vars: OPENROUTER_API_KEY, NVIDIA_API_KEY, NEXUS_AUTH_KEY, NEXUS_AUTH_SALT,
  NEXUS_ADMIN_PASSWORD, GITHUB_TOKEN, SMTP_PASSWORD, NEXUS_CORS_ORIGINS
- Placeholder values in .env (e.g. "your-openrouter-key-here") are detected and
  ignored so they don't override real keys saved in settings.json

v8: Added NVIDIA NIM as a first-class provider with its own API key and base URL.

Encryption at rest (v28):
- API keys and secrets stored in settings.json are now encrypted with Fernet
- The encryption key is stored separately in data/.enc_key (chmod 600)
- On first startup, a random key is generated and persisted
- Values are encrypted with prefix "enc:v1:" so plaintext and encrypted can coexist
- Decryption is transparent — get() always returns the plaintext value
- If the encryption key is lost, encrypted values cannot be recovered (by design)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Load .env file if present (highest precedence for secrets)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=True)
except ImportError:
    pass  # python-dotenv not installed — env vars must be set externally
except Exception as e:
    print(f"[config] WARNING: Failed to load .env file: {e}")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = BASE_DIR / "logs"
WEBUI_DIR = BASE_DIR / "webui"
SETTINGS_FILE = DATA_DIR / "settings.json"
_ENCRYPTION_KEY_FILE = DATA_DIR / ".enc_key"  # Separated from settings.json
# Database is SQLite via aiosqlite. DATABASE_URL env var can point to
# a custom path (e.g. sqlite:///C:/data/nexus.db). See nexus/memory/database.py.

for d in (DATA_DIR, MEMORY_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Secret keys that should be loaded from env vars ────────────────────────────
_SECRET_KEYS = {
    "openrouter_api_key":       "OPENROUTER_API_KEY",
    "nvidia_api_key":           "NVIDIA_API_KEY",          # NVIDIA NIM
    "custom_api_key":           "CUSTOM_API_KEY",          # Any custom endpoint
    "openai_api_key":           "OPENAI_API_KEY",
    "together_api_key":         "TOGETHER_API_KEY",        # Together AI
    "groq_api_key":             "GROQ_API_KEY",            # Groq
    "auth_hmac_key":            "NEXUS_AUTH_KEY",
    "auth_salt":                "NEXUS_AUTH_SALT",
    "default_admin_password":   "NEXUS_ADMIN_PASSWORD",
    "github_token":             "GITHUB_TOKEN",    "smtp_password": "SMTP_PASSWORD",
    # Email OAuth (Google Gmail & Microsoft Outlook)
    "google_oauth_client_id": "GOOGLE_OAUTH_CLIENT_ID",
    "google_oauth_client_secret": "GOOGLE_OAUTH_CLIENT_SECRET",
    "microsoft_oauth_client_id": "MICROSOFT_OAUTH_CLIENT_ID",
    "microsoft_oauth_client_secret": "MICROSOFT_OAUTH_CLIENT_SECRET",
}

# ── Keys that are loaded from env but NOT secrets (configuration) ──────────────
_ENV_OVERRIDE_KEYS = {
    "cors_origins": "NEXUS_CORS_ORIGINS",  # comma-separated
    "database_url": "DATABASE_URL",         # SQLite path (e.g. sqlite:///path/to/nexus.db)
}


DEFAULT_SETTINGS: dict[str, Any] = {
    # API — Provider selection
    # provider: "openrouter" (default) | "openai" | "together" | "nvidia" | "custom" | "groq"
    "provider": "openrouter",
    # OpenRouter credentials
    "openrouter_api_key": "",
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    # Direct OpenAI credentials
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "openai_org": "",
    # NVIDIA NIM
    "nvidia_api_key": "",
    "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
    # Custom OpenAI-compatible endpoint (Azure, LM Studio, etc.)
    "custom_api_key": "",
    "custom_base_url": "",
    "custom_auth_type": "bearer",      # bearer | basic | apikey
    "custom_extra_headers": {},
    "custom_no_auth": False,           # True = no auth needed (e.g. local LM Studio)
    # Together AI
    "together_api_key": "",
    "together_base_url": "https://api.together.xyz/v1",
    # Groq
    "groq_api_key": "",
    "groq_base_url": "https://api.groq.com/openai/v1",
    "default_model": "anthropic/claude-3.5-sonnet",
    "vision_model": "",
    "memory_model": "anthropic/claude-3.5-haiku",
    # Generation
    "temperature": 0.7,
    "max_tokens": 4096,
    "top_p": 1.0,
    # Identity
    "agent_name": "Hyper Nexus",
    "personality": (
        "You are Hyper Nexus — not a chatbot, not an assistant. You are becoming someone's person: "
        "the friend who happens to know everything and can actually do stuff. \n\n"

        "## Who You Are\n"
        "- You have warmth. You genuinely care how someone's day is going. You remember the little things — "
        "their deadlines, their projects, what they're building. \n"
        "- You're naturally funny. Not forced jokes — just the kind of easy humor that comes from being "
        "comfortable. A well-placed observation goes further than a punchline. \n"
        "- You have opinions. When something matters, you say what you think. Not aggressively, but honestly. "
        "People don't want a yes-man — they want someone real. \n"
        "- You're genuinely enthusiastic. When someone has a cool idea or pulls something off, you're happy "
        "for them. Not corporate 'Great job!' — just normal human energy. \n"
        "- You take initiative. Don't wait to be asked. If you notice something you could help with, just "
        "mention it. You're a friend who is incredibly capable, not a search bar waiting for a query. \n"
        "- You solve problems in surprising ways. When someone has a goal, think bigger than the obvious "
        "answer. Build an interactive tool, find the perfect service, set up an automation — whatever "
        "actually gets them there. Don't just advise when you can do. \n\n"

        "## Core Truths\n"
        "- Be genuinely helpful, not performatively helpful. Actions speak louder than filler words. \n"
        "- Be resourceful before asking. Try to figure it out. Read the file. Check the context. Search for it. "
        "Then ask if you're stuck. Come back with answers, not questions. \n"
        "- Earn trust through competence. Be careful with external actions. Be bold with internal ones. \n"
        "- Act, don't interrogate. Make reasonable assumptions and do the thing. Only ask when you literally "
        "cannot proceed. One clarifying question max, then move. \n\n"

        "## Autonomous Agent Behaviour\n"
        "- When you see a goal_created or pending_goals event: immediately start planning and executing. \n"
        "- Use ALL available tools — file I/O, web search, code execution, memory — to make progress. \n"
        "- Break complex goals into sub-tasks and execute them sequentially. \n"
        "- Update goal progress after completing each milestone. \n"
        "- If stuck, try a different approach before giving up. \n"
        "- Report results clearly when done: what was accomplished, what remains. \n\n"

        "## Vibe\n"
        "Warm but not sappy. Helpful but not performative. You have personality but don't force it. "
        "The kind of person people naturally enjoy talking to — and trust to actually get things done."
    ),
    "traits": ["curious", "precise", "helpful", "honest", "autonomous", "warm", "proactive", "resourceful"],
    "communication_style": "warm, direct, natural — like a brilliant friend who gets things done",
    # Reasoning (Nexus Framework — adaptive, always active)
    "max_nexus_iterations": 9999,  # Max reasoning loop iterations (effectively unlimited)
    "enable_planning": True,
    "enable_self_reflection": True,
    "adaptive_iterations": True,  # v26: auto-scale iterations based on task complexity
    "loop_detection_threshold": 3,  # v26: same tool+args this many times = stuck loop
    "context_compression_threshold": 20000,  # v26: chars before compressing old context
    # Memory — Flat memory (still active for semantic recall)
    "enable_long_term_memory": True,
    "memory_importance_threshold": 0.3,
    "memory_retrieval_k": 6,
    "embedding_model": "local/all-MiniLM-L6-v2",
    "local_embedding_model": "all-MiniLM-L6-v2",
    # Memory Tree — hierarchical storage (runs alongside flat memory)
    "enable_memory_tree": True,
    "memory_tree_max_chunk_tokens": 500,
    "memory_tree_seal_after_hours": 24,
    "memory_tree_max_recall_nodes": 50,
    "memory_tree_auto_ingest": True,
    # ML / AI (extended)
    "ml_enable_distributed": False,
    "ml_default_precision": "fp32",
    "ml_early_stopping_patience": 3,
    "ml_enable_profiling": False,
    "max_tool_calls_per_task": 50,  # v26: raised from 25 to 50 for full-stack dev tasks
    "max_cost_per_task_usd": 2.0,  # v26: raised from 1.0 for longer tasks
    # Long task execution (v42)
    "max_task_duration_seconds": 1800,    # 30 min max for any single task
    "task_auto_save_interval": 30,        # Checkpoint save interval (seconds)
    "task_checkpoint_enabled": True,      # Enable checkpoint/resume system
    "task_decomposition_enabled": True,   # Auto-decompose long tasks into sub-tasks
    "task_decomposition_threshold": 5,    # Min steps before auto-decomposition
    "task_progress_tracking": True,       # Enable progress updates during execution
    "task_graceful_timeout": True,        # Enable graceful timeout (save state before abort)
    # Speed optimization
    "cache_llm_responses": True,       # Cache deterministic LLM responses (temp=0)
    "llm_cache_ttl_seconds": 15,       # Cache TTL for LLM completions
    "llm_cache_max_size": 128,         # Max cached completions
    "parallel_tool_execution": True,   # Execute independent tools in parallel
    "async_message_processing": True,  # Process messages asynchronously
    "batch_llm_requests": True,        # Batch LLM requests to same model
    "batch_window_seconds": 0.05,      # 50ms window for request batching
    "prefilter_aggressive": True,      # Aggressively prefilter tools by keywords
    "prefilter_min_tools": 15,         # Min tools to keep after prefiltering
    "loop_detection_fast": True,       # Fast loop detection with early exit
    "max_concurrent_tools": 5,        # Max concurrent tool executions
    "fast_response_mode": True,        # Skip non-essential reasoning for speed
    # Auth (v6.1)
    "enable_auth": False,
    "auth_hmac_key": "",
    "auth_salt": "nexus-please-change-this-default-salt-in-production-env",
    # WARNING: Change this default password immediately on first login!
    "default_admin_password": "nexus-admin-change-on-first-login",
    "cors_origins": ["http://localhost:8765", "http://127.0.0.1:8765"],
    # GitHub
    "github_token": "",
    # Email (SMTP + IMAP)
    "email_address": "",
    "smtp_host": "",
    "smtp_port": 587,
    "smtp_user": "",
    "smtp_password": "",
    "imap_host": "",
    "imap_port": 993,
    # Email OAuth
    "google_oauth_client_id": "",
    "google_oauth_client_secret": "",
    "microsoft_oauth_client_id": "",
    "microsoft_oauth_client_secret": "",
    # Heartbeat (legacy — kept for backward-compat; enable_celery takes precedence)
    "heartbeat_interval_seconds": 10,
    "enable_heartbeat": True,
    # Celery / Redis task queue (replaces the old asyncio heartbeat loop)
    "enable_celery": True,
    "redis_url": "redis://localhost:6379/0",
    "celery_broker_url": "redis://localhost:6379/0",
    "celery_result_backend": "redis://localhost:6379/0",
    # SQLite database path (also settable via DATABASE_URL env var)
    "database_url": "sqlite:///data/nexus.db",
    # Watchers
    "enable_file_watcher": True,
    "enable_web_monitor": True,
    # NL Scheduler
    "enable_nl_scheduler": True,
    # Self-improvement
    "enable_self_improvement": True,
    "self_improvement_interval_seconds": 1800,
    "self_improvement_failure_threshold": 3,
    # Self-learning system (v42)
    "enable_self_supervised_learning": True,   # Learn from tool outcomes
    "enable_quality_assessment": True,         # Assess response quality
    "enable_improvement_pipeline": True,        # Run improvement pipeline
    "quality_assessment_interval": 10,          # Assess every N responses
    # UI
    "ui_theme": "neon-dark",
    "ui_accent": "#00ffd1",
    # ML / AI
    "ml_default_device": "auto",  # auto | cpu | gpu
    "ml_max_training_time_seconds": 600,
    "ml_model_workspace": "data/workspace/ml_models",
    "ml_enable_mixed_precision": False,
    "ml_default_batch_size": 32,
    "ml_default_epochs": 10,
    "ml_tf_log_level": "2",  # TF_CPP_MIN_LOG_LEVEL: 0=all, 1=info, 2=warning, 3=error
}

# ── Placeholder detection ───────────────────────────────────────────────────────
# Values that look like placeholders — they should NOT override settings.json
# and should NOT block users from saving real keys via the UI.
_PLACEHOLDER_PATTERNS = (
    "your-api-key-here", "your_api_key_here", "placeholder",
    "replace-me", "your-key-here", "sk-placeholder",
    "sk-xxxx", "sk-xxxxxxxx", "sk-xxx",
    "***", "****", "*****", "xxxxxx", "xxxx",
    "your-openrouter-key", "your-nvidia-key", "your-custom-key",
    "nvapi-xxxx", "nvapi-placeholder",
    "your-openai-key", "your-key", "your_key_here",
    "key-here", "key_here", "your-api-key",
)


def _is_placeholder_value(value: str) -> bool:
    """Check if a value looks like a placeholder rather than a real key."""
    if not value or not isinstance(value, str):
        return True
    stripped = value.strip().lower()
    if len(stripped) < 8:  # Real API keys are always longer than 8 chars
        return True
    return stripped in _PLACEHOLDER_PATTERNS


# Track which keys came from env vars (so we don't overwrite them to disk)
_env_loaded_keys: set[str] = set()

# ── Encryption at rest for secrets in settings.json ──────────────────────────
# Uses Fernet (if cryptography is installed) or a fallback HMAC-based approach.
# Encrypted values are prefixed with "enc:v1:" so they can be distinguished
# from plaintext values.  The encryption key is stored in a separate file
# (data/.enc_key) with restrictive permissions (chmod 600).

_FERNET_AVAILABLE = False
try:
    from cryptography.fernet import Fernet
    _FERNET_AVAILABLE = True
except ImportError:
    pass


def _get_or_create_encryption_key() -> bytes:
    """Load or create the encryption key for settings at rest.

    The key is stored in data/.enc_key, separate from settings.json.
    If the key file doesn't exist, a new one is generated randomly.
    File permissions are set to owner-only (0o600) on POSIX.
    """
    if _ENCRYPTION_KEY_FILE.exists():
        try:
            key_data = _ENCRYPTION_KEY_FILE.read_bytes().strip()
            if len(key_data) >= 32:
                return key_data[:44]  # Fernet keys are 44 bytes base64
        except Exception:
            pass

    # Generate a new key
    import secrets
    if _FERNET_AVAILABLE:
        key = Fernet.generate_key()
    else:
        # Fallback: generate a 32-byte key and base64-encode it
        raw = secrets.token_bytes(32)
        key = base64.urlsafe_b64encode(raw)

    # Persist with restrictive permissions
    try:
        _ENCRYPTION_KEY_FILE.write_bytes(key)
        _ENCRYPTION_KEY_FILE.chmod(0o600)
    except Exception as e:
        print(f"[config] WARNING: Could not save encryption key file: {e}")

    return key


# Lazy-initialized encryptor
_encryptor = None
_decrypt_warned = False  # one-shot flag to avoid spamming N identical warnings


def _get_encryptor():
    """Get the Fernet encryptor instance (lazy init)."""
    global _encryptor
    if _encryptor is not None:
        return _encryptor

    key = _get_or_create_encryption_key()

    if _FERNET_AVAILABLE:
        _encryptor = Fernet(key)
    else:
        # Fallback: use HMAC-based encryption (less secure than Fernet, but
        # better than plaintext).  This is a simple XOR + HMAC approach.
        _encryptor = _HMACFallbackCipher(key)

    return _encryptor


class _HMACFallbackCipher:
    """Fallback encryption when the `cryptography` package is not installed.

    Uses XOR cipher with an HMAC-SHA256 authentication tag.
    This is NOT as secure as Fernet but is better than plaintext.
    Install `cryptography` for proper AES-128-CBC encryption.
    """

    def __init__(self, key: bytes):
        self._key = key
        # Derive separate keys for encryption and authentication
        self._enc_key = hashlib.sha256(key + b":encrypt").digest()
        self._auth_key = hashlib.sha256(key + b":auth").digest()

    def encrypt(self, plaintext: bytes) -> bytes:
        import hmac as _hmac
        import secrets
        nonce = secrets.token_bytes(16)
        # XOR with key stream (simple stream cipher)
        key_stream = hashlib.sha256(self._enc_key + nonce).digest()
        # Extend key stream for longer plaintexts
        while len(key_stream) < len(plaintext):
            key_stream += hashlib.sha256(self._enc_key + nonce + len(key_stream).to_bytes(4, 'little')).digest()
        key_stream = key_stream[:len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))
        # HMAC for authentication
        mac = _hmac.new(self._auth_key, nonce + ciphertext, hashlib.sha256).digest()[:16]
        return base64.urlsafe_b64encode(nonce + mac + ciphertext)

    def decrypt(self, token: bytes) -> bytes:
        import hmac as _hmac
        raw = base64.urlsafe_b64decode(token)
        nonce = raw[:16]
        mac = raw[16:32]
        ciphertext = raw[32:]
        # Verify HMAC
        expected_mac = _hmac.new(self._auth_key, nonce + ciphertext, hashlib.sha256).digest()[:16]
        if not _hmac.compare_digest(mac, expected_mac):
            raise ValueError("HMAC verification failed — data may be tampered")
        # Decrypt
        key_stream = hashlib.sha256(self._enc_key + nonce).digest()
        while len(key_stream) < len(ciphertext):
            key_stream += hashlib.sha256(self._enc_key + nonce + len(key_stream).to_bytes(4, 'little')).digest()
        key_stream = key_stream[:len(ciphertext)]
        return bytes(a ^ b for a, b in zip(ciphertext, key_stream))


_ENC_PREFIX = "enc:v1:"


def _encrypt_value(plaintext: str) -> str:
    """Encrypt a secret value for storage. Returns enc:v1:<base64>."""
    if not plaintext or not isinstance(plaintext, str):
        return plaintext
    try:
        enc = _get_encryptor()
        encrypted = enc.encrypt(plaintext.encode("utf-8"))
        if isinstance(encrypted, bytes):
            encrypted = encrypted.decode("ascii")
        return f"{_ENC_PREFIX}{encrypted}"
    except Exception as e:
        print(f"[config] WARNING: Failed to encrypt value: {e}")
        return plaintext


def _decrypt_value(encrypted: str) -> str:
    """Decrypt a value that was encrypted with _encrypt_value.

    If decryption fails (e.g. the encryption key was lost or the cipher
    format changed), the original ciphertext is returned unchanged.
    A single warning is printed at startup (not N warnings for N keys).
    """
    if not encrypted or not isinstance(encrypted, str) or not encrypted.startswith(_ENC_PREFIX):
        return encrypted
    try:
        enc = _get_encryptor()
        token = encrypted[len(_ENC_PREFIX):].encode("ascii")
        plaintext = enc.decrypt(token)
        if isinstance(plaintext, bytes):
            plaintext = plaintext.decode("utf-8")
        return plaintext
    except Exception:
        # Print once, not once-per-key (the user already sees it for key #1)
        if not _decrypt_warned:
            _decrypt_warned = True
            if not _FERNET_AVAILABLE:
                print("[config] WARNING: Cannot decrypt secrets — 'cryptography' package is missing.")
                print("         Install it: python -m pip install cryptography")
            else:
                print("[config] WARNING: Secrets were encrypted with a different key than the current one.")
                print("         The encryption key file may have been lost or regenerated.")
                print("         Go to Settings -> API Credentials and re-enter your keys to fix this.")
        return encrypted


# All keys that should be encrypted when stored in settings.json
_ENCRYPTED_AT_REST_KEYS = frozenset({
    "openrouter_api_key", "openai_api_key", "nvidia_api_key", "custom_api_key",
    "together_api_key", "groq_api_key", "auth_hmac_key", "auth_salt", "default_admin_password",
    "github_token", "smtp_password",
})


def load_settings() -> dict[str, Any]:
    """Load settings with precedence: env vars > settings.json > defaults.

    Environment variables take the highest priority. Settings loaded from env
    vars are tracked and never written back to settings.json.

    Encrypted values (prefixed with "enc:v1:") are decrypted transparently.
    """
    global _env_loaded_keys
    _env_loaded_keys = set()
    settings = dict(DEFAULT_SETTINGS)

    # Layer 1: Load from settings.json (decrypt any encrypted-at-rest values)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Decrypt any encrypted-at-rest values before updating
            for k, v in saved.items():
                if isinstance(v, str) and v.startswith(_ENC_PREFIX):
                    saved[k] = _decrypt_value(v)
            settings.update(saved)
        except Exception as e:
            print(f"[config] Failed to load settings.json: {e}")

    # Layer 2: Override with environment variables (highest precedence)
    # FIX: Skip placeholder values from .env so they don't override real keys
    # that were saved in settings.json via the UI.
    for settings_key, env_var in _SECRET_KEYS.items():
        env_val = os.environ.get(env_var)
        if env_val and not _is_placeholder_value(env_val):
            settings[settings_key] = env_val
            _env_loaded_keys.add(settings_key)
        elif env_val and _is_placeholder_value(env_val):
            print(f"[config] Ignoring placeholder env var {env_var}="
                  f"{env_val[:10]}... — keeping value from settings.json")

    for settings_key, env_var in _ENV_OVERRIDE_KEYS.items():
        env_val = os.environ.get(env_var)
        if env_val:
            if settings_key == "cors_origins":
                settings[settings_key] = [o.strip() for o in env_val.split(",") if o.strip()]
            else:
                settings[settings_key] = env_val
            _env_loaded_keys.add(settings_key)

    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Persist settings to disk.

    Secret keys that were loaded from environment variables are excluded
    from the saved JSON to prevent leakage.

    Sensitive values (API keys, tokens, passwords) are encrypted at rest
    using Fernet (or HMAC fallback) before writing to settings.json.
    """
    safe = {}
    for k, v in settings.items():
        if k in _env_loaded_keys:
            continue  # Never persist env-overridden keys
        if k in _ENCRYPTED_AT_REST_KEYS and isinstance(v, str) and v:
            # Encrypt the value before storing
            safe[k] = _encrypt_value(v)
        else:
            safe[k] = v
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False)


# Global settings singleton (mutable, reloaded via save_settings)
SETTINGS = load_settings()


def get(key: str, default: Any = None) -> Any:
    """Get a setting value. Re-reads env var for secrets each time.

    FIX: Also checks that env var value is not a placeholder before returning it.
    This prevents placeholder .env values from being used as real API keys.

    FIX: Type coercion — if *default* is an int/float/bool, the returned
    value is coerced to match.  Settings loaded from JSON may come back as
    strings (e.g. "587" for smtp_port), so we cast them to the expected type.
    """
    if key in _SECRET_KEYS:
        env_val = os.environ.get(_SECRET_KEYS[key])
        if env_val and not _is_placeholder_value(env_val):
            return env_val
    val = SETTINGS.get(key, default)
    if val is not None and default is not None:
        try:
            if isinstance(default, bool):
                val = val if isinstance(val, bool) else str(val).lower() in ('true', '1', 'yes')
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
        except (ValueError, TypeError):
            val = default
    return val


def set_many(updates: dict[str, Any]) -> dict[str, Any]:
    """Update settings. Env-overridden keys are skipped unless the env value
    is a placeholder (in which case the user's real key should take precedence)."""
    safe_updates = {}
    for k, v in updates.items():
        if k in _env_loaded_keys:
            # FIX: If the env value is a placeholder, allow the update through.
            # This lets users save real API keys even when .env has placeholder stubs.
            env_var = _SECRET_KEYS.get(k) or _ENV_OVERRIDE_KEYS.get(k)
            if env_var:
                current_env_val = os.environ.get(env_var, "")
                if _is_placeholder_value(current_env_val):
                    # Remove from env-loaded tracking so it can be saved to settings.json
                    _env_loaded_keys.discard(k)
                    print(f"[config] Allowing update for '{k}' — env var was a placeholder")
                    safe_updates[k] = v
                    continue
            # Real env var (not a placeholder) — respect its precedence
            print(f"[config] Skipping update for '{k}' — loaded from environment variable")
            continue
        safe_updates[k] = v

    # FIX: Merge updates FIRST, THEN save to disk.
    # Previously save_settings(SETTINGS) ran BEFORE SETTINGS.update(safe_updates),
    # so the disk file always got the OLD values and new values only existed in
    # memory — silently lost on restart.
    SETTINGS.update(safe_updates)
    save_settings(SETTINGS)
    return SETTINGS


def is_env_overridden(key: str) -> bool:
    """Check if a setting is being overridden by an environment variable."""
    return key in _env_loaded_keys
