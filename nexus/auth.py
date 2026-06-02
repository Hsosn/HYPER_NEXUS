"""
Authentication and rate-limiting layer for the Nexus API.

Provides:
- JWT-based token authentication (HMAC-SHA256)
- API key fallback (header: X-API-Key)
- Per-IP and per-session rate limiting
- WebSocket authentication via token query param

Token lifetime: 24 hours (configurable via AUTH_TOKEN_HOURS).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .events import emit

# ── Configuration ───────────────────────────────────────────────────────────────

_TOKEN_HOURS = 24
_MAX_REQUESTS_PER_MINUTE = 300       # Generous limit for WebUI (was 60 — too aggressive)
_MAX_WS_MESSAGES_PER_MINUTE = 120
_MAX_LOGIN_ATTEMPTS_PER_MINUTE = 10
_MAX_CHAT_REQUESTS_PER_MINUTE = 30   # Separate tighter limit for chat/AI calls
_MAX_BURST_REQUESTS = 50            # Short-window burst allowance (per 10s)


# ── Password / token helpers ───────────────────────────────────────────────────

def _hash_secret(secret: str) -> str:
    """Hash a secret with the app salt for storage."""
    salt = config.get("auth_salt", "") or "nexus-default-salt-change-me"
    return hashlib.sha256(f"{salt}:{secret}".encode()).hexdigest()


def verify_password(plain: str, stored_hash: str) -> bool:
    """Constant-time comparison of password against stored hash."""
    candidate = _hash_secret(plain)
    return hmac.compare_digest(candidate, stored_hash)


def hash_password(plain: str) -> str:
    """Hash a plaintext password for storage."""
    return _hash_secret(plain)


# ── JWT-like token (stateless HMAC-SHA256) ─────────────────────────────────────

def create_token(session_id: str, extra: dict | None = None) -> str:
    """Create a signed token containing session_id and expiry."""
    now = int(time.time())
    expiry = now + (_TOKEN_HOURS * 3600)
    payload = {"sid": session_id, "exp": expiry, "iat": now}
    if extra:
        payload.update(extra)
    payload_json = json.dumps(payload, separators=(",", ":"))
    secret = _get_hmac_key()
    sig = hmac.new(secret.encode(), payload_json.encode(), hashlib.sha256).hexdigest()
    # Format: base64_payload.signature
    import base64
    b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip("=")
    return f"{b64}.{sig}"


def verify_token(token: str) -> dict | None:
    """Verify and decode a token. Returns payload dict or None if invalid/expired."""
    try:
        import base64
        parts = token.split(".")
        if len(parts) != 2:
            return None
        b64_payload, sig = parts
        # Restore padding
        padding = 4 - len(b64_payload) % 4
        if padding != 4:
            b64_payload += "=" * padding
        payload_json = base64.urlsafe_b64decode(b64_payload).decode()
        secret = _get_hmac_key()
        expected = hmac.new(secret.encode(), payload_json.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(payload_json)
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def _get_hmac_key() -> str:
    return config.get("auth_hmac_key", "") or os.environ.get(
        "NEXUS_AUTH_KEY",
        "nexus-change-this-hmac-secret-in-production-or-set-NEXUS_AUTH_KEY",
    )


# ── User store (in-memory + persisted to DB) ───────────────────────────────────

_users: dict[str, dict] = {}  # username -> {password_hash, role, created_at}
_api_keys: dict[str, dict] = {}  # api_key -> {username, created_at}


async def init_auth() -> None:
    """Initialize auth from DB or create default admin user."""
    try:
        from .memory.database import _connect

        async with _connect() as conn:
            # Ensure users table exists
            try:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        username TEXT PRIMARY KEY,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        api_key TEXT,
                        created_at DOUBLE PRECISION
                    )
                """)
            except Exception:
                pass  # Table may already exist

            c = await conn.execute(
                "SELECT username, password_hash, role, api_key, created_at FROM users"
            )
            rows = await c.fetchall()

            for row in rows:
                user = dict(row)
                _users[user["username"]] = user
                if user.get("api_key"):
                    _api_keys[user["api_key"]] = {"username": user["username"]}

            # Create default admin if no users exist
            if not _users:
                default_pw = config.get("default_admin_password", "admin")
                admin_hash = hash_password(default_pw)
                import uuid
                api_key = f"nk-{uuid.uuid4().hex[:24]}"
                now = time.time()
                await conn.execute(
                    "INSERT INTO users(username, password_hash, role, api_key, created_at) "
                    "VALUES(?,?,?,?,?)",
                    ("admin", admin_hash, "admin", api_key, now),
                )
                _users["admin"] = {
                    "username": "admin", "password_hash": admin_hash,
                    "role": "admin", "api_key": api_key,
                }
                _api_keys[api_key] = {"username": "admin"}
                print(f"[auth] Default admin created — username: admin, password: {default_pw}")
                print(f"[auth] API key: {api_key}")
    except Exception as e:
        print(f"[auth] Init failed (running without auth): {e}")


# ── User Management ──────────────────────────────────────────────────────────────

def list_users() -> list[dict]:
    """Return all registered users (without password hashes)."""
    return [
        {"username": u["username"], "role": u.get("role", "user"),
         "created_at": u.get("created_at", 0), "has_api_key": bool(u.get("api_key"))}
        for u in _users.values()
    ]


async def create_user(username: str, password: str, role: str = "user") -> dict | None:
    """Create a new user. Returns user dict or None on failure."""
    if not username or not password:
        return None
    if username in _users:
        return None
    if role not in ("admin", "user", "viewer"):
        role = "user"

    pw_hash = hash_password(password)
    import uuid
    api_key = f"nk-{uuid.uuid4().hex[:24]}"
    now = time.time()

    user = {"username": username, "password_hash": pw_hash, "role": role, "api_key": api_key, "created_at": now}
    _users[username] = user
    _api_keys[api_key] = {"username": username}

    # Persist to DB
    try:
        from .memory.database import _connect
        async with _connect() as conn:
            await conn.execute(
                "INSERT INTO users(username, password_hash, role, api_key, created_at) "
                "VALUES(?,?,?,?,?)",
                (username, pw_hash, role, api_key, now),
            )
    except Exception as e:
        print(f"[auth] Failed to persist user: {e}")

    return {"username": username, "role": role, "api_key": api_key, "created_at": now}


async def update_user(username: str, role: str | None = None, password: str | None = None) -> dict | None:
    """Update a user's role or password. Returns updated user dict or None."""
    user = _users.get(username)
    if not user:
        return None

    if role and role in ("admin", "user", "viewer"):
        user["role"] = role
    if password:
        user["password_hash"] = hash_password(password)

    # Persist
    try:
        from .memory.database import _connect
        async with _connect() as conn:
            if role:
                await conn.execute(
                    "UPDATE users SET role=? WHERE username=?", (role, username),
                )
            if password:
                await conn.execute(
                    "UPDATE users SET password_hash=? WHERE username=?",
                    (user["password_hash"], username),
                )
    except Exception as e:
        print(f"[auth] Failed to update user: {e}")

    return {"username": username, "role": user.get("role", "user")}


async def delete_user(username: str) -> bool:
    """Delete a user (except the last admin). Returns True on success."""
    if username not in _users:
        return False

    # Prevent deleting the last admin
    user = _users[username]
    if user.get("role") == "admin":
        admin_count = sum(1 for u in _users.values() if u.get("role") == "admin")
        if admin_count <= 1:
            return False  # Can't delete the last admin

    # Remove API key
    api_key = user.get("api_key")
    if api_key:
        _api_keys.pop(api_key, None)

    del _users[username]

    # Persist
    try:
        from .memory.database import _connect
        async with _connect() as conn:
            await conn.execute("DELETE FROM users WHERE username=?", (username,))
    except Exception as e:
        print(f"[auth] Failed to delete user: {e}")

    return True


async def regenerate_api_key(username: str) -> str | None:
    """Generate a new API key for a user. Returns new key or None."""
    user = _users.get(username)
    if not user:
        return None

    # Remove old key
    old_key = user.get("api_key")
    if old_key:
        _api_keys.pop(old_key, None)

    # Generate new
    import uuid
    new_key = f"nk-{uuid.uuid4().hex[:24]}"
    user["api_key"] = new_key
    _api_keys[new_key] = {"username": username}

    # Persist
    try:
        from .memory.database import _connect
        async with _connect() as conn:
            await conn.execute(
                "UPDATE users SET api_key=? WHERE username=?", (new_key, username),
            )
    except Exception:
        pass

    return new_key


def authenticate(username: str, password: str) -> str | None:
    """Authenticate user and return a token, or None if invalid."""
    user = _users.get(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return create_token(session_id=username, extra={"role": user.get("role", "user")})


def validate_api_key(api_key: str) -> dict | None:
    """Validate an API key and return user info, or None."""
    return _api_keys.get(api_key)


# ── Rate limiting ──────────────────────────────────────────────────────────────
# Uses a tiered approach:
#   1. Burst window (10s) — prevents rapid hammering
#   2. Sustained window (60s) — prevents sustained abuse
# Both must pass for a request to be allowed.

@dataclass
class _RateBucket:
    tokens: int = 0
    window_start: float = field(default_factory=time.time)


_rate_limits: dict[str, _RateBucket] = {}
_rate_limits_burst: dict[str, _RateBucket] = {}
_bucket_max_age = 300  # Evict buckets idle for >5 minutes
_last_eviction: float = 0.0


def _evict_stale_buckets() -> None:
    """Remove expired rate limit buckets to prevent memory leak on long-running servers.

    Called on every rate limit check but only actually scans periodically (every 60s)
    to keep the overhead negligible. Each scan is O(n) but n is typically small
    (number of unique IPs active in the last 5 minutes).
    """
    global _last_eviction
    now = time.time()
    # Only scan every 60 seconds to avoid hot-path overhead
    if now - _last_eviction < 60:
        return
    _last_eviction = now

    cutoff = now - _bucket_max_age
    stale_keys = [k for k, v in _rate_limits.items() if v.window_start < cutoff]
    for k in stale_keys:
        _rate_limits.pop(k, None)

    stale_burst = [k for k, v in _rate_limits_burst.items() if v.window_start < cutoff]
    for k in stale_burst:
        _rate_limits_burst.pop(k, None)

    if stale_keys or stale_burst:
        print(f"[auth] Evicted {len(stale_keys)} sustained + {len(stale_burst)} burst rate limit buckets")


def check_rate_limit(key: str, max_requests: int = _MAX_REQUESTS_PER_MINUTE,
                     window_seconds: int = 60) -> tuple[bool, int]:
    """
    Check if a request is allowed under the rate limit.
    Returns (allowed, remaining_requests).
    Uses a sliding window counter with burst protection.
    """
    # Evict stale buckets periodically
    _evict_stale_buckets()

    now = time.time()
    remaining = 0

    # --- Burst check (10s window, max _MAX_BURST_REQUESTS) ---
    burst_bucket = _rate_limits_burst.get(key)
    if burst_bucket is None or now - burst_bucket.window_start >= 10:
        _rate_limits_burst[key] = _RateBucket(tokens=1, window_start=now)
    else:
        if burst_bucket.tokens >= _MAX_BURST_REQUESTS:
            # Burst exceeded — calculate reset time
            reset_in = max(0, 10 - (now - burst_bucket.window_start))
            return False, 0
        burst_bucket.tokens += 1

    # --- Sustained check (window_seconds) ---
    bucket = _rate_limits.get(key)
    if bucket is None or now - bucket.window_start >= window_seconds:
        _rate_limits[key] = _RateBucket(tokens=1, window_start=now)
        return True, max_requests - 1

    if bucket.tokens >= max_requests:
        reset_in = max(0, window_seconds - (now - bucket.window_start))
        return False, 0

    bucket.tokens += 1
    remaining = max_requests - bucket.tokens
    return True, remaining


def reset_rate_limit(key: str) -> None:
    """Reset rate limit for a specific key (useful after testing/config changes)."""
    _rate_limits.pop(key, None)
    _rate_limits_burst.pop(key, None)


def reset_all_rate_limits() -> int:
    """Reset ALL rate limit buckets. Returns total count cleared.

    Public API for admin endpoints — avoids importing private dicts (#8 fix).
    """
    count = len(_rate_limits) + len(_rate_limits_burst)
    _rate_limits.clear()
    _rate_limits_burst.clear()
    return count


# ── Auth dependency for FastAPI ────────────────────────────────────────────────

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer(auto_error=False)


def get_rate_limit_config() -> dict:
    """Return current rate limit configuration (readable, for admin)."""
    return {
        "max_http_per_minute": _MAX_REQUESTS_PER_MINUTE,
        "max_burst_10s": _MAX_BURST_REQUESTS,
        "max_ws_per_minute": _MAX_WS_MESSAGES_PER_MINUTE,
        "max_login_per_minute": _MAX_LOGIN_ATTEMPTS_PER_MINUTE,
        "max_chat_per_minute": _MAX_CHAT_REQUESTS_PER_MINUTE,
        "active_buckets": len(_rate_limits),
    }


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency that validates authentication.
    Accepts either:
    - Bearer token in Authorization header
    - X-API-Key header
    Returns the decoded token payload.
    """
    auth_enabled = config.get("enable_auth", False)
    if not auth_enabled:
        return {"sid": "anonymous", "role": "admin", "exp": time.time() + 999999}

    # Try Bearer token first
    if credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            return payload

    # Try API key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        user = validate_api_key(api_key)
        if user:
            return {"sid": user["username"], "role": _users.get(user["username"], {}).get("role", "user"),
                    "exp": time.time() + 999999}

    raise HTTPException(status_code=401, detail="Authentication required")


async def optional_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict | None:
    """Like require_auth but returns None instead of 401 (for public-read endpoints)."""
    auth_enabled = config.get("enable_auth", False)
    if not auth_enabled:
        return None

    if credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            return payload

    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        user = validate_api_key(api_key)
        if user:
            return {"sid": user["username"], "role": "user"}

    return None
