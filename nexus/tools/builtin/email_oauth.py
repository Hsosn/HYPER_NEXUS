"""
Email OAuth handler — Google Gmail & Microsoft Outlook XOAUTH2 flows.

Provides:
- Google & Microsoft OAuth2 authorization URL generation
- Authorization code → token exchange
- Access token refresh (with automatic expiry tracking)
- XOAUTH2 SMTP auth string generation

Config stored in the email integration's config dict:
  provider: "gmail" | "outlook"
  oauth_token: { access_token, refresh_token, token_type, expires_at }
  email: user's email address
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_SCOPES = "https://mail.google.com/ openid profile email"  # Full Gmail access + email/profile

MICROSOFT_AUTH_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
# Microsoft Graph scope for SMTP send — "https://outlook.office.com/SMTP.Send"
# For broader access: "https://outlook.office.com/IMAP.AccessAsUser.All https://outlook.office.com/SMTP.Send"
MICROSOFT_SCOPES = (
    "https://outlook.office.com/SMTP.Send "
    "https://outlook.office.com/IMAP.AccessAsUser.All "
    "offline_access openid profile email"
)

# ──────────────────────────────────────────────
# Google OAuth Flow
# ──────────────────────────────────────────────


async def _http_post_json(url: str, data: dict[str, str]) -> dict[str, Any]:
    """Non-blocking POST with form-encoded body, return parsed JSON."""
    import httpx

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, data=data)
        resp.raise_for_status()
        return resp.json()


def build_google_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    """Build the Google OAuth2 authorization URL."""
    import urllib.parse

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",  # Gets refresh_token
        "prompt": "consent",       # Forces consent screen to get refresh_token
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


async def exchange_google_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange an authorization code for Google tokens."""
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    token_data = await _http_post_json(GOOGLE_TOKEN_URL, data)
    return _normalize_token_response(token_data)


async def refresh_google_token(refresh_token: str, client_id: str, client_secret: str) -> dict[str, Any]:
    """Refresh an expired Google access token."""
    data = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
    }
    token_data = await _http_post_json(GOOGLE_TOKEN_URL, data)
    return _normalize_token_response(token_data, refresh_token)


# ──────────────────────────────────────────────
# Microsoft Outlook OAuth Flow
# ──────────────────────────────────────────────


def build_microsoft_auth_url(client_id: str, redirect_uri: str, state: str) -> str:
    """Build the Microsoft OAuth2 authorization URL."""
    import urllib.parse

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": MICROSOFT_SCOPES,
        "response_mode": "query",
        "state": state,
    }
    return f"{MICROSOFT_AUTH_URL}?{urllib.parse.urlencode(params)}"


async def exchange_microsoft_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange an authorization code for Microsoft tokens."""
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    token_data = await _http_post_json(MICROSOFT_TOKEN_URL, data)
    return _normalize_token_response(token_data)


async def refresh_microsoft_token(refresh_token: str, client_id: str, client_secret: str) -> dict[str, Any]:
    """Refresh an expired Microsoft access token."""
    data = {
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
    }
    token_data = await _http_post_json(MICROSOFT_TOKEN_URL, data)
    return _normalize_token_response(token_data, refresh_token)


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────


def _normalize_token_response(raw: dict[str, Any], fallback_refresh: str = "") -> dict[str, Any]:
    """Normalize OAuth token response into a consistent shape."""
    now = int(time.time())
    token = {
        "access_token": raw.get("access_token", ""),
        "refresh_token": raw.get("refresh_token", fallback_refresh),
        "token_type": raw.get("token_type", "Bearer"),
        "expires_at": now + raw.get("expires_in", 3600),
    }
    # Extract email from id_token (JWT) if present
    id_token = raw.get("id_token", "")
    if id_token:
        try:
            # Decode JWT payload (middle segment) without verification
            # The token came directly from the OAuth provider over HTTPS, so trust is implicit
            parts = id_token.split(".")
            if len(parts) == 3:
                payload = parts[1]
                # Add padding for base64 decoding
                padding = 4 - len(payload) % 4
                if padding != 4:
                    payload += "=" * padding
                import json as _json
                claims = _json.loads(base64.urlsafe_b64decode(payload))
                email = claims.get("email", "")
                if email:
                    token["email"] = email
        except Exception:
            pass
    return token


def is_token_expired(token: dict[str, Any] | None) -> bool:
    """Check if the access token is expired or missing."""
    if not token or not token.get("access_token"):
        return True
    expires_at = token.get("expires_at", 0)
    # Add 30 second buffer to avoid edge-case expiry
    return int(time.time()) >= (expires_at - 30)


def build_xoauth2_string(email: str, access_token: str) -> str:
    """Build the XOAUTH2 base64-encoded SMTP auth string.

    Format: base64("user={email}\x01auth=Bearer {access_token}\x01\x01")
    """
    auth_bytes = f"user={email}\x01auth=Bearer {access_token}\x01\x01".encode("utf-8")
    return base64.b64encode(auth_bytes).decode("utf-8")


# ──────────────────────────────────────────────
# Provider-aware helpers
# ──────────────────────────────────────────────


def build_auth_url(provider: str, client_id: str, redirect_uri: str, state: str) -> str:
    """Build the authorization URL for the given provider."""
    if provider == "gmail":
        return build_google_auth_url(client_id, redirect_uri, state)
    elif provider == "outlook":
        return build_microsoft_auth_url(client_id, redirect_uri, state)
    else:
        raise ValueError(f"Unknown email provider: {provider}")


async def exchange_code(provider: str, code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict[str, Any]:
    """Exchange auth code for tokens."""
    if provider == "gmail":
        return await exchange_google_code(code, client_id, client_secret, redirect_uri)
    elif provider == "outlook":
        return await exchange_microsoft_code(code, client_id, client_secret, redirect_uri)
    else:
        raise ValueError(f"Unknown email provider: {provider}")


async def refresh_token(provider: str, refresh_token: str, client_id: str, client_secret: str) -> dict[str, Any]:
    """Refresh an expired token."""
    if provider == "gmail":
        return await refresh_google_token(refresh_token, client_id, client_secret)
    elif provider == "outlook":
        return await refresh_microsoft_token(refresh_token, client_id, client_secret)
    else:
        raise ValueError(f"Unknown email provider: {provider}")
