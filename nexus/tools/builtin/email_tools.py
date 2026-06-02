"""
Email tools — SMTP send + IMAP read/search/reply.
Credentials configured in Settings (smtp_host, smtp_port, smtp_user, smtp_password,
imap_host, imap_port, email_address).
"""
from __future__ import annotations

import asyncio
import base64
import email as _email_lib
import imaplib
import json
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from ...config import get as cfg
from ..registry import tool


async def _get_email_oauth_access_token(email_cfg: dict) -> str | None:
    """Get a valid OAuth2 access token, refreshing if expired."""
    token = email_cfg.get("oauth_token")
    if not token or not isinstance(token, dict):
        return None
    from .email_oauth import is_token_expired, refresh_token
    if is_token_expired(token):
        provider = email_cfg.get("provider", "gmail")
        refresh_tok = token.get("refresh_token", "")
        if not refresh_tok:
            return None
        client_id = cfg("google_oauth_client_id" if provider == "gmail" else "microsoft_oauth_client_id", "")
        client_secret = cfg("google_oauth_client_secret" if provider == "gmail" else "microsoft_oauth_client_secret", "")
        if not client_id or not client_secret:
            return None
        try:
            new_token = await refresh_token(provider, refresh_tok, client_id, client_secret)
            # Persist refreshed token
            try:
                from ...memory import db as _db
                existing = await _db.get_integration_config("email") or {}
                if isinstance(existing, dict):
                    existing["oauth_token"] = new_token
                    await _db.update_integration_config("email", existing)
            except Exception:
                pass
            return new_token.get("access_token", "")
        except Exception:
            return None
    return token.get("access_token", "")


async def _get_email_config() -> dict:
    """Unified: check integration table first, then legacy settings."""
    result = {
        "smtp_host": cfg("smtp_host", ""),
        "smtp_port": int(cfg("smtp_port", 587)),
        "smtp_user": cfg("smtp_user", ""),
        "smtp_password": cfg("smtp_password", ""),
        "email_address": cfg("email_address", cfg("smtp_user", "")),
        "imap_host": cfg("imap_host", ""),
        "imap_port": int(cfg("imap_port", 993)),
        "provider": "",
        "oauth_token": None,
    }
    try:
        from ...memory import db as _db
        intg = await _db.get_integration("email")
        if intg and intg.get("connected"):
            config = await _db.get_integration_config("email")
            if config.get("smtp_host"):
                result["smtp_host"] = config["smtp_host"]
            if config.get("smtp_port"):
                result["smtp_port"] = int(config["smtp_port"])
            if config.get("smtp_user"):
                result["smtp_user"] = config["smtp_user"]
            if config.get("smtp_password"):
                result["smtp_password"] = config["smtp_password"]
            if config.get("imap_host"):
                result["imap_host"] = config["imap_host"]
            if config.get("imap_port"):
                result["imap_port"] = int(config["imap_port"])
            if config.get("email_address"):
                result["email_address"] = config["email_address"]
            # OAuth provider & token
            if config.get("provider"):
                result["provider"] = config["provider"]
            if config.get("oauth_token"):
                result["oauth_token"] = config["oauth_token"]
            # Update from address (fallback to smtp_user)
            if result["email_address"]:
                pass
            elif result["smtp_user"]:
                result["email_address"] = result["smtp_user"]
    except Exception:
        pass
    return result


async def _smtp_cfg() -> dict:
    c = await _get_email_config()
    return {
        "host":     c["smtp_host"],
        "port":     c["smtp_port"],
        "user":     c["smtp_user"],
        "password": c["smtp_password"],
        "from":     c["email_address"],
    }


async def _imap_cfg() -> dict:
    c = await _get_email_config()
    return {
        "host":     c["imap_host"],
        "port":     c["imap_port"],
        "user":     c["smtp_user"],
        "password": c["smtp_password"],
    }


async def _missing(*keys, oauth_exempt: bool = True) -> str | None:
    c = await _get_email_config()
    has_oauth = bool(c.get("oauth_token") and isinstance(c.get("oauth_token"), dict) and c["oauth_token"].get("access_token"))
    exempt = has_oauth and oauth_exempt
    missing = []
    for k in keys:
        if k in ("smtp_host",) and not c.get("smtp_host"):
            missing.append(k)
        elif k == "smtp_user" and not c.get(k) and not exempt:
            missing.append(k)
        elif k == "smtp_password" and not c.get(k) and not exempt:
            missing.append(k)
        elif k == "imap_host" and not c.get("imap_host") and not exempt:
            missing.append(k)
    return f"Missing config: {', '.join(missing)}. Connect Email in Integrations or set in Settings." if missing else None


@tool(
    name="email_send",
    description="Send an email via SMTP. Requires smtp_host, smtp_user, smtp_password in Settings.",
    parameters_schema={
        "type": "object",
        "properties": {
            "to":      {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string"},
            "body":    {"type": "string"},
            "cc":      {"type": "string", "default": ""},
        },
        "required": ["to", "subject", "body"],
    },
    risk="high",
    category="email",
)
async def email_send(params):
    err = await _missing("smtp_host", "smtp_user", "smtp_password")
    if err:
        return err

    c = await _smtp_cfg()
    msg = MIMEMultipart()
    msg["From"]    = c["from"]
    msg["To"]      = params["to"]
    msg["Subject"] = params["subject"]
    if cc := params.get("cc"):
        msg["Cc"] = cc
    msg.attach(MIMEText(params["body"], "plain"))

    # Resolve access token BEFORE entering blocking executor
    oauth_token = c.get("oauth_token")
    access_token = None
    if oauth_token:
        email_cfg = await _get_email_config()
        access_token = await _get_email_oauth_access_token(email_cfg)
        if not access_token:
            return "OAuth token expired and could not be refreshed. Reconnect the email integration."

    def _send():
        ctx = ssl.create_default_context()
        with smtplib.SMTP(c["host"], c["port"]) as server:
            server.ehlo()
            server.starttls(context=ctx)
            if access_token:
                from .email_oauth import build_xoauth2_string
                auth_str = build_xoauth2_string(c["user"] or c["from"], access_token)
                server.auth("XOAUTH2", lambda _=None: auth_str.encode("utf-8"))
            else:
                server.login(c["user"], c["password"])
            recipients = [params["to"]] + ([params.get("cc")] if params.get("cc") else [])
            server.sendmail(c["from"], recipients, msg.as_string())

    try:
        await asyncio.get_running_loop().run_in_executor(None, _send)
        return f"Email sent to {params['to']} — subject: {params['subject']}"
    except Exception as e:
        return f"SMTP error: {e}"


@tool(
    name="email_inbox",
    description="Read the latest emails from your inbox. Requires imap_host, smtp_user, smtp_password in Settings.",
    parameters_schema={
        "type": "object",
        "properties": {
            "limit":  {"type": "integer", "default": 10},
            "folder": {"type": "string", "default": "INBOX"},
        },
        "required": [],
    },
    category="email",
)
async def email_inbox(params):
    err = await _missing("imap_host", "smtp_user", "smtp_password", oauth_exempt=False)
    if err:
        return err

    c    = await _imap_cfg()
    n    = min(50, int(params.get("limit", 10)))
    folder = params.get("folder", "INBOX")

    def _fetch():
        with imaplib.IMAP4_SSL(c["host"], c["port"]) as mail:
            mail.login(c["user"], c["password"])
            mail.select(folder)
            _, data = mail.search(None, "ALL")
            ids = data[0].split()
            ids = ids[-n:]  # latest N
            messages = []
            for uid in reversed(ids):
                _, msg_data = mail.fetch(uid, "(RFC822)")
                raw = msg_data[0][1]
                msg = _email_lib.message_from_bytes(raw)
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(errors="replace")[:500]
                            break
                else:
                    body = msg.get_payload(decode=True).decode(errors="replace")[:500]
                messages.append({
                    "id":      uid.decode(),
                    "from":    msg.get("From", ""),
                    "subject": msg.get("Subject", ""),
                    "date":    msg.get("Date", ""),
                    "preview": body.strip()[:200],
                })
            return messages

    try:
        msgs = await asyncio.get_running_loop().run_in_executor(None, _fetch)
        if not msgs:
            return "Inbox is empty."
        lines = [f"## {folder} ({len(msgs)} messages)"]
        for m in msgs:
            lines.append(f"\n[{m['id']}] {m['date']}\nFrom: {m['from']}\nSubject: {m['subject']}\n{m['preview']}")
        return "\n".join(lines)
    except Exception as e:
        return f"IMAP error: {e}"


@tool(
    name="email_search",
    description="Search emails by subject or sender keyword.",
    parameters_schema={
        "type": "object",
        "properties": {
            "query":  {"type": "string", "description": "Search keyword"},
            "field":  {"type": "string", "enum": ["subject", "from", "body"], "default": "subject"},
            "limit":  {"type": "integer", "default": 10},
        },
        "required": ["query"],
    },
    category="email",
)
async def email_search(params):
    err = await _missing("imap_host", "smtp_user", "smtp_password", oauth_exempt=False)
    if err:
        return err

    c     = await _imap_cfg()
    query = params["query"]
    field = params.get("field", "subject").upper()
    n     = min(50, int(params.get("limit", 10)))
    imap_field = {"SUBJECT": "SUBJECT", "FROM": "FROM", "BODY": "BODY"}.get(field, "SUBJECT")

    def _search():
        with imaplib.IMAP4_SSL(c["host"], c["port"]) as mail:
            mail.login(c["user"], c["password"])
            mail.select("INBOX")
            _, data = mail.search(None, f'{imap_field} "{query}"')
            ids = data[0].split()[-n:]
            results = []
            for uid in reversed(ids):
                _, msg_data = mail.fetch(uid, "(RFC822)")
                msg = _email_lib.message_from_bytes(msg_data[0][1])
                results.append(
                    f"[{uid.decode()}] From: {msg.get('From','')} | "
                    f"Subject: {msg.get('Subject','')} | {msg.get('Date','')}"
                )
            return results

    try:
        results = await asyncio.get_running_loop().run_in_executor(None, _search)
        return "\n".join(results) if results else f"No emails matching '{query}'"
    except Exception as e:
        return f"Search error: {e}"


@tool(
    name="email_reply",
    description="Reply to an email by message ID.",
    parameters_schema={
        "type": "object",
        "properties": {
            "message_id": {"type": "string", "description": "Message ID from email_inbox"},
            "body":       {"type": "string"},
        },
        "required": ["message_id", "body"],
    },
    risk="high",
    category="email",
)
async def email_reply(params):
    err = await _missing("imap_host", "smtp_host", "smtp_user", "smtp_password", oauth_exempt=False)
    if err:
        return err

    c    = await _imap_cfg()
    smtp = await _smtp_cfg()

    def _fetch_and_reply():
        with imaplib.IMAP4_SSL(c["host"], c["port"]) as mail:
            mail.login(c["user"], c["password"])
            mail.select("INBOX")
            uid = params["message_id"].encode()
            _, msg_data = mail.fetch(uid, "(RFC822)")
            original = _email_lib.message_from_bytes(msg_data[0][1])

        reply = MIMEMultipart()
        reply["From"]    = smtp["from"]
        reply["To"]      = original.get("From", "")
        reply["Subject"] = "Re: " + original.get("Subject", "")
        reply["In-Reply-To"] = original.get("Message-ID", "")
        reply.attach(MIMEText(params["body"], "plain"))

        ctx = ssl.create_default_context()
        with smtplib.SMTP(smtp["host"], smtp["port"]) as server:
            server.starttls(context=ctx)
            server.login(smtp["user"], smtp["password"])
            server.sendmail(smtp["from"], [reply["To"]], reply.as_string())
        return reply["To"]

    try:
        to = await asyncio.get_running_loop().run_in_executor(None, _fetch_and_reply)
        return f"Reply sent to {to}"
    except Exception as e:
        return f"Reply error: {e}"
