"""File system tools - scoped to a sandbox directory for safety.

Deep-optimized v4:
  • Atomic writes via temp-file + rename  — no more partial/corrupt files
  • Per-path asyncio write lock            — safe concurrent writes
  • Windows path hardening                 — long-path prefix, case-insensitive sandbox
  • Base64 auto-decode for binary payloads
  • Auto-fallback: aiofiles → sync I/O    — works even when aiofiles is missing
  • Size-optimized writes                 — direct write for small files, chunked for large files
  • Windows file-lock retry               — retries with backoff if file is temporarily locked
  • Strict workspace confinement          — all reads/writes confined to data/workspace/
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

# aiofiles is optional — fall back to synchronous I/O if missing
try:
    import aiofiles
    _HAS_AIOFILES = True
except ImportError:
    _HAS_AIOFILES = False

from ...config import BASE_DIR
from ..registry import tool

# ---------------------------------------------------------------------------
# Sandbox  (resolve once, reuse)
# ---------------------------------------------------------------------------
SANDBOX = BASE_DIR / "data" / "workspace"
SANDBOX = SANDBOX.resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)

_WIN = os.name == "nt"

# Pre-computed lowercased sandbox for case-insensitive comparison on Windows
_SANDBOX_LOWER = Path(str(SANDBOX).lower()) if _WIN else SANDBOX


def _resolve_sandboxed(rel: str) -> Path:
    """Resolve *rel* into an absolute path inside the workspace sandbox.

    Absolute paths are rejected up-front unless they already live inside the
    workspace — this prevents the agent from writing files anywhere on disk by
    passing an absolute path.  Relative paths are always joined onto SANDBOX.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("path must be a non-empty string")
    rel = rel.strip()

    # Strip an accidental leading "data/workspace/" or "workspace/" prefix the
    # model sometimes adds, so the user's intent of staying in-workspace is
    # honoured rather than rejected.
    norm = rel.replace("\\", "/").lstrip("./")
    for prefix in ("data/workspace/", "workspace/"):
        if norm.lower().startswith(prefix):
            rel = norm[len(prefix):]
            break

    p = Path(rel)
    if p.is_absolute():
        pp = p.resolve()
        check = Path(str(pp).lower()) if _WIN else pp
        if not _is_within(_SANDBOX_LOWER, check):
            raise ValueError(
                f"Absolute path '{rel}' is outside the workspace ({SANDBOX}). "
                "Use a path relative to data/workspace/."
            )
        return pp

    if rel.startswith(("/", "\\")):
        raise ValueError(f"Absolute path without drive letter not allowed: {rel}")
    return (SANDBOX / rel).resolve()


def _is_within(parent_lower: Path, check: Path) -> bool:
    """Check if *check* is within *parent_lower* (both lowercased on Windows)."""
    return parent_lower in check.parents or check == parent_lower


def _sandbox_check(resolved: Path) -> None:
    """Raise ValueError if *resolved* is outside the workspace sandbox."""
    check = Path(str(resolved).lower()) if _WIN else resolved
    if _is_within(_SANDBOX_LOWER, check):
        return
    raise ValueError(
        f"Path escape attempt: {resolved} — all file operations must stay inside {SANDBOX}"
    )



def _safe_path(rel: str) -> Path:
    pp = _resolve_sandboxed(rel)
    _sandbox_check(pp)
    return pp


def _project_path(project: str) -> Path:
    p = Path(project)
    if p.is_absolute():
        pp = p.resolve()
    else:
        pp = (SANDBOX / project).resolve()
    _sandbox_check(pp)
    return pp


# ---------------------------------------------------------------------------
# Write lock  — one lock per normalised path so concurrent writes don't corrupt
# ---------------------------------------------------------------------------
_write_locks: dict[str, asyncio.Lock] = {}
_write_locks_lock = asyncio.Lock()


async def _lock_for(path: Path) -> asyncio.Lock:
    n = str(path)
    async with _write_locks_lock:
        lock = _write_locks.get(n)
        if lock is None:
            lock = asyncio.Lock()
            _write_locks[n] = lock
        return lock


def _is_binary_content(content: Any, sample_size: int = 1024) -> bool:
    """Detect if content appears to be binary (not text)."""
    if isinstance(content, bytes):
        return True
    if isinstance(content, str):
        if len(content) == 0:
            return False
        sample = content[:sample_size]
        if "\x00" in sample:
            return True
        non_printable = sum(1 for c in sample if ord(c) < 32 and c not in "\n\r\t")
        return non_printable > len(sample) * 0.3
    return False


def _looks_like_base64(s: str) -> bool:
    """Heuristic: base64 strings are 4-char aligned, padded, with mixed-case + digits.
    Only checks the first 200 chars to avoid copying large strings in memory."""
    if len(s) < 40:
        return False
    # Only check the first portion — avoids .strip() on huge strings
    head = s[:200].strip()
    if not bool(re.fullmatch(r"[A-Za-z0-9+/=]+", head)):
        return False
    if len(head) % 4 != 0:
        return False
    if "=" in head[-4:]:
        return True
    has_upper = any(c.isupper() for c in head)
    has_lower = any(c.islower() for c in head)
    has_digit = any(c.isdigit() for c in head)
    return has_upper and has_lower and has_digit


@tool(
    name="file_read",
    description="Read any file from the Hyper Nexus workspace (data/workspace/). "
                "Automatically detects text vs binary files. For large files, uses "
                "streaming reads and returns the beginning portion.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within workspace"},
            "max_chars": {"type": "integer", "default": 20000,
                          "description": "Max characters to return (0 = no limit)"},
            "encoding": {"type": "string", "default": "utf-8",
                          "description": "File encoding (e.g. utf-8, latin-1, utf-16)"},
            "offset": {"type": "integer", "default": 0,
                        "description": "Starting byte offset to read from"},
            "binary": {"type": "boolean", "default": False,
                        "description": "Force binary mode (returns base64)"},
        },
        "required": ["path"],
    },
    category="files",
)
async def file_read(params):
    try:
        # Case-insensitive param lookup
        param_lower = {k.lower(): v for k, v in params.items() if isinstance(k, str)}
        raw_path = param_lower.get("path")
        if not raw_path:
            return "Error: 'path' parameter is required"
        p = _safe_path(raw_path)
        if not p.exists():
            return f"File not found: {raw_path}"
        stat = p.stat()
        file_size = stat.st_size
        max_chars = int(param_lower.get("max_chars", 20000))
        encoding = param_lower.get("encoding", "utf-8")
        offset = int(param_lower.get("offset", 0))
        force_binary = param_lower.get("binary", False)

        # Reject directories
        if stat.st_mode & 0o170000 == 0o040000:
            return f"Error: '{raw_path}' is a directory, not a file"

        # Clamp offset
        if offset >= file_size:
            return f"Error: offset {offset} is beyond file size ({file_size} bytes)"

        # Detect binary: check first 8KB for null bytes or high ratio of non-printable chars
        is_binary = force_binary
        if not is_binary and file_size > 0:
            try:
                with open(p, "rb") as _detect:
                    sample = _detect.read(8192)
                if b"\x00" in sample:
                    is_binary = True
                else:
                    text_char = bytes(range(9, 14)) + bytes(range(32, 128))
                    non_text = sum(1 for b in sample if b not in text_char)
                    if len(sample) > 0 and non_text / len(sample) > 0.3:
                        is_binary = True
            except Exception:
                is_binary = True

        # ── Binary file handling ──────────────────────────────────────────
        if is_binary:
            read_size = min(file_size - offset, 10 * 1024 * 1024)
            try:
                with open(p, "rb") as f:
                    if offset > 0:
                        f.seek(offset)
                    chunk = f.read(read_size)
                b64 = base64.b64encode(chunk).decode("ascii")
                summary = (
                    f"[Binary file: {raw_path}]\n"
                    f"Size: {_format_size(file_size)}\n"
                    f"MIME: {_guess_mime(p.name)}\n"
                )
                if file_size > 10 * 1024 * 1024:
                    summary += f"Showing first 10MB (base64: {len(b64)} chars). Use offset to read further.\n"
                summary += f"Base64 ({len(chunk)} bytes):\n{b64}"
                return summary
            except Exception as e:
                return f"Error reading binary file: {e}"

        # ── Text file handling ────────────────────────────────────────────
        if file_size > 50 * 1024 * 1024:
            return (
                f"File too large ({_format_size(file_size)}). "
                f"Please use a more targeted approach or process the file with Python tools."
            )

        # Read entire file for small files, chunked for large
        if file_size < 1024 * 1024 or max_chars == 0:
            if _HAS_AIOFILES:
                async with aiofiles.open(p, "r", encoding=encoding, errors="replace") as f:
                    if offset > 0:
                        await f.seek(offset)
                    content = await f.read()
            else:
                with open(p, "r", encoding=encoding, errors="replace") as f:
                    if offset > 0:
                        f.seek(offset)
                    content = f.read()
        else:
            if _HAS_AIOFILES:
                async with aiofiles.open(p, "r", encoding=encoding, errors="replace") as f:
                    if offset > 0:
                        await f.seek(offset)
                    content = await f.read(max_chars + 500)
            else:
                with open(p, "r", encoding=encoding, errors="replace") as f:
                    if offset > 0:
                        f.seek(offset)
                    content = f.read(max_chars + 500)

        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated at {max_chars} chars, file is {_format_size(file_size)}]"

        # Prepend file info header for context
        header = f"[File: {raw_path}  |  {_format_size(file_size)}]\n\n"
        return header + content

    except KeyError as e:
        return f"Error: Missing parameter '{e.args[0]}'"
    except ValueError as e:
        return f"Error: {e}"
    except UnicodeDecodeError:
        return (f"Error: Cannot decode '{raw_path}' with encoding '{encoding}'. "
                f"The file may be binary or use a different encoding. Try binary=true or a different encoding.")
    except Exception as e:
        return f"Error reading '{params['path']}': {e}"


def _guess_mime(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".svg": "image/svg+xml", ".ico": "image/x-icon",
        ".pdf": "application/pdf", ".json": "application/json",
        ".xml": "application/xml", ".yaml": "text/yaml", ".yml": "text/yaml",
        ".csv": "text/csv", ".md": "text/markdown", ".html": "text/html",
        ".js": "application/javascript", ".ts": "application/typescript",
        ".py": "text/x-python", ".rs": "text/x-rust", ".cpp": "text/x-c++",
        ".c": "text/x-c", ".h": "text/x-c-header", ".go": "text/x-go",
        ".zip": "application/zip", ".tar": "application/x-tar",
        ".gz": "application/gzip", ".mp3": "audio/mpeg",
        ".mp4": "video/mp4", ".wav": "audio/wav",
    }
    return mime_map.get(ext, "application/octet-stream")


# ── Retry helper for Windows file locks ──────────────────────────────────────
_MAX_RETRIES = 5
_RETRY_BACKOFF = 0.1  # seconds, doubles each retry


async def _retry_replace(src: Path, dst: Path) -> None:
    """Replace dst with src, retrying on Windows sharing violations."""
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last_err = e
            if _WIN:
                wait = _RETRY_BACKOFF * (2 ** attempt)
                await asyncio.sleep(wait)
                continue
            raise
        except OSError as e:
            last_err = e
            if _WIN and e.winerror in (32, 33):  # sharing violation / file in use
                wait = _RETRY_BACKOFF * (2 ** attempt)
                await asyncio.sleep(wait)
                continue
            raise
    raise last_err or RuntimeError(f"Failed to replace {dst} after {_MAX_RETRIES} retries")


# ── Async / sync I/O helpers ──────────────────────────────────────────────────


async def _write_chunked(f, content, chunk_size=65536):
    """Write content in chunks to avoid buffer issues with large strings."""
    for i in range(0, len(content), chunk_size):
        await f.write(content[i:i + chunk_size])


def _write_chunked_sync(f, content, chunk_size=65536):
    """Synchronous chunked write — used when aiofiles is not available."""
    for i in range(0, len(content), chunk_size):
        f.write(content[i:i + chunk_size])


def _win_long_path(path: Path) -> str:
    """Add Windows long-path prefix if needed (paths > 260 chars)."""
    if _WIN:
        p = str(path)
        if len(p) > 260 and not p.startswith("\\\\?\\"):
            return "\\\\?\\" + p
    return str(path)


async def _write_file_atomic(
    path: Path,
    content: str | bytes,
    encoding: str = "utf-8",
    mode: str = "w",
) -> tuple[str, int]:
    """Atomically write content to path, returning (size_info, byte_count).

    Falls back to synchronous I/O if aiofiles is not available.
    Supports append modes ('a', 'ab') — writes directly without atomic replace.
    """
    is_append = mode in ("a", "ab")

    if is_append:
        write_mode = "ab" if mode == "ab" or isinstance(content, bytes) else "a"
        try:
            if _HAS_AIOFILES:
                async with aiofiles.open(_win_long_path(path), write_mode, encoding=encoding if write_mode == "a" else None) as f:
                    if isinstance(content, str) and write_mode == "ab":
                        byte_content = content.encode(encoding, errors="replace")
                        await f.write(byte_content)
                    else:
                        await f.write(content)
            else:
                with open(_win_long_path(path), write_mode, encoding=encoding if write_mode == "a" else None) as f:
                    if isinstance(content, str) and write_mode == "ab":
                        f.write(content.encode(encoding, errors="replace"))
                    else:
                        f.write(content)
            byte_count = len(content) if isinstance(content, bytes) else len(content.encode(encoding, errors="replace"))
            return f"appended {byte_count} bytes", byte_count
        except BaseException:
            raise

    tmp = path.with_suffix(path.suffix + ".tmp" + os.urandom(4).hex())
    try:
        is_binary = mode == "wb" or isinstance(content, bytes)

        if is_binary:
            byte_content = content.encode(encoding, errors="replace") if isinstance(content, str) else content
            if _HAS_AIOFILES:
                async with aiofiles.open(tmp, "wb") as f:
                    await f.write(byte_content)
                byte_count = (await asyncio.to_thread(lambda: tmp.stat())).st_size
            else:
                with open(tmp, "wb") as f:
                    f.write(byte_content)
                byte_count = tmp.stat().st_size
            size_info = f"{byte_count} bytes"
        else:
            if _HAS_AIOFILES:
                async with aiofiles.open(tmp, "w", encoding=encoding, errors="replace") as f:
                    await _write_chunked(f, content)
                byte_count = (await asyncio.to_thread(lambda: tmp.stat())).st_size
            else:
                with open(tmp, "w", encoding=encoding, errors="replace") as f:
                    _write_chunked_sync(f, content)
                byte_count = tmp.stat().st_size
            size_info = f"{len(content)} chars ({byte_count} bytes)"

        await _retry_replace(tmp, path)
        return size_info, byte_count

    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


_LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB — use chunked streaming above this


def _get_file_size_hint(content: str | bytes | dict | list | int | float | bool | None) -> int:
    """Estimate the byte size of content without encoding it fully."""
    if content is None:
        return 0
    if isinstance(content, bytes):
        return len(content)
    if isinstance(content, str):
        return len(content) * 4
    if isinstance(content, (dict, list)):
        return len(str(content)) * 4
    if isinstance(content, (int, float, bool)):
        return len(str(content))
    return 4096


def _format_size(bytes_: int) -> str:
    """Format byte count as human-readable string."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024 * 1024:
        return f"{bytes_ / 1024:.1f} KB"
    elif bytes_ < 1024 * 1024 * 1024:
        return f"{bytes_ / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_ / (1024 * 1024 * 1024):.1f} GB"


def _resolve_file_write_params(params: dict) -> dict:
    """Resolve flexible param names for file_write.

    NOTE: The registry's middleware already normalises standard aliases
    (filepath→path, data→content, etc.).  This is a defence-in-depth fallback.
    """
    resolved = {}

    # Build case-insensitive lookup: if original has "File", find "file"
    param_lower = {k.lower(): v for k, v in params.items() if isinstance(k, str)}

    path_keys = ("path", "filepath", "filename", "name", "file_path", "file",
                 "dest", "destination", "output", "out", "target")
    for key in path_keys:
        if key in params:
            resolved["path"] = params[key]
            break
        lk = param_lower.get(key)
        if lk is not None:
            resolved["path"] = lk
            break
    resolved.setdefault("path", param_lower.get("path", ""))

    content_keys = ("content", "data", "text", "body", "payload", "value",
                    "contents", "html", "code", "json", "markdown", "md",
                    "yaml", "yml", "source", "src")
    for key in content_keys:
        if key in params:
            resolved["content"] = params[key]
            break
        lk = param_lower.get(key)
        if lk is not None:
            resolved["content"] = lk
            break
    resolved.setdefault("content", "")

    resolved["encoding"] = param_lower.get("encoding",
                          param_lower.get("enc", "utf-8"))
    mode_raw = param_lower.get("mode",
               param_lower.get("file_mode", "w"))
    if mode_raw in ("a", "ab", "wb", "w"):
        resolved["mode"] = mode_raw
    else:
        resolved["mode"] = "w"
    return resolved


@tool(
    name="file_write",
    description="Write content to a file in the Hyper Nexus workspace. Creates parent dirs. Overwrites existing files. "
                "Supports text files, binary files, JSON serialization, and various encodings.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path within workspace"},
            "filepath": {"type": "string", "description": "Alias for path"},
            "filename": {"type": "string", "description": "Alias for path"},
            "file_path": {"type": "string", "description": "Alias for path"},
            "file": {"type": "string", "description": "Alias for path"},
            "name": {"type": "string", "description": "Alias for path"},
            "dest": {"type": "string", "description": "Alias for path"},
            "destination": {"type": "string", "description": "Alias for path"},
            "output": {"type": "string", "description": "Alias for path"},
            "out": {"type": "string", "description": "Alias for path"},
            "target": {"type": "string", "description": "Alias for path"},
            "content": {"description": "Content to write (string, dict, list, number, or base64-encoded binary)"},
            "data": {"description": "Alias for content"},
            "text": {"description": "Alias for content"},
            "body": {"description": "Alias for content"},
            "code": {"description": "Alias for content"},
            "payload": {"description": "Alias for content"},
            "html": {"description": "Alias for content"},
            "json": {"description": "Alias for content"},
            "markdown": {"description": "Alias for content"},
            "value": {"description": "Alias for content"},
            "contents": {"description": "Alias for content"},
            "source": {"description": "Alias for content"},
            "src": {"description": "Alias for content"},
            "encoding": {"type": "string", "description": "File encoding (default: utf-8)", "default": "utf-8"},
            "enc": {"type": "string", "description": "Alias for encoding"},
            "mode": {"type": "string", "description": "Write mode: 'w' (text, default) or 'wb' (binary)", "default": "w"},
            "file_mode": {"type": "string", "description": "Alias for mode"},
        },
        "required": ["path", "content"],
    },
    timeout=300,
    risk="medium",
    category="files",
)
async def file_write(params):
    try:
        resolved = _resolve_file_write_params(params)
        raw_path = resolved["path"]
        raw_content = resolved["content"]
        encoding = resolved["encoding"]
        mode = resolved["mode"]

        if not raw_path:
            return ("Error: 'path' parameter is required. "
                    "I accept these aliases: path, filepath, filename, name, file_path, file, "
                    "dest, destination, output, out, target. "
                    "Provide one of these with the file path relative to workspace.")
        if raw_content is None:
            return ("Error: 'content' parameter is required. "
                    "I accept these aliases: content, data, text, body, code, source, src, "
                    "html, json, markdown, md, yaml, yml. "
                    "Provide one of these with the content to write.")

        p = _safe_path(raw_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Serialise content to a string / bytes
        if isinstance(raw_content, (dict, list)):
            content_str = json.dumps(raw_content, indent=2, ensure_ascii=False)
            content_type = "json"
        elif isinstance(raw_content, (int, float, bool)):
            content_str = str(raw_content)
            content_type = "text"
        elif isinstance(raw_content, bytes):
            content_str = raw_content
            content_type = "binary"
        else:
            content_str = str(raw_content)
            content_type = "text"

        # Rough size estimate from the already-serialized content (no re-encoding)
        size_estimate = len(content_str) if isinstance(content_str, str) else _get_file_size_hint(raw_content)

        # Only run base64/binary detection on reasonably-sized content (<10MB)
        if size_estimate < 10 * 1024 * 1024:
            if isinstance(content_str, str) and len(content_str) >= 40 and _looks_like_base64(content_str):
                try:
                    decoded = base64.b64decode(content_str)
                    content_str = decoded
                    content_type = "binary"
                    mode = "wb"
                except Exception:
                    pass

            if _is_binary_content(content_str) and mode in ("w", "a"):
                mode = "wb" if mode == "w" else "ab"
        elif isinstance(content_str, bytes):
            mode = "wb" if mode in ("w", "a") else mode
            content_type = "binary"

        if size_estimate > 100 * 1024 * 1024:
            import logging as _logging
            _logging.getLogger("nexus.file_write").warning(
                "Writing large file (%s): %s — %s",
                _format_size(size_estimate),
                raw_path,
                "will use chunked streaming" if size_estimate > _LARGE_FILE_THRESHOLD else "",
            )

        lock = await _lock_for(p)
        async with lock:
            size_info, byte_count = await _write_file_atomic(
                path=p,
                content=content_str,
                encoding=encoding,
                mode=mode,
            )

        try:
            rel_path = p.relative_to(SANDBOX)
        except ValueError:
            rel_path = p

        type_hint = f" ({content_type})" if content_type != "text" else ""
        size_hint = f" - {_format_size(byte_count)}"
        return f"Wrote {rel_path}{type_hint}{size_hint}"

    except KeyError as e:
        return (f"Error: Missing parameter '{e.args[0]}'. "
                f"Required: 'path' (or alias) and 'content' (or alias).")
    except ValueError as e:
        return f"Error: {e}"
    except UnicodeEncodeError as e:
        return (f"Error: Cannot encode content with encoding '{encoding}' for '{raw_path}'. "
                f"The content contains characters not supported by this encoding. "
                f"Try a different encoding (e.g. utf-8) or use mode='wb' for binary. Error: {e}")
    except Exception as e:
        err = str(e).lower()
        if "file exists" in err and "directory" in err:
            return (f"Error: Cannot write '{raw_path}' - a directory with that name might exist. "
                    f"Error: {e}")
        if "permission" in err or "access" in err:
            return (f"Error: Permission denied writing to '{raw_path}'. "
                    f"The file may be open in another program. Error: {e}")
        if "encoding" in err:
            return (f"Error: Encoding issue with '{raw_path}' using '{encoding}'. "
                    f"Try a different encoding or binary mode. Error: {e}")
        return f"Error writing '{raw_path}': {e}"


@tool(
    name="file_list",
    description="List files in a workspace directory.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path":      {"type": "string", "default": ".", "description": "Directory path relative to workspace"},
            "dir":       {"type": "string", "default": ".", "description": "Alias for path"},
            "directory": {"type": "string", "default": ".", "description": "Alias for path"},
        },
        "required": [],
    },
    category="files",
)
async def file_list(params):
    try:
        rel = params.get("path", ".")
        p = _safe_path(rel)
        if not p.exists():
            return "Directory not found"
        entries = []
        for item in sorted(p.iterdir()):
            kind = "DIR" if item.is_dir() else "FILE"
            size = item.stat().st_size if item.is_file() else 0
            entries.append(f"{kind:<4} {item.name:<40} {size}B")
        return "\n".join(entries) if entries else "(empty)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="file_delete",
    description="Delete a file in the Hyper Nexus workspace.",
    parameters_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    risk="high",
    category="files",
)
async def file_delete(params):
    try:
        p = _safe_path(params["path"])
        if not p.exists():
            return "File not found"
        p.unlink()
        return f"Deleted {params['path']}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="project_read_context",
    description="Read project context from PROJECT.md file in a workspace project. "
                "This loads the project's goals, current state, and important notes.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project folder name in workspace"},
        },
        "required": ["project"],
    },
    category="files",
)
async def project_read_context(params):
    try:
        project = params.get("project", "").strip()
        if not project:
            return "Error: project name required"
        project_dir = _project_path(project)
        project_dir.mkdir(parents=True, exist_ok=True)
        project_md = project_dir / "PROJECT.md"
        if not project_md.exists():
            return f"Project '{project}' has no PROJECT.md yet. Use project_write_context to create one."
        if _HAS_AIOFILES:
            async with aiofiles.open(project_md, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read()
        else:
            with open(project_md, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        return f"# Project: {project}\n\n{content}"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="project_write_context",
    description="Write project context to PROJECT.md file in a workspace project. "
                "Stores goals, current work state, and important notes for cross-session continuity.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project folder name in workspace"},
            "content": {"type": "string", "description": "Project context to save (goals, state, notes)"},
        },
        "required": ["project", "content"],
    },
    timeout=300,
    risk="medium",
    category="files",
)
async def project_write_context(params):
    try:
        project = params.get("project", "").strip()
        content = params.get("content", "")
        if not project:
            return "Error: project name required"
        project_dir = _project_path(project)
        project_dir.mkdir(parents=True, exist_ok=True)
        project_md = project_dir / "PROJECT.md"
        lock = await _lock_for(project_md)
        async with lock:
            size_info, _ = await _write_file_atomic(
                path=project_md,
                content=content,
                encoding="utf-8",
                mode="w",
            )
        return f"Updated PROJECT.md for project '{project}' ({len(content)} chars)"
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="file_search",
    description="Search for files by glob pattern in the workspace (data/workspace/). "
                "Use '**/*.ext' for recursive search (e.g. '**/*.py' finds all Python files). "
                "Use '*.ext' for root-level only. "
                "Returns up to 200 matching file paths relative to workspace.",
    parameters_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to search for (e.g. '*.py', '**/*.json', 'data/**')",
            },
        },
        "required": ["pattern"],
    },
    category="files",
)
async def file_search(params):
    """Search for files matching a glob pattern in the workspace."""
    import glob
    try:
        pattern = params.get("pattern", "")
        if not pattern:
            return "Error: pattern required"
        # Search relative to workspace sandbox
        search_path = str(SANDBOX / pattern)
        matches = sorted(glob.glob(search_path, recursive=True))
        if not matches:
            return f"No files matching '{pattern}'"
        # Limit to first 200 results
        if len(matches) > 200:
            result = matches[:200]
            result.append(f"... and {len(matches) - 200} more")
        else:
            result = matches
        # Make paths relative to workspace
        rel_matches = []
        for m in result:
            try:
                rel = Path(m).relative_to(SANDBOX)
                rel_matches.append(str(rel))
            except ValueError:
                rel_matches.append(m)
        return "\n".join(rel_matches)
    except Exception as e:
        return f"Error: {e}"
