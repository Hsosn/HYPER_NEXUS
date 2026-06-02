"""
Loads user-defined custom tools and skills from the database and file uploads.

Custom tools are Python code fragments stored in the custom_tools table.
Each must define a function `run(params: dict) -> str`. Both sync and async
functions are supported:
  - sync:   def run(params: dict) -> str: ...
  - async:  async def run(params: dict) -> str: ...

Custom skills are loaded from SKILL.MD files or zip archives uploaded via
the WebUI.  Each SKILL.MD contains metadata and optional Python code that
is parsed and registered as both a skill entry and (if code is present) a
tool in the registry.

Security hardening (v6.1):
- Restricted namespace with NO access to os, sys, subprocess, import, exec, eval, open, __import__
- Bytecode verification before execution
- Execution timeout
- Admin-only creation enforced at API level
- Output size limit

v23 fixes:
- Async run() functions now properly awaited (was silently returning coroutine objects)
- Custom skills now require confirmation, same as custom tools
- Prompt bloat limit added to get_custom_skills_for_prompt() (2KB cap)
"""
from __future__ import annotations

import asyncio
import dis
import inspect
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..registry import REGISTRY, Tool
from ...memory import db
from ... import config


# ── Skill storage directory ──────────────────────────────────────────────────
SKILLS_DIR = Path(config.BASE_DIR) / "data" / "custom_skills"


def _ensure_skills_dir() -> Path:
    """Create and return the custom skills directory."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return SKILLS_DIR


# Strictly limited safe builtins — no file I/O, no imports, no exec/eval, no subprocess
SAFE_BUILTINS: dict[str, Any] = {
    # Types
    "print": print, "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "len": len, "range": range, "enumerate": enumerate,
    "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset,
    "str": str, "int": int, "float": float, "bool": bool, "bytes": bytes, "bytearray": bytearray,
    "True": True, "False": False, "None": None,
    # Math / logic
    "sum": sum, "min": min, "max": max, "sorted": sorted, "reversed": reversed,
    "any": any, "all": all, "zip": zip, "map": map, "filter": filter,
    "abs": abs, "round": round, "divmod": divmod, "pow": pow,
    "hash": hash, "id": id, "repr": repr, "format": format,
    "ord": ord, "chr": chr, "bin": bin, "oct": oct, "hex": hex,
    "iter": iter, "next": next,
    # Exceptions
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "RuntimeError": RuntimeError, "StopIteration": StopIteration,
    "Exception": Exception,
}

# Opcodes that are dangerous and must NOT appear in custom tool bytecode
_FORBIDDEN_OPCODES = {
    # Import-related
    dis.opmap.get("IMPORT_NAME"), dis.opmap.get("IMPORT_FROM"), dis.opmap.get("IMPORT_STAR"),
    # Execution / class creation
    dis.opmap.get("LOAD_BUILD_CLASS"),
    # v22 FIX: Removed EXEC_STMT (removed in Python 3.x, exec() uses CALL opcodes now)
    # and EXECUTE (never existed in CPython). Added CALL_FUNCTION_EX which handles
    # *args/**kwargs calls that could bypass import restrictions.
    dis.opmap.get("CALL_FUNCTION_EX"),
}
# Remove any None entries from opcodes that don't exist in this Python version
_FORBIDDEN_OPCODES = {op for op in _FORBIDDEN_OPCODES if op is not None}

# Names that must not appear in the compiled code
_FORBIDDEN_NAMES = {
    "os", "sys", "subprocess", "import", "exec", "eval", "open", "__import__",
    "compile", "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "hasattr", "property", "super", "breakpoint", "input",
    "exit", "quit", "memoryview", "object",
}


def _verify_bytecode(code_obj) -> list[str]:
    """Scan bytecode for forbidden opcodes. Returns list of violations."""
    violations = []
    try:
        for instr in dis.get_instructions(code_obj):
            if instr.opcode in _FORBIDDEN_OPCODES:
                violations.append(f"Forbidden opcode: {instr.opname}")
    except Exception:
        pass
    return violations


def _verify_source(code: str) -> list[str]:
    """Scan source for forbidden name references. Returns list of violations."""
    violations = []
    for name in _FORBIDDEN_NAMES:
        if re.search(rf'\b{name}\b', code):
            violations.append(f"Forbidden reference: {name}")
    dunder_pattern = re.findall(r'__\w+__', code)
    for d in dunder_pattern:
        if d not in ("__init__", "__name__", "__doc__"):
            violations.append(f"Forbidden dunder: {d}")
    return violations


_MAX_OUTPUT_CHARS = 10000
_EXEC_TIMEOUT = 10  # seconds


# ── Custom Tool Loader (existing) ────────────────────────────────────────────

def load_custom_tool(name: str, description: str, schema: dict[str, Any], code: str) -> str:
    """Compile and register a custom tool with security verification.
    Returns 'ok' or error message.

    v23: Fixed async function handling — if run() is async def, it is properly
    awaited instead of being passed through run_in_executor (which silently
    returns a coroutine object for async functions).
    """
    try:
        source_violations = _verify_source(code)
        if source_violations:
            return f"Security violation: {'; '.join(source_violations[:3])}"

        ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
        compiled = compile(code, f"<custom:{name}>", "exec")  # noqa: S102

        byte_violations = _verify_bytecode(compiled)
        if byte_violations:
            return f"Security violation: {'; '.join(byte_violations[:3])}"

        exec(compiled, ns)  # noqa: S102
        fn = ns.get("run")
        if not fn or not callable(fn):
            return "Custom tool code must define a function named 'run(params)'."

        # v23: Detect whether run() is async or sync, and handle both correctly.
        # Previously, async run() was silently broken — run_in_executor can't
        # await coroutines, so it just returned the coroutine object as a string.
        _is_async = asyncio.iscoroutinefunction(fn)

        async def handler(params):
            try:
                if _is_async:
                    # Async run() — call directly and await (runs in event loop)
                    result = await asyncio.wait_for(
                        fn(params),
                        timeout=_EXEC_TIMEOUT,
                    )
                else:
                    # Sync run() — run in thread pool to avoid blocking
                    result = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, lambda: fn(params)
                        ),
                        timeout=_EXEC_TIMEOUT,
                    )
                output = str(result)
                if len(output) > _MAX_OUTPUT_CHARS:
                    output = output[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"
                return output
            except asyncio.TimeoutError:
                return f"[Error] Custom tool '{name}' timed out after {_EXEC_TIMEOUT}s"
            except Exception as e:
                return f"[Error] Custom tool '{name}': {type(e).__name__}: {e}"

        REGISTRY.register(Tool(
            name=name, description=description,
            parameters_schema=schema, handler=handler,
            risk="medium", category="custom",
        ))
        return "ok"
    except SyntaxError as e:
        return f"Syntax error in custom tool: {e}"
    except Exception as e:
        return f"Failed to load custom tool: {e}"


async def load_all_from_db() -> None:
    """Called at startup to register all saved custom tools."""
    for row in await db.list_custom_tools():
        if row.get("enabled"):
            try:
                schema = json.loads(row["schema"])
            except Exception:
                schema = {"type": "object", "properties": {}}
            load_custom_tool(row["name"], row["description"], schema, row["code"])


# ── SKILL.MD Parser ─────────────────────────────────────────────────────────

def parse_skill_md(content: str) -> dict[str, Any]:
    """Parse a SKILL.MD file and extract structured metadata.

    SKILL.MD format (markdown with YAML-like frontmatter):

        ---
        name: my_skill
        description: What this skill does
        category: custom
        icon: PLG
        ---

        # Skill Instructions
        Detailed instructions for the agent...

        ```python
        # Optional: tool implementation (sync or async)
        def run(params: dict) -> str:
            ...
        ```
    """
    result: dict[str, Any] = {
        "name": "",
        "description": "",
        "category": "custom",
        "icon": "PLG",
        "instructions": "",
        "code": "",
        "parameters_schema": {"type": "object", "properties": {}},
    }

    # Parse frontmatter (between --- delimiters)
    fm_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        rest = content[fm_match.end():]
        for line in frontmatter.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key == "name":
                    result["name"] = val
                elif key == "description":
                    result["description"] = val
                elif key == "category":
                    result["category"] = val
                elif key == "icon":
                    result["icon"] = val[:3].upper()
                elif key == "parameters":
                    try:
                        result["parameters_schema"] = json.loads(val)
                    except Exception:
                        pass
    else:
        rest = content

    # Extract Python code blocks
    code_blocks = re.findall(r'```python\s*\n(.*?)\n```', rest, re.DOTALL)
    if code_blocks:
        result["code"] = "\n\n".join(code_blocks)

    # Instructions are the rest of the content minus code blocks
    instructions = rest
    # Remove code blocks from instructions
    instructions = re.sub(r'```\w*\n.*?\n```', '', instructions, flags=re.DOTALL).strip()
    result["instructions"] = instructions

    # Try to derive name from first heading if not in frontmatter
    if not result["name"]:
        heading_match = re.search(r'^#\s+(.+)', rest, re.MULTILINE)
        if heading_match:
            result["name"] = heading_match.group(1).strip().lower().replace(" ", "_")

    # Try to derive description from first paragraph if not in frontmatter
    if not result["description"]:
        para_match = re.search(r'(?:^|\n\n)([A-Za-z][^\n]{10,200})', rest)
        if para_match:
            result["description"] = para_match.group(1).strip()

    return result


# ── Custom Skill Loader ──────────────────────────────────────────────────────

def load_custom_skill(name: str, description: str, category: str, icon: str,
                      skill_md: str, code: str) -> str:
    """Register a custom skill. If code is present, also register as a tool.
    Returns 'ok' or error message.

    v23: Fixed async function handling (same fix as load_custom_tool).
    v23: Skills now require confirmation, same as custom tools — both
    execute arbitrary user code and should have the same safety guard.
    """
    try:
        # If there's Python code, register it as a tool too
        if code and code.strip():
            # Security verification for the code
            source_violations = _verify_source(code)
            if source_violations:
                return f"Security violation: {'; '.join(source_violations[:3])}"

            ns: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
            compiled = compile(code, f"<skill:{name}>", "exec")  # noqa: S102

            byte_violations = _verify_bytecode(compiled)
            if byte_violations:
                return f"Security violation: {'; '.join(byte_violations[:3])}"

            exec(compiled, ns)  # noqa: S102
            fn = ns.get("run")
            if fn and callable(fn):
                # v23: Detect whether run() is async or sync
                _is_async = asyncio.iscoroutinefunction(fn)

                async def handler(params):
                    try:
                        if _is_async:
                            # Async run() — call directly and await
                            result = await asyncio.wait_for(
                                fn(params),
                                timeout=_EXEC_TIMEOUT,
                            )
                        else:
                            # Sync run() — run in thread pool to avoid blocking
                            result = await asyncio.wait_for(
                                asyncio.get_running_loop().run_in_executor(
                                    None, lambda: fn(params)
                                ),
                                timeout=_EXEC_TIMEOUT,
                            )
                        output = str(result)
                        if len(output) > _MAX_OUTPUT_CHARS:
                            output = output[:_MAX_OUTPUT_CHARS] + "\n... [output truncated]"
                        return output
                    except asyncio.TimeoutError:
                        return f"[Error] Skill '{name}' timed out after {_EXEC_TIMEOUT}s"
                    except Exception as e:
                        return f"[Error] Skill '{name}': {type(e).__name__}: {e}"

                # Build a schema from the code's docstring or use default
                schema = {"type": "object", "properties": {"input": {"type": "string", "description": f"Input for {name}"}}}

                # v23: Skills require confirmation, just like custom tools.
                # Both execute arbitrary user code and need the same safety guard.
                REGISTRY.register(Tool(
                    name=name, description=description,
                    parameters_schema=schema, handler=handler,
                    risk="medium", category=category,
                ))

        return "ok"
    except SyntaxError as e:
        return f"Syntax error in custom skill: {e}"
    except Exception as e:
        return f"Failed to load custom skill: {e}"


async def load_all_custom_skills_from_db() -> None:
    """Called at startup to register all saved custom skills."""
    for row in await db.list_custom_skills():
        if row.get("enabled"):
            load_custom_skill(
                name=row["name"],
                description=row.get("description", ""),
                category=row.get("category", "custom"),
                icon=row.get("icon", "PLG"),
                skill_md=row.get("skill_md", ""),
                code=row.get("code", ""),
            )


# ── File Upload Processing ───────────────────────────────────────────────────

async def process_skill_md_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Process an uploaded SKILL.MD file.

    Saves the file, parses it, registers the skill, and stores it in the DB.
    Returns a result dict with 'ok' and skill info, or 'error'.
    """
    text = content.decode("utf-8", errors="replace")
    parsed = parse_skill_md(text)

    if not parsed["name"]:
        return {"error": "SKILL.MD must have a 'name' field in frontmatter or a # heading"}

    # Sanitize name
    skill_name = re.sub(r'[^a-zA-Z0-9_-]', '_', parsed["name"]).strip("_")
    if not skill_name:
        return {"error": "Invalid skill name after sanitization"}

    # Save file to custom_skills directory
    skill_dir = _ensure_skills_dir() / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(text, encoding="utf-8")

    # Register the skill
    status = load_custom_skill(
        name=skill_name,
        description=parsed["description"],
        category=parsed["category"],
        icon=parsed["icon"],
        skill_md=text,
        code=parsed.get("code", ""),
    )
    if status != "ok":
        return {"error": status}

    # Store in DB
    await db.add_custom_skill(
        name=skill_name,
        description=parsed["description"],
        category=parsed["category"],
        icon=parsed["icon"],
        skill_md=text,
        code=parsed.get("code", ""),
        file_path=str(skill_dir),
        source_type="skill_md_upload",
    )

    return {
        "ok": True,
        "name": skill_name,
        "description": parsed["description"],
        "category": parsed["category"],
        "has_code": bool(parsed.get("code", "").strip()),
    }


async def process_skill_zip_upload(filename: str, content: bytes) -> dict[str, Any]:
    """Process an uploaded zip file containing skill(s).

    Looks for SKILL.md or SKILL.MD files inside the zip, extracts them,
    and registers each as a custom skill.  Also extracts any .py files
    in the same directory as the SKILL.md.
    Returns a result dict with 'ok', 'results' list, or 'error'.
    """
    # Save and extract zip
    skill_base = _ensure_skills_dir()
    tmp_dir = tempfile.mkdtemp(dir=str(skill_base))

    try:
        zip_path = os.path.join(tmp_dir, filename)
        with open(zip_path, "wb") as f:
            f.write(content)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Security: check for path traversal
                for member in zf.namelist():
                    if member.startswith("/") or ".." in member:
                        return {"error": f"Zip contains unsafe path: {member}"}
                zf.extractall(tmp_dir)
        except zipfile.BadZipFile:
            return {"error": "Invalid zip file"}

        # Find all SKILL.md files
        results = []
        for root, dirs, files in os.walk(tmp_dir):
            for fname in files:
                if fname.lower() == "skill.md":
                    skill_md_path = Path(root) / fname
                    text = skill_md_path.read_text(encoding="utf-8", errors="replace")
                    parsed = parse_skill_md(text)

                    if not parsed["name"]:
                        continue

                    skill_name = re.sub(r'[^a-zA-Z0-9_-]', '_', parsed["name"]).strip("_")
                    if not skill_name:
                        continue

                    # Look for companion .py files in the same directory
                    companion_code = parsed.get("code", "")
                    for py_file in Path(root).glob("*.py"):
                        if py_file.name.lower().startswith("skill") or py_file.name.lower().startswith(skill_name.lower()):
                            companion_code += "\n\n" + py_file.read_text(encoding="utf-8", errors="replace")

                    # Move to permanent location
                    dest_dir = skill_base / skill_name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    shutil.copytree(root, dest_dir)

                    # Register
                    status = load_custom_skill(
                        name=skill_name,
                        description=parsed["description"],
                        category=parsed["category"],
                        icon=parsed["icon"],
                        skill_md=text,
                        code=companion_code,
                    )

                    result_entry = {
                        "name": skill_name,
                        "description": parsed["description"],
                        "category": parsed["category"],
                        "has_code": bool(companion_code.strip()),
                    }

                    if status != "ok":
                        result_entry["error"] = status
                    else:
                        result_entry["ok"] = True
                        # Store in DB
                        await db.add_custom_skill(
                            name=skill_name,
                            description=parsed["description"],
                            category=parsed["category"],
                            icon=parsed["icon"],
                            skill_md=text,
                            code=companion_code,
                            file_path=str(dest_dir),
                            source_type="zip_upload",
                        )

                    results.append(result_entry)

        if not results:
            return {"error": "No SKILL.md file found in zip archive"}

        return {"ok": True, "results": results}

    finally:
        # Cleanup temp dir (but keep extracted skills)
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


async def delete_custom_skill_files(skill_name: str) -> bool:
    """Remove a custom skill's files from disk. Returns True if deleted."""
    skill_dir = SKILLS_DIR / skill_name
    if skill_dir.exists() and skill_dir.is_dir():
        shutil.rmtree(skill_dir, ignore_errors=True)
        return True
    return False


# ── Get Custom Skills for Agent Awareness ────────────────────────────────────

def get_custom_skills_for_prompt() -> str:
    """Build a summary of all loaded custom skills for the agent's system prompt.

    This is called during system prompt construction so the agent knows
    what custom skills are available and how to use them.

    v23: Added prompt bloat limit — total output is capped at _MAX_PROMPT_CHARS
    to prevent 20+ skills from injecting 10,000+ tokens into every message.
    When the budget is exceeded, skills are listed with name+description only
    (no inline instructions). Skills with code are always shown since the
    agent needs to know they're callable tools.
    """
    # v23: Prompt budget limit (approx 500 tokens ≈ 2000 chars)
    _MAX_PROMPT_CHARS = 2000
    # How many chars of instructions to include per skill when budget allows
    _INSTR_PREVIEW_SHORT = 150

    if not _custom_skills_cache:
        return ""

    # Build compact entries first (name + description only)
    compact_lines: list[str] = []
    full_lines: list[str] = []

    for skill in _custom_skills_cache:
        status = "ENABLED" if skill.get("enabled") else "DISABLED"
        name = skill["name"]
        desc = skill.get("description", "No description")
        category = skill.get("category", "custom")
        has_code = bool(skill.get("code"))

        # Compact version (always fits)
        compact = f"- **{name}** ({category}): {desc} [{status}]"
        if has_code:
            compact += " [has tool]"
        compact_lines.append(compact)

        # Full version (with instructions preview)
        full = compact
        if skill.get("skill_md"):
            parsed = parse_skill_md(skill["skill_md"])
            if parsed.get("instructions"):
                # v23: Use shorter preview to fit more skills
                instr_preview = parsed["instructions"][:_INSTR_PREVIEW_SHORT]
                if len(parsed["instructions"]) > _INSTR_PREVIEW_SHORT:
                    instr_preview += "..."
                full += f"\n  Instructions: {instr_preview}"
        full_lines.append(full)

    # Try full version first; if too long, fall back to compact
    header = "# Custom Skills (User-Installed)"
    footer = "When a user's request matches a custom skill, use it automatically."

    full_result = header + "\n\n" + "\n".join(full_lines) + "\n\n" + footer
    if len(full_result) <= _MAX_PROMPT_CHARS:
        return full_result

    # Budget exceeded — use compact version (name + desc only, no instructions)
    compact_result = header + "\n\n" + "\n".join(compact_lines) + "\n\n" + footer
    if len(compact_result) <= _MAX_PROMPT_CHARS:
        return compact_result

    # Still too long even with compact — truncate to the first N skills that fit
    lines_that_fit: list[str] = []
    budget = _MAX_PROMPT_CHARS - len(header) - len(footer) - 10
    for line in compact_lines:
        if budget - len(line) - 1 < 0:
            break
        lines_that_fit.append(line)
        budget -= len(line) + 1

    remaining = len(compact_lines) - len(lines_that_fit)
    result = header + "\n\n" + "\n".join(lines_that_fit)
    if remaining > 0:
        result += f"\n... and {remaining} more skills (details omitted to save context)"
    result += "\n\n" + footer
    return result


# Cache for custom skills (refreshed on load/toggle)
_custom_skills_cache: list[dict] = []


async def refresh_custom_skills_cache() -> None:
    """Refresh the in-memory cache of custom skills from the database."""
    global _custom_skills_cache
    _custom_skills_cache = await db.list_custom_skills()


# Auto-load is deferred — call load_all_from_db() and load_all_custom_skills_from_db()
# from async startup (e.g. on_startup in server.py) after db.init().
