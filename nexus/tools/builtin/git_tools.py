"""
Git tools — full git workflow scoped to workspace projects.
Supports init, clone, status, diff, add, commit, push, pull, log, branch.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...config import BASE_DIR
from ..registry import tool

_WORKSPACE = Path(BASE_DIR) / "data" / "workspace"


async def _git(args: list[str], cwd: Path, timeout: int = 30) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        result = out.decode("utf-8", errors="replace").strip()
        error  = err.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            return f"[git error rc={proc.returncode}]\n{error or result}"
        return result or error or "(ok)"
    except asyncio.TimeoutError:
        return "Error: git command timed out"
    except FileNotFoundError:
        return "Error: git is not installed or not in PATH"
    except Exception as e:
        return f"Error: {e}"


def _repo_path(project: str) -> Path:
    p = Path(project)
    if p.is_absolute():
        p = p.resolve()
        ws = _WORKSPACE.resolve()
    else:
        p = (_WORKSPACE / project).resolve()
        ws = _WORKSPACE.resolve()
    if ws not in p.parents and p != ws:
        raise ValueError(f"Path escape: {project}")
    return p


@tool(
    name="git_init",
    description="Initialise a git repo inside a workspace project folder.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "Project folder name inside workspace"},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_init(params):
    p = _repo_path(params["project"])
    p.mkdir(parents=True, exist_ok=True)
    return await _git(["init"], p)


@tool(
    name="git_clone",
    description="Clone a remote git repository into the workspace.",
    parameters_schema={
        "type": "object",
        "properties": {
            "url":     {"type": "string", "description": "Remote URL to clone"},
            "folder":  {"type": "string", "description": "Target folder name in workspace"},
        },
        "required": ["url"],
    },
    risk="medium",
    category="git",
)
async def git_clone(params):
    url    = params["url"]
    folder = params.get("folder", url.rstrip("/").split("/")[-1].replace(".git", ""))
    target = _WORKSPACE / folder
    target = target.resolve()
    ws_resolved = Path(_WORKSPACE).resolve()
    if ws_resolved not in target.parents and target != ws_resolved:
        return "Error: Cannot clone outside the workspace directory"
    return await _git(["clone", url, str(target)], _WORKSPACE, timeout=120)


@tool(
    name="git_status",
    description="Show the working tree status of a workspace project.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_status(params):
    return await _git(["status", "--short", "--branch"], _repo_path(params["project"]))


@tool(
    name="git_diff",
    description="Show unstaged or staged diffs in a project. Optionally diff a specific file.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "file":    {"type": "string", "description": "Optional: specific file to diff"},
            "staged":  {"type": "boolean", "default": False},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_diff(params):
    args = ["diff"]
    if params.get("staged"):
        args.append("--cached")
    if f := params.get("file"):
        args += ["--", f]
    result = await _git(args, _repo_path(params["project"]))
    return result[:8000] if len(result) > 8000 else result


@tool(
    name="git_add",
    description="Stage files for commit. Use '.' to stage all changes.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "files":   {"type": "string", "default": ".", "description": "File pattern to stage"},
        },
        "required": ["project"],
    },
    risk="medium",
    category="git",
)
async def git_add(params):
    return await _git(["add", params.get("files", ".")], _repo_path(params["project"]))


@tool(
    name="git_commit",
    description="Commit staged changes with a message.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["project", "message"],
    },
    risk="medium",
    category="git",
)
async def git_commit(params):
    return await _git(
        ["commit", "-m", params["message"],
         "--author=Hyper Nexus Agent <nexus@local>"],
        _repo_path(params["project"]),
    )


@tool(
    name="git_push",
    description="Push committed changes to the remote.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "branch":  {"type": "string", "default": "main"},
        },
        "required": ["project"],
    },
    risk="high",
    category="git",
)
async def git_push(params):
    branch = params.get("branch", "main")
    return await _git(["push", "origin", branch], _repo_path(params["project"]), timeout=60)


@tool(
    name="git_pull",
    description="Pull latest changes from remote.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_pull(params):
    return await _git(["pull"], _repo_path(params["project"]), timeout=60)


@tool(
    name="git_log",
    description="Show recent commit history for a project.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "limit":   {"type": "integer", "default": 10},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_log(params):
    n = str(min(50, int(params.get("limit", 10))))
    return await _git(
        ["log", f"-{n}", "--oneline", "--graph", "--decorate"],
        _repo_path(params["project"]),
    )


@tool(
    name="git_branch",
    description="List, create, or switch branches in a project.",
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "action":  {"type": "string", "enum": ["list", "create", "switch"],
                        "default": "list"},
            "name":    {"type": "string", "description": "Branch name (for create/switch)"},
        },
        "required": ["project"],
    },
    category="git",
)
async def git_branch(params):
    action = params.get("action", "list")
    name   = params.get("name", "")
    path   = _repo_path(params["project"])
    if action == "list":
        return await _git(["branch", "-a"], path)
    elif action == "create":
        return await _git(["checkout", "-b", name], path)
    elif action == "switch":
        return await _git(["checkout", name], path)
    return "Unknown action"
