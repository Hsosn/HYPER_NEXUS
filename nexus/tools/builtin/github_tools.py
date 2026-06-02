"""
GitHub API tools via REST v3.
Requires github_token in Settings.
"""
from __future__ import annotations

import json
import httpx

from ...config import get as cfg
from ..registry import tool

_BASE = "https://api.github.com"


async def _get_github_token() -> str:
    """Unified: check integration table first, then legacy settings."""
    try:
        from ...memory import db as _db
        intg = await _db.get_integration("github")
        if intg and intg.get("connected"):
            config = await _db.get_integration_config("github")
            token = config.get("token", "") or config.get("api_key", "")
            if token:
                return token
    except Exception:
        pass
    return cfg("github_token", "")


async def _headers() -> dict:
    token = await _get_github_token()
    if not token:
        raise ValueError("GitHub not configured. Connect it in Integrations or set github_token in Settings.")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def _gh(method: str, path: str, json_body: dict | None = None, timeout: float = 60) -> dict | list | str:
    try:
        headers = await _headers()
    except ValueError as e:
        return str(e)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.request(method, f"{_BASE}{path}",
                                     headers=headers, json=json_body)
            if r.status_code >= 400:
                return f"GitHub API error {r.status_code}: {r.text[:400]}"
            if r.text:
                return r.json()
            return "(ok)"
    except Exception as e:
        return f"Request error: {e}"


def _fmt(data) -> str:
    if isinstance(data, str):
        return data
    return json.dumps(data, indent=2)[:6000]


@tool(
    name="github_list_repos",
    description="List your GitHub repositories.",
    parameters_schema={
        "type": "object",
        "properties": {
            "type":    {"type": "string", "enum": ["all","owner","public","private"], "default": "owner"},
            "limit":   {"type": "integer", "default": 20},
        },
        "required": [],
    },
    category="github",
)
async def github_list_repos(params):
    n    = min(100, int(params.get("limit", 20)))
    kind = params.get("type", "owner")
    data = await _gh("GET", f"/user/repos?type={kind}&per_page={n}&sort=updated")
    if isinstance(data, str):
        return data
    lines = [f"## Your Repos ({len(data)})"]
    for r in data:
        lines.append(f"  {r['full_name']} — {r.get('description','') or ''} "
                     f"[{'private' if r['private'] else 'public'}] ⭐{r.get('stargazers_count',0)}")
    return "\n".join(lines)


@tool(
    name="github_get_repo",
    description="Get detailed info about a specific repository.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "owner/repo format"},
        },
        "required": ["repo"],
    },
    category="github",
)
async def github_get_repo(params):
    return _fmt(await _gh("GET", f"/repos/{params['repo']}"))


@tool(
    name="github_list_issues",
    description="List open issues for a repository.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":  {"type": "string", "description": "owner/repo"},
            "state": {"type": "string", "enum": ["open","closed","all"], "default": "open"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["repo"],
    },
    category="github",
)
async def github_list_issues(params):
    n     = min(100, int(params.get("limit", 20)))
    state = params.get("state", "open")
    data  = await _gh("GET", f"/repos/{params['repo']}/issues?state={state}&per_page={n}")
    if isinstance(data, str):
        return data
    lines = [f"## Issues for {params['repo']} ({state})"]
    for i in data:
        if "pull_request" not in i:  # exclude PRs
            lines.append(f"  #{i['number']} [{i['state']}] {i['title']} — @{i['user']['login']}")
    return "\n".join(lines) if len(lines) > 1 else "No issues found."


@tool(
    name="github_create_issue",
    description="Create a new issue in a repository.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":   {"type": "string", "description": "owner/repo"},
            "title":  {"type": "string"},
            "body":   {"type": "string", "default": ""},
            "labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["repo", "title"],
    },
    risk="medium",
    category="github",
)
async def github_create_issue(params):
    data = await _gh("POST", f"/repos/{params['repo']}/issues", {
        "title":  params["title"],
        "body":   params.get("body", ""),
        "labels": params.get("labels", []),
    })
    if isinstance(data, str):
        return data
    return f"Issue created: #{data['number']} — {data['html_url']}"


@tool(
    name="github_list_prs",
    description="List pull requests for a repository.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":  {"type": "string"},
            "state": {"type": "string", "enum": ["open","closed","all"], "default": "open"},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["repo"],
    },
    category="github",
)
async def github_list_prs(params):
    n     = min(100, int(params.get("limit", 20)))
    state = params.get("state", "open")
    data  = await _gh("GET", f"/repos/{params['repo']}/pulls?state={state}&per_page={n}&sort=updated")
    if isinstance(data, str):
        return data
    lines = [f"## PRs for {params['repo']} ({state})"]
    for p in data:
        lines.append(f"  #{p['number']} [{p['state']}] {p['title']} "
                     f"← {p['head']['ref']} → {p['base']['ref']}")
    return "\n".join(lines) if len(lines) > 1 else "No PRs found."


@tool(
    name="github_create_pr",
    description="Create a pull request.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":   {"type": "string"},
            "title":  {"type": "string"},
            "body":   {"type": "string", "default": ""},
            "head":   {"type": "string", "description": "Source branch"},
            "base":   {"type": "string", "default": "main", "description": "Target branch"},
        },
        "required": ["repo", "title", "head"],
    },
    risk="medium",
    category="github",
)
async def github_create_pr(params):
    data = await _gh("POST", f"/repos/{params['repo']}/pulls", {
        "title": params["title"],
        "body":  params.get("body", ""),
        "head":  params["head"],
        "base":  params.get("base", "main"),
    })
    if isinstance(data, str):
        return data
    return f"PR created: #{data['number']} — {data['html_url']}"


@tool(
    name="github_read_file",
    description="Read the contents of a file in a GitHub repository.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":   {"type": "string", "description": "owner/repo"},
            "path":   {"type": "string", "description": "File path in repo"},
            "branch": {"type": "string", "default": "main"},
        },
        "required": ["repo", "path"],
    },
    category="github",
)
async def github_read_file(params):
    import base64
    branch = params.get("branch", "main")
    data   = await _gh("GET", f"/repos/{params['repo']}/contents/{params['path']}?ref={branch}")
    if isinstance(data, str):
        return data
    if isinstance(data, dict) and data.get("encoding") == "base64":
        content = base64.b64decode(data["content"]).decode(errors="replace")
        return f"# {params['path']} ({data.get('size',0)} bytes)\n\n{content[:8000]}"
    return _fmt(data)


@tool(
    name="github_commits",
    description="Show recent commits on a repository branch.",
    parameters_schema={
        "type": "object",
        "properties": {
            "repo":   {"type": "string"},
            "branch": {"type": "string", "default": "main"},
            "limit":  {"type": "integer", "default": 10},
        },
        "required": ["repo"],
    },
    category="github",
)
async def github_commits(params):
    n      = min(50, int(params.get("limit", 10)))
    branch = params.get("branch", "main")
    data   = await _gh("GET", f"/repos/{params['repo']}/commits?sha={branch}&per_page={n}")
    if isinstance(data, str):
        return data
    lines = [f"## Recent commits — {params['repo']} ({branch})"]
    for c in data:
        sha   = c["sha"][:7]
        msg   = c["commit"]["message"].splitlines()[0][:80]
        author = c["commit"]["author"]["name"]
        lines.append(f"  {sha} {msg} — {author}")
    return "\n".join(lines)
