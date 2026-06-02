"""
Full-stack development tools — project scaffolding, dev server management, and templates.

Provides:
- fullstack_scaffold  — Create new projects from common templates (Next.js, Vite+React, Express, etc.)
- fullstack_status    — Check status of dev servers and project state
- fullstack_template  — Inject pre-built full-stack project templates with best-practice structure

All operations are sandboxed to the workspace directory.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ...config import BASE_DIR
from ..registry import tool

logger = logging.getLogger(__name__)

_WORKSPACE = BASE_DIR / "data" / "workspace"
_WORKSPACE.mkdir(parents=True, exist_ok=True)

# ── Dev server tracking ──────────────────────────────────────────────────────
# Track running dev servers so we can check status and suggest reconnection
_dev_servers: dict[str, dict[str, Any]] = {}  # project -> {port, pid, started_at, type}


def _safe_project_path(project: str) -> Path:
    """Resolve a project path safely within the workspace."""
    sandbox = _WORKSPACE.resolve()
    p = Path(project)
    if p.is_absolute():
        p = p.resolve()
    else:
        p = (sandbox / project).resolve()
    if sandbox not in p.parents and p != sandbox:
        raise ValueError(f"Project path escape attempt: {project}")
    return p


# ── Scaffold templates ───────────────────────────────────────────────────────

SCAFFOLD_TEMPLATES = {
    "nextjs": {
        "name": "Next.js (App Router)",
        "description": "Full-stack Next.js with App Router, TypeScript, Tailwind CSS",
        "command": "npx create-next-app@latest {project} --typescript --tailwind --eslint --app --src-dir --import-alias '@/*' --use-npm",
        "post_install": "npm run dev",
        "port": 3000,
        "files": {},
    },
    "vite-react": {
        "name": "Vite + React (TypeScript)",
        "description": "Fast Vite dev server with React, TypeScript, and basic project structure",
        "command": "npm create vite@latest {project} -- --template react-ts",
        "post_install": "cd {project} && npm install && npm run dev",
        "port": 5173,
        "files": {},
    },
    "express-api": {
        "name": "Express REST API",
        "description": "Express.js REST API with TypeScript, routing, middleware, and error handling",
        "command": None,  # Uses file-based scaffolding below
        "post_install": "cd {project} && npm install",
        "port": 3001,
        "files": {
            "package.json": json.dumps({
                "name": "{project}",
                "version": "1.0.0",
                "scripts": {
                    "dev": "tsx watch src/index.ts",
                    "build": "tsc",
                    "start": "node dist/index.js",
                },
                "dependencies": {
                    "express": "^4.18.2",
                    "cors": "^2.8.5",
                    "helmet": "^7.1.0",
                    "morgan": "^1.10.0",
                },
                "devDependencies": {
                    "typescript": "^5.3.3",
                    "tsx": "^4.7.0",
                    "@types/express": "^4.17.21",
                    "@types/cors": "^2.8.17",
                    "@types/morgan": "^1.9.9",
                    "@types/node": "^20.11.0",
                },
            }, indent=2),
            "tsconfig.json": json.dumps({
                "compilerOptions": {
                    "target": "ES2022",
                    "module": "ESNext",
                    "moduleResolution": "bundler",
                    "esModuleInterop": True,
                    "strict": True,
                    "outDir": "dist",
                    "rootDir": "src",
                    "skipLibCheck": True,
                    "forceConsistentCasingInFileNames": True,
                },
                "include": ["src"],
            }, indent=2),
            "src/index.ts": (
                "import express from 'express';\n"
                "import cors from 'cors';\n"
                "import helmet from 'helmet';\n"
                "import morgan from 'morgan';\n\n"
                "const app = express();\n"
                "const PORT = process.env.PORT || 3001;\n\n"
                "app.use(helmet());\n"
                "app.use(cors());\n"
                "app.use(express.json());\n"
                "app.use(morgan('dev'));\n\n"
                "app.get('/api/health', (_req, res) => {\n"
                '  res.json({ status: "ok", timestamp: new Date().toISOString() });\n'
                "});\n\n"
                "app.get('/api', (_req, res) => {\n"
                '  res.json({ message: "Express API is running" });\n'
                "});\n\n"
                "app.listen(PORT, () => {\n"
                '  console.log(`🚀 Server running on http://localhost:${PORT}`);\n'
                "});\n"
            ),
            "src/routes/": None,  # Directory marker
            "src/middleware/": None,
            "src/models/": None,
            ".env": "PORT=3001\nNODE_ENV=development\n",
            ".gitignore": "node_modules/\ndist/\n.env\n",
        },
    },
    "fastapi": {
        "name": "FastAPI Python Backend",
        "description": "FastAPI backend with Pydantic models, async SQLAlchemy, and auto-docs",
        "command": None,
        "post_install": "cd {project} && pip install -r requirements.txt",
        "port": 8000,
        "files": {
            "requirements.txt": (
                "fastapi>=0.109.0\n"
                "uvicorn[standard]>=0.27.0\n"
                "pydantic>=2.5.0\n"
                "sqlalchemy[asyncio]>=2.0.25\n"
                "alembic>=1.13.0\n"
                "python-dotenv>=1.0.0\n"
                "httpx>=0.26.0\n"
            ),
            "main.py": (
                "from fastapi import FastAPI\n"
                "from fastapi.middleware.cors import CORSMiddleware\n"
                "import uvicorn\n\n"
                'app = FastAPI(title="FastAPI Project", version="1.0.0")\n\n'
                "app.add_middleware(\n"
                "    CORSMiddleware,\n"
                "    allow_origins=[\"*\"],\n"
                "    allow_credentials=True,\n"
                "    allow_methods=[\"*\"],\n"
                "    allow_headers=[\"*\"],\n"
                ")\n\n\n"
                '@app.get("/api/health")\n'
                "async def health_check():\n"
                '    return {"status": "ok"}\n\n\n'
                '@app.get("/api")\n'
                "async def root():\n"
                '    return {"message": "FastAPI is running"}\n\n\n'
                'if __name__ == "__main__":\n'
                "    uvicorn.run(\"main:app\", host=\"0.0.0.0\", port=8000, reload=True)\n"
            ),
            "models.py": (
                "from pydantic import BaseModel\n"
                "from typing import Optional\n\n\n"
                "class Item(BaseModel):\n"
                "    id: Optional[int] = None\n"
                "    name: str\n"
                "    description: Optional[str] = None\n"
            ),
            ".env": "DATABASE_URL=sqlite:///./dev.db\n",
        },
    },
    "nextjs-pages": {
        "name": "Next.js (Pages Router)",
        "description": "Next.js with Pages Router, TypeScript, and CSS Modules",
        "command": "npx create-next-app@latest {project} --typescript --eslint --use-npm",
        "post_install": "npm run dev",
        "port": 3000,
        "files": {},
    },
}


# ── Tool: fullstack_scaffold ─────────────────────────────────────────────────

@tool(
    name="fullstack_scaffold",
    description=(
        "Scaffold a new full-stack project from a template. "
        "Creates the project folder, installs dependencies, and provides "
        "instructions to start the dev server. "
        "Supported templates: nextjs, vite-react, express-api, fastapi, nextjs-pages."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "enum": list(SCAFFOLD_TEMPLATES.keys()),
                "description": "Project template to use",
            },
            "project": {
                "type": "string",
                "description": "Project folder name (must be a valid directory name)",
            },
            "install_deps": {
                "type": "boolean",
                "default": True,
                "description": "Run npm/pip install after scaffolding",
            },
        },
        "required": ["template", "project"],
    },
    risk="medium",
    category="fullstack_dev",
)
async def fullstack_scaffold(params):
    template_name = params.get("template", "").strip()
    project_name = params.get("project", "").strip()
    install_deps = bool(params.get("install_deps", True))

    if not template_name or template_name not in SCAFFOLD_TEMPLATES:
        available = ", ".join(SCAFFOLD_TEMPLATES.keys())
        return f"Unknown template '{template_name}'. Available: {available}"

    if not project_name or not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
        return "Error: project name must contain only letters, numbers, hyphens, and underscores."

    template = SCAFFOLD_TEMPLATES[template_name]

    try:
        project_dir = _safe_project_path(project_name)
        if project_dir.exists() and any(project_dir.iterdir()):
            return f"Error: project '{project_name}' already exists and is not empty."

        project_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # If template has file-based scaffolding, write files
        if template["files"]:
            for file_rel, content in template["files"].items():
                if content is None:
                    # Directory marker
                    (project_dir / file_rel).mkdir(parents=True, exist_ok=True)
                    created_files.append(f"  DIR  {file_rel}")
                else:
                    file_path = project_dir / file_rel
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_content = content.replace("{project}", project_name)
                    file_path.write_text(file_content, encoding="utf-8")
                    created_files.append(f"  FILE {file_rel} ({len(file_content)} chars)")

        # If template uses a CLI command, wrap it
        if template["command"]:
            cmd = template["command"].format(project=project_name)
            created_files.append(f"\n  Scaffolding via: {cmd}")

        result_lines = [
            f"✅ Scaffolded '{project_name}' from template '{template_name}'",
            f"   Description: {template['description']}",
            f"   Default port: {template['port']}",
            "",
            "Files created:",
            *created_files,
            "",
        ]

        if install_deps and template["files"]:
            result_lines.append("To install dependencies and start:")
            result_lines.append(f"  cd {project_name}")
            result_lines.append(f"  {template['post_install']}")
            result_lines.append("")
            result_lines.append("Or use the dev server to manage automatically.")

        # Write PROJECT.md for cross-session continuity
        project_md = project_dir / "PROJECT.md"
        project_md.write_text(
            f"# {project_name}\n\n"
            f"Template: {template_name}\n"
            f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Description: {template['description']}\n"
            f"Default port: {template['port']}\n\n"
            "## Project State\n\n"
            "- [ ] Dependencies installed\n"
            "- [ ] Dev server started\n"
            "- [ ] First endpoint tested\n\n",
            encoding="utf-8",
        )
        result_lines.append("   📝 PROJECT.md created for context tracking")

        return "\n".join(result_lines)

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Scaffold failed for %s", project_name)
        return f"Error scaffolding project: {e}"


# ── Tool: fullstack_status ───────────────────────────────────────────────────

@tool(
    name="fullstack_status",
    description=(
        "Check the status of all full-stack projects and running dev servers "
        "in the workspace. Shows project structure, dependencies, and server status."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Optional project name to check specific project status",
            },
        },
        "required": [],
    },
    category="fullstack_dev",
)
async def fullstack_status(params):
    project = params.get("project", "").strip()

    try:
        if project:
            return await _check_single_project(project)
        return await _check_all_projects()
    except Exception as e:
        return f"Error checking status: {e}"


async def _check_single_project(project: str) -> str:
    project_dir = _safe_project_path(project)
    if not project_dir.exists():
        return f"Project '{project}' not found in workspace."

    lines = [f"## Project: {project}", ""]

    # Check PROJECT.md
    proj_md = project_dir / "PROJECT.md"
    if proj_md.exists():
        try:
            content = proj_md.read_text(encoding="utf-8")
            lines.append(content[:500])
            lines.append("")
        except Exception:
            pass

    # Check package.json
    pkg_json = project_dir / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            name = pkg.get("name", project)
            scripts = pkg.get("scripts", {})
            deps = len(pkg.get("dependencies", {}))
            dev_deps = len(pkg.get("devDependencies", {}))
            lines.append(f"📦 Node.js project: {name}")
            lines.append(f"   Dependencies: {deps} | DevDeps: {dev_deps}")
            if scripts:
                lines.append(f"   Scripts: {', '.join(scripts.keys())}")
        except Exception:
            lines.append("📦 package.json found (parse error)")

    # Check requirements.txt
    req_txt = project_dir / "requirements.txt"
    if req_txt.exists():
        try:
            reqs = req_txt.read_text(encoding="utf-8").strip().splitlines()
            lines.append(f"🐍 Python project: {len(reqs)} dependencies")
        except Exception:
            lines.append("🐍 requirements.txt found")

    # Check for node_modules
    if (project_dir / "node_modules").exists():
        lines.append("   ✅ Dependencies installed")
    elif pkg_json.exists():
        lines.append("   ❌ Dependencies NOT installed (run npm install)")

    # Check for .venv or venv
    if (project_dir / ".venv").exists() or (project_dir / "venv").exists():
        lines.append("   ✅ Virtual environment found")

    # Check dev server status
    dev_info = _dev_servers.get(project)
    if dev_info:
        elapsed = time.time() - dev_info["started_at"]
        lines.append(f"   🟢 Dev server running on port {dev_info['port']} ({elapsed:.0f}s ago)")
    else:
        lines.append("   ⚪ Dev server not running")

    # Detect project type from structure
    has_next = (project_dir / "next.config.js").exists() or (project_dir / "next.config.mjs").exists()
    has_vite = (project_dir / "vite.config.ts").exists() or (project_dir / "vite.config.js").exists()
    has_fastapi = (project_dir / "main.py").exists() and "fastapi" in (req_txt.read_text(encoding="utf-8") if req_txt.exists() else "")
    has_express = pkg_json.exists() and "express" in pkg_json.read_text(encoding="utf-8") if pkg_json.exists() else False

    if has_next:
        lines.append(f"   🏷️  Type: Next.js (default port 3000)")
    elif has_vite:
        lines.append(f"   🏷️  Type: Vite (default port 5173)")
    elif has_fastapi:
        lines.append(f"   🏷️  Type: FastAPI (default port 8000)")
    elif has_express:
        lines.append(f"   🏷️  Type: Express (default port 3001)")

    return "\n".join(lines)


async def _check_all_projects() -> str:
    lines = ["## Full-Stack Projects in Workspace", ""]
    project_found = False

    for entry in sorted(_WORKSPACE.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Heuristic: look for project files
        has_project_file = (
            (entry / "package.json").exists()
            or (entry / "requirements.txt").exists()
            or (entry / "PROJECT.md").exists()
            or (entry / "main.py").exists()
            or (entry / "index.html").exists()
        )
        if not has_project_file:
            continue

        project_found = True
        proj_md = entry / "PROJECT.md"
        if proj_md.exists():
            try:
                first_line = proj_md.read_text(encoding="utf-8").splitlines()[0]
                name = first_line.lstrip("# ").strip()
            except Exception:
                name = entry.name
        else:
            name = entry.name

        dev_info = _dev_servers.get(entry.name)
        status = "🟢" if dev_info else "⚪"
        dev_port = f" (port {dev_info['port']})" if dev_info else ""
        lines.append(f"  {status} {name}{dev_port}")

    if not project_found:
        lines.append("  No full-stack projects found in workspace.")
        lines.append("  Use fullstack_scaffold to create one.")

    return "\n".join(lines)


# ── Tool: fullstack_template ─────────────────────────────────────────────────

@tool(
    name="fullstack_template",
    description=(
        "Inject pre-built code templates into an existing full-stack project. "
        "Provides common patterns like API routes, database models, auth middleware, "
        "form components, and data tables."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project folder name in workspace",
            },
            "pattern": {
                "type": "string",
                "enum": [
                    "api_crud",       # REST CRUD endpoints
                    "auth_middleware",  # JWT/basic auth
                    "db_model",        # SQLAlchemy/Prisma model
                    "form_component",  # React form with validation
                    "data_table",      # React data table with sorting
                    "docker_compose",  # Docker Compose for full stack
                ],
                "description": "Code pattern to inject",
            },
            "target": {
                "type": "string",
                "description": "Optional target file path relative to project",
            },
        },
        "required": ["project", "pattern"],
    },
    category="fullstack_dev",
)
async def fullstack_template(params):
    project = params.get("project", "").strip()
    pattern = params.get("pattern", "").strip()
    target = params.get("target", "").strip()

    try:
        project_dir = _safe_project_path(project)
        if not project_dir.exists():
            return f"Project '{project}' not found."

        template_code = _get_template_code(pattern)
        if not template_code:
            return f"Unknown pattern '{pattern}'."

        # Determine file path
        if not target:
            target = _suggest_target_path(project_dir, pattern)

        file_path = project_dir / target
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            return (f"File '{target}' already exists. "
                    f"\n\nTemplate content for '{pattern}':\n\n{template_code}\n\n"
                    f"Save it manually to a new file.")

        file_path.write_text(template_code, encoding="utf-8")
        return (
            f"✅ Injected '{pattern}' template into {project}/{target}\n"
            f"   ({len(template_code)} chars written)\n\n"
            f"```\n{template_code[:1000]}\n```"
        )

    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


def _suggest_target_path(project_dir: Path, pattern: str) -> str:
    """Suggest a target file path for the given pattern based on project structure."""
    has_src = (project_dir / "src").exists()
    prefix = "src/" if has_src else ""

    suggestions = {
        "api_crud": f"{prefix}routes/items.py",
        "auth_middleware": f"{prefix}middleware/auth.py",
        "db_model": f"{prefix}models/user.py",
        "form_component": f"{prefix}components/Form.tsx",
        "data_table": f"{prefix}components/DataTable.tsx",
        "docker_compose": "docker-compose.yml",
    }
    return suggestions.get(pattern, f"{prefix}{pattern}.py")


def _get_template_code(pattern: str) -> str | None:
    """Get code for a given pattern."""
    templates = {
        "api_crud": (
            '# REST CRUD Endpoints Template\n'
            'from fastapi import APIRouter, HTTPException, Depends\n'
            'from typing import List, Optional\n\n'
            'router = APIRouter(prefix="/api/items", tags=["items"])\n\n'
            '# In-memory storage (replace with DB)\n'
            'items_db: list[dict] = []\n'
            '_next_id: int = 1\n\n\n'
            '@router.get("/")\n'
            'async def list_items(skip: int = 0, limit: int = 100):\n'
            '    """List all items with pagination."""\n'
            '    return {"items": items_db[skip:skip + limit], "total": len(items_db)}\n\n\n'
            '@router.get("/{item_id}")\n'
            'async def get_item(item_id: int):\n'
            '    """Get a single item by ID."""\n'
            '    for item in items_db:\n'
            '        if item["id"] == item_id:\n'
            '            return item\n'
            '    raise HTTPException(status_code=404, detail="Item not found")\n\n\n'
            '@router.post("/")\n'
            'async def create_item(item: dict):\n'
            '    """Create a new item."""\n'
            '    global _next_id\n'
            '    new_item = {"id": _next_id, **item}\n'
            '    _next_id += 1\n'
            '    items_db.append(new_item)\n'
            '    return new_item\n\n\n'
            '@router.put("/{item_id}")\n'
            'async def update_item(item_id: int, item: dict):\n'
            '    """Update an existing item."""\n'
            '    for i, existing in enumerate(items_db):\n'
            '        if existing["id"] == item_id:\n'
            '            items_db[i] = {"id": item_id, **item}\n'
            '            return items_db[i]\n'
            '    raise HTTPException(status_code=404, detail="Item not found")\n\n\n'
            '@router.delete("/{item_id}")\n'
            'async def delete_item(item_id: int):\n'
            '    """Delete an item."""\n'
            '    for i, existing in enumerate(items_db):\n'
            '        if existing["id"] == item_id:\n'
            '            items_db.pop(i)\n'
            '            return {"message": "Item deleted"}\n'
            '    raise HTTPException(status_code=404, detail="Item not found")\n'
        ),
        "auth_middleware": (
            '# Authentication Middleware Template\n'
            'from fastapi import Request, HTTPException, Depends\n'
            'from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\n'
            'import jwt\n'
            'from datetime import datetime, timedelta\n\n'
            'security = HTTPBearer()\n\n'
            '# FIXME: Move to environment variable\n'
            'SECRET_KEY = "your-secret-key-change-in-production"\n'
            'ALGORITHM = "HS256"\n'
            'ACCESS_TOKEN_EXPIRE_MINUTES = 30\n\n\n'
            'def create_access_token(data: dict) -> str:\n'
            '    """Create a JWT access token."""\n'
            '    to_encode = data.copy()\n'
            '    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)\n'
            '    to_encode.update({"exp": expire})\n'
            '    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n\n\n'
            'async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:\n'
            '    """Verify and decode a JWT token."""\n'
            '    token = credentials.credentials\n'
            '    try:\n'
            '        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n'
            '        return payload\n'
            '    except jwt.ExpiredSignatureError:\n'
            '        raise HTTPException(status_code=401, detail="Token has expired")\n'
            '    except jwt.InvalidTokenError:\n'
            '        raise HTTPException(status_code=401, detail="Invalid token")\n'
        ),
        "db_model": (
            '# Database Model Template (SQLAlchemy Async)\n'
            'from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey\n'
            'from sqlalchemy.ext.declarative import declarative_base\n'
            'from sqlalchemy.orm import relationship\n'
            'from datetime import datetime\n\n'
            'Base = declarative_base()\n\n\n'
            'class User(Base):\n'
            '    """User model with timestamps."""\n'
            '    __tablename__ = "users"\n\n'
            '    id = Column(Integer, primary_key=True, index=True)\n'
            '    email = Column(String(255), unique=True, index=True, nullable=False)\n'
            '    username = Column(String(100), unique=True, index=True, nullable=False)\n'
            '    hashed_password = Column(String(255), nullable=False)\n'
            '    is_active = Column(Boolean, default=True)\n'
            '    created_at = Column(DateTime, default=datetime.utcnow)\n'
            '    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)\n\n'
            '    # Relationships\n'
            '    # posts = relationship("Post", back_populates="author")\n\n'
            '    def __repr__(self):\n'
            '        return f"<User(id={self.id}, username={self.username})>"\n'
        ),
        "form_component": (
            '// React Form Component with Validation\n'
            'import React, { useState } from "react";\n\n'
            'interface FormField {\n'
            '  name: string;\n'
            '  label: string;\n'
            '  type: "text" | "email" | "password" | "number" | "textarea" | "select";\n'
            '  required?: boolean;\n'
            '  options?: { value: string; label: string }[];\n'
            '  validation?: (value: string) => string | undefined;\n'
            '}\n\n'
            'interface FormProps {\n'
            '  fields: FormField[];\n'
            '  onSubmit: (values: Record<string, string>) => Promise<void>;\n'
            '  submitLabel?: string;\n'
            '}\n\n'
            'export default function Form({ fields, onSubmit, submitLabel = "Submit" }: FormProps) {\n'
            '  const [values, setValues] = useState<Record<string, string>>({});\n'
            '  const [errors, setErrors] = useState<Record<string, string>>({});\n'
            '  const [submitting, setSubmitting] = useState(false);\n\n'
            '  const handleChange = (name: string, value: string) => {\n'
            '    setValues((prev) => ({ ...prev, [name]: value }));\n'
            '    // Clear error on change\n'
            '    if (errors[name]) {\n'
            '      setErrors((prev) => {\n'
            '        const next = { ...prev };\n'
            '        delete next[name];\n'
            '        return next;\n'
            '      });\n'
            '    }\n'
            '  };\n\n'
            '  const validate = (): boolean => {\n'
            '    const newErrors: Record<string, string> = {};\n'
            '    for (const field of fields) {\n'
            '      if (field.validation) {\n'
            '        const error = field.validation(values[field.name] || "");\n'
            '        if (error) newErrors[field.name] = error;\n'
            '      }\n'
            '      if (field.required && !values[field.name]?.trim()) {\n'
            '        newErrors[field.name] = `${field.label} is required`;\n'
            '      }\n'
            '    }\n'
            '    setErrors(newErrors);\n'
            '    return Object.keys(newErrors).length === 0;\n'
            '  };\n\n'
            '  const handleSubmit = async (e: React.FormEvent) => {\n'
            '    e.preventDefault();\n'
            '    if (!validate() || submitting) return;\n'
            '    setSubmitting(true);\n'
            '    try {\n'
            '      await onSubmit(values);\n'
            '    } catch (err) {\n'
            '      setErrors({ _form: err instanceof Error ? err.message : "Submission failed" });\n'
            '    } finally {\n'
            '      setSubmitting(false);\n'
            '    }\n'
            '  };\n\n'
            '  return (\n'
            '    <form onSubmit={handleSubmit} className="space-y-4">\n'
            '      {errors._form && (\n'
            '        <div className="rounded bg-red-50 p-3 text-red-700 text-sm">{errors._form}</div>\n'
            '      )}\n'
            '      {fields.map((field) => (\n'
            '        <div key={field.name}>\n'
            '          <label className="block text-sm font-medium text-gray-700 mb-1">\n'
            '            {field.label}\n'
            '            {field.required && <span className="text-red-500 ml-1">*</span>}\n'
            '          </label>\n'
            '          {field.type === "textarea" ? (\n'
            '            <textarea\n'
            '              value={values[field.name] || ""}\n'
            '              onChange={(e) => handleChange(field.name, e.target.value)}\n'
            '              className="w-full rounded border p-2 text-sm"\n'
            '              rows={4}\n'
            '            />\n'
            '          ) : field.type === "select" ? (\n'
            '            <select\n'
            '              value={values[field.name] || ""}\n'
            '              onChange={(e) => handleChange(field.name, e.target.value)}\n'
            '              className="w-full rounded border p-2 text-sm"\n'
            '            >\n'
            '              <option value="">Select...</option>\n'
            '              {field.options?.map((opt) => (\n'
            '                <option key={opt.value} value={opt.value}>{opt.label}</option>\n'
            '              ))}\n'
            '            </select>\n'
            '          ) : (\n'
            '            <input\n'
            '              type={field.type}\n'
            '              value={values[field.name] || ""}\n'
            '              onChange={(e) => handleChange(field.name, e.target.value)}\n'
            '              className="w-full rounded border p-2 text-sm"\n'
            '            />\n'
            '          )}\n'
            '          {errors[field.name] && (\n'
            '            <p className="mt-1 text-sm text-red-600">{errors[field.name]}</p>\n'
            '          )}\n'
            '        </div>\n'
            '      )}\n'
            '      <button\n'
            '        type="submit"\n'
            '        disabled={submitting}\n'
            '        className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:opacity-50"\n'
            '      >\n'
            '        {submitting ? "Submitting..." : submitLabel}\n'
            '      </button>\n'
            '    </form>\n'
            '  );\n'
            '}\n'
        ),
        "data_table": (
            '// React Data Table Component with Sorting\n'
            'import React, { useState, useMemo } from "react";\n\n'
            'interface Column<T> {\n'
            '  key: string;\n'
            '  label: string;\n'
            '  sortable?: boolean;\n'
            '  render?: (value: any, row: T) => React.ReactNode;\n'
            '}\n\n'
            'interface DataTableProps<T> {\n'
            '  columns: Column<T>[];\n'
            '  data: T[];\n'
            '  pageSize?: number;\n'
            '}\n\n'
            'export default function DataTable<T extends Record<string, any>>({\n'
            '  columns,\n'
            '  data,\n'
            '  pageSize = 10,\n'
            '}: DataTableProps<T>) {\n'
            '  const [sortKey, setSortKey] = useState<string | null>(null);\n'
            '  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");\n'
            '  const [page, setPage] = useState(0);\n\n'
            '  const sorted = useMemo(() => {\n'
            '    if (!sortKey) return data;\n'
            '    return [...data].sort((a, b) => {\n'
            '      const aVal = a[sortKey];\n'
            '      const bVal = b[sortKey];\n'
            '      if (aVal < bVal) return sortDir === "asc" ? -1 : 1;\n'
            '      if (aVal > bVal) return sortDir === "asc" ? 1 : -1;\n'
            '      return 0;\n'
            '    });\n'
            '  }, [data, sortKey, sortDir]);\n\n'
            '  const totalPages = Math.ceil(sorted.length / pageSize);\n'
            '  const pageData = sorted.slice(page * pageSize, (page + 1) * pageSize);\n\n'
            '  const handleSort = (key: string) => {\n'
            '    if (sortKey === key) {\n'
            '      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));\n'
            '    } else {\n'
            '      setSortKey(key);\n'
            '      setSortDir("asc");\n'
            '    }\n'
            '  };\n\n'
            '  return (\n'
            '    <div>\n'
            '      <div className="overflow-x-auto rounded border">\n'
            '        <table className="min-w-full divide-y divide-gray-200">\n'
            '          <thead className="bg-gray-50">\n'
            '            <tr>\n'
            '              {columns.map((col) => (\n'
            '                <th\n'
            '                  key={col.key}\n'
            '                  onClick={() => col.sortable !== false && handleSort(col.key)}\n'
            '                  className={`px-4 py-2 text-left text-sm font-medium text-gray-500 ${\n'
            '                    col.sortable !== false ? "cursor-pointer hover:bg-gray-100" : ""\n'
            '                  }`}\n'
            '                >\n'
            '                  {col.label}\n'
            '                  {sortKey === col.key && (\n'
            '                    <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>\n'
            '                  )}\n'
            '                </th>\n'
            '              ))}\n'
            '            </tr>\n'
            '          </thead>\n'
            '          <tbody className="divide-y divide-gray-200">\n'
            '            {pageData.map((row, i) => (\n'
            '              <tr key={i} className="hover:bg-gray-50">\n'
            '                {columns.map((col) => (\n'
            '                  <td key={col.key} className="px-4 py-2 text-sm">\n'
            '                    {col.render ? col.render(row[col.key], row) : String(row[col.key] ?? "")}\n'
            '                  </td>\n'
            '                ))}\n'
            '              </tr>\n'
            '            ))}\n'
            '          </tbody>\n'
            '        </table>\n'
            '      </div>\n'
            '      {totalPages > 1 && (\n'
            '        <div className="flex items-center justify-between mt-4">\n'
            '          <button\n'
            '            onClick={() => setPage(Math.max(0, page - 1))}\n'
            '            disabled={page === 0}\n'
            '            className="rounded px-3 py-1 text-sm border disabled:opacity-50"\n'
            '          >\n'
            '            Previous\n'
            '          </button>\n'
            '          <span className="text-sm text-gray-600">\n'
            '            Page {page + 1} of {totalPages}\n'
            '          </span>\n'
            '          <button\n'
            '            onClick={() => setPage(Math.min(totalPages - 1, page + 1))}\n'
            '            disabled={page >= totalPages - 1}\n'
            '            className="rounded px-3 py-1 text-sm border disabled:opacity-50"\n'
            '          >\n'
            '            Next\n'
            '          </button>\n'
            '        </div>\n'
            '      )}\n'
            '    </div>\n'
            '  );\n'
            '}\n'
        ),
        "docker_compose": (
            '# Docker Compose for Full-Stack Development\n'
            'version: "3.8"\n\n'
            'services:\n'
            '  frontend:\n'
            '    build:\n'
            '      context: ./frontend\n'
            '      dockerfile: Dockerfile\n'
            '    ports:\n'
            '      - "3000:3000"\n'
            '    environment:\n'
            '      - NEXT_PUBLIC_API_URL=http://localhost:8000/api\n'
            '    depends_on:\n'
            '      - backend\n'
            '    volumes:\n'
            '      - ./frontend:/app\n'
            '      - /app/node_modules\n\n'
            '  backend:\n'
            '    build:\n'
            '      context: ./backend\n'
            '      dockerfile: Dockerfile\n'
            '    ports:\n'
            '      - "8000:8000"\n'
            '    environment:\n'
            '      - DATABASE_URL=postgresql://postgres:postgres@db:5432/app\n'
            '    depends_on:\n'
            '      - db\n'
            '    volumes:\n'
            '      - ./backend:/app\n\n'
            '  db:\n'
            '    image: postgres:16-alpine\n'
            '    ports:\n'
            '      - "5432:5432"\n'
            '    environment:\n'
            '      - POSTGRES_USER=postgres\n'
            '      - POSTGRES_PASSWORD=postgres\n'
            '      - POSTGRES_DB=app\n'
            '    volumes:\n'
            '      - pgdata:/var/lib/postgresql/data\n\n'
            'volumes:\n'
            '  pgdata:\n'
        ),
    }
    return templates.get(pattern)


# ── Dev server utilities (used by shell_session and engine) ──────────────────

def track_dev_server(project: str, port: int, server_type: str = "unknown") -> None:
    """Register a running dev server for status tracking."""
    _dev_servers[project] = {
        "port": port,
        "started_at": time.time(),
        "type": server_type,
    }


def untrack_dev_server(project: str) -> None:
    """Remove a dev server from tracking."""
    _dev_servers.pop(project, None)


def get_dev_server(project: str) -> dict | None:
    """Get dev server info for a project."""
    return _dev_servers.get(project)


def list_dev_servers() -> dict[str, dict]:
    """List all tracked dev servers."""
    return dict(_dev_servers)
