"""
MCP (Model Context Protocol) Server Manager

Manages connections to MCP servers (both stdio and SSE transport).
Provides tool discovery, invocation, and auto-connect capabilities.
The agent can auto-discover and connect to MCP servers for non-technical users.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from typing import Any
from dataclasses import dataclass, field

import httpx

from ...memory import db
from ...events import BUS, emit

logger = logging.getLogger("nexus.mcp")


# ── stdio Content-Length framing reader ────────────────────────────────────────

def _read_stdio_message(proc: subprocess.Popen, timeout: float = 30.0) -> dict | None:
    """Read exactly one JSON-RPC message from a stdio MCP server process.

    The MCP stdio transport uses the **JSON-RPC Content-Length framing** protocol::

        Content-Length: <N>\r\n\r\n<exactly N bytes of JSON>

    This helper:
      1. Reads the header line (``Content-Length: <N>\r\n``),
      2. Skips the empty separator line (``\r\n``),
      3. Reads **exactly** *N* bytes of body,
      4. Parses and returns the JSON dict.

    It correctly handles:
    - Servers that send multiple messages back-to-back (e.g. a response
      followed by a progress notification),
    - Partial reads by looping until the full header and body are consumed,
    - Timeouts via *select* / *poll* so we never block indefinitely.
    """
    import select
    import sys

    # ── Read header ─────────────────────────────────────────────────────────
    header_bytes = b""
    while True:
        # Wait for data with timeout
        if sys.platform == "win32":
            # Windows: no select on pipes — use readline with a timeout thread
            if not _wait_readable_win32(proc.stdout, timeout):
                logger.warning("stdio MCP: timed out waiting for header")
                return None
            chunk = proc.stdout.readline(256)
        else:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                logger.warning("stdio MCP: timed out waiting for header")
                return None
            # Use readline() instead of read(1) for efficient header reading.
            # MCP framing sends "Content-Length: N\r\n" as a single line, so
            # readline() will consume it in one syscall rather than byte-by-byte.
            chunk = proc.stdout.readline(4096)
        if not chunk:
            logger.warning("stdio MCP: process closed while reading header")
            return None
        header_bytes += chunk
        # Header ends with \r\n\r\n (blank line after Content-Length)
        if header_bytes.endswith(b"\r\n\r\n"):
            break
        # Safety: bail if header grows unreasonably large
        if len(header_bytes) > 4096:
            logger.error("stdio MCP: header too large, aborting")
            return None

    # ── Parse Content-Length ─────────────────────────────────────────────────
    header_str = header_bytes.decode("ascii", errors="replace")
    content_length = 0
    for line in header_str.split("\r\n"):
        line = line.strip()
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":", 1)[1].strip())
            except ValueError:
                logger.error("stdio MCP: bad Content-Length header: %r", line)
                return None
            break
    if content_length <= 0:
        logger.error("stdio MCP: Content-Length is 0 or missing")
        return None

    # ── Read exactly content_length bytes ────────────────────────────────────
    body = b""
    remaining = content_length
    while remaining > 0:
        if sys.platform == "win32":
            if not _wait_readable_win32(proc.stdout, timeout):
                logger.warning("stdio MCP: timed out reading body (%d/%d)",
                               content_length - remaining, content_length)
                return None
            chunk = proc.stdout.read(remaining)
        else:
            ready, _, _ = select.select([proc.stdout], [], [], timeout)
            if not ready:
                logger.warning("stdio MCP: timed out reading body (%d/%d)",
                               content_length - remaining, content_length)
                return None
            chunk = proc.stdout.read(remaining)
        if not chunk:
            logger.warning("stdio MCP: process closed while reading body")
            return None
        body += chunk
        remaining -= len(chunk)

    # ── Parse JSON ───────────────────────────────────────────────────────────
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("stdio MCP: invalid JSON body: %s", exc)
        return None


def _wait_readable_win32(stream, timeout: float) -> bool:
    """Best-effort readability check on Windows (no select on pipes)."""
    # On Windows we can't select() on subprocess pipes.
    # We use a thread with a short poll as fallback.
    import threading
    result = [False]
    def _check():
        try:
            # Peek or just try a non-blocking read of 1 byte
            import msvcrt
            if msvcrt.peek(stream.fileno()):  # type: ignore[attr-defined]
                result[0] = True
        except Exception:
            # Fallback: assume ready (will block briefly at worst)
            result[0] = True
    t = threading.Thread(target=_check, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


@dataclass
class MCPTool:
    """Represents a tool exposed by an MCP server."""
    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)
    server_id: str = ""
    server_name: str = ""


@dataclass
class MCPServer:
    """Represents a connected MCP server."""
    id: str
    name: str
    transport: str  # "stdio" or "sse"
    command: str = ""
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    url: str = ""
    headers: dict = field(default_factory=dict)
    description: str = ""
    category: str = "custom"
    status: str = "disconnected"
    connected: bool = False
    tools: list = field(default_factory=list)
    auto_connect: bool = False
    _process: subprocess.Popen | None = None
    _last_health_check: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "transport": self.transport,
            "command": self.command,
            "args": self.args,
            "env": {k: "***" if any(s in k.lower() for s in ["key", "token", "secret", "password"]) else v for k, v in self.env.items()},
            "url": self.url,
            "headers": self.headers,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "connected": self.connected,
            "tools": self.tools,
            "auto_connect": self.auto_connect,
        }


class MCPManager:
    """
    Manages MCP server connections.

    Supports:
    - SSE (Server-Sent Events) transport for remote MCP servers
    - stdio transport for local MCP server processes
    - Auto-discovery and connection for non-technical users
    - Tool discovery and invocation
    - Health monitoring with auto-reconnect
    """

    def __init__(self):
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool
        self._http_client: httpx.AsyncClient | None = None
        self._initialized = False
        # Track which REGISTRY tool names belong to which MCP server_id.
        # Key: server_id, Value: set of REGISTRY tool names (e.g. "mcp_read_file")
        self._registry_tool_map: dict[str, set[str]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def initialize(self):
        """Load all MCP servers from DB and auto-connect those marked for it.

        v22 FIX: Also pre-seeds common MCP server configurations into the DB
        on first run, auto-connects marked servers, registers their tools
        in the REGISTRY, and runs a sync to ensure consistency.
        """
        if self._initialized:
            return
        self._initialized = True
        try:
            servers = await db.list_mcp_servers()

            # FIX: If no servers exist in DB, pre-seed common configurations
            # so first-time users have MCP servers to connect to.
            if not servers:
                await self._preseed_common_servers()

            # Reload after potential pre-seeding
            servers = await db.list_mcp_servers()

            for s in servers:
                server = self._dict_to_server(s)
                self._servers[server.id] = server
                if server.auto_connect and server.connected:
                    try:
                        await self.connect(server.id)
                    except Exception as e:
                        logger.warning(f"Auto-connect failed for {server.name}: {e}")
                        server.status = "error"

            # v22: Sync REGISTRY with current MCP state after initialization
            try:
                sync_result = await self.sync_registry()
                if sync_result["unregistered"] or sync_result["registered"]:
                    logger.info(
                        f"MCP REGISTRY sync: unregistered={len(sync_result['unregistered'])}, "
                        f"registered={len(sync_result['registered'])}"
                    )
            except Exception as e:
                logger.warning(f"MCP REGISTRY sync error: {e}")

        except Exception as e:
            logger.warning(f"MCP initialize error: {e}")

    async def _preseed_common_servers(self) -> None:
        """Pre-seed common MCP server configurations into the database.

        This fixes the bootstrap problem where first-time users hit an empty
        MCP state. These are only stored in DB (not connected) — the user or
        agent must explicitly connect them. But they appear in the UI so users
        know what's available.
        """
        common_servers = [
            {
                "mcp_id": "mcp_preset_filesystem",
                "name": "Filesystem",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "description": "Read/write files on the local filesystem",
                "category": "preset",
                "auto_connect": False,
            },
            {
                "mcp_id": "mcp_preset_github",
                "name": "GitHub",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "description": "Interact with GitHub repos, issues, PRs",
                "category": "preset",
                "auto_connect": False,
            },
            {
                "mcp_id": "mcp_preset_postgres",
                "name": "PostgreSQL",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "description": "Query PostgreSQL databases",
                "category": "preset",
                "auto_connect": False,
            },
            {
                "mcp_id": "mcp_preset_brave_search",
                "name": "Brave Search",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                "description": "Web search via Brave Search API",
                "category": "preset",
                "auto_connect": False,
            },
            {
                "mcp_id": "mcp_preset_memory",
                "name": "Memory",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-memory"],
                "description": "Persistent key-value memory store",
                "category": "preset",
                "auto_connect": False,
            },
        ]

        for cfg in common_servers:
            try:
                await db.upsert_mcp_server(
                    mcp_id=cfg["mcp_id"],
                    name=cfg["name"],
                    transport=cfg["transport"],
                    command=cfg["command"],
                    args=cfg["args"],
                    description=cfg["description"],
                    category=cfg["category"],
                    auto_connect=cfg["auto_connect"],
                )
                logger.info(f"Pre-seeded MCP server: {cfg['name']}")
            except Exception as e:
                logger.warning(f"Failed to pre-seed MCP server {cfg['name']}: {e}")

    def _dict_to_server(self, d: dict) -> MCPServer:
        """Convert a DB dict to MCPServer object."""
        config = {}
        if d.get("config"):
            try:
                config = json.loads(d["config"]) if isinstance(d["config"], str) else d["config"]
            except Exception:
                pass
        tools = []
        if d.get("tools"):
            try:
                tools = json.loads(d["tools"]) if isinstance(d["tools"], str) else d["tools"]
            except Exception:
                pass
        args = []
        if d.get("args"):
            try:
                args = json.loads(d["args"]) if isinstance(d["args"], str) else d["args"]
            except Exception:
                pass
        env = config.get("env", {})
        if d.get("env"):
            try:
                env = json.loads(d["env"]) if isinstance(d["env"], str) else d["env"]
            except Exception:
                pass
        headers = {}
        if d.get("headers"):
            try:
                headers = json.loads(d["headers"]) if isinstance(d["headers"], str) else d["headers"]
            except Exception:
                pass
        return MCPServer(
            id=d["id"],
            name=d.get("name", d["id"]),
            transport=d.get("transport", "stdio"),
            command=d.get("command", "") or config.get("command", ""),
            args=args,
            env=env,
            url=d.get("url", "") or config.get("url", ""),
            headers=headers,
            description=d.get("description", ""),
            category=d.get("category", "custom"),
            status=d.get("status", "disconnected"),
            connected=bool(d.get("connected", 0)),
            tools=tools,
            auto_connect=bool(d.get("auto_connect", 0)),
        )

    async def add_server(self, mcp_id: str | None = None, name: str = "",
                         transport: str = "stdio", command: str = "",
                         args: list | None = None, env: dict | None = None,
                         url: str = "", headers: dict | None = None,
                         description: str = "", category: str = "custom",
                         auto_connect: bool = False,
                         config: dict | None = None) -> MCPServer:
        """Add a new MCP server to the manager and persist to DB."""
        if not mcp_id:
            mcp_id = f"mcp_{uuid.uuid4().hex[:12]}"
        await db.upsert_mcp_server(
            mcp_id=mcp_id, name=name, transport=transport,
            command=command, args=args, env=env, url=url,
            headers=headers, description=description,
            category=category, auto_connect=auto_connect,
            config=config,
        )
        server = MCPServer(
            id=mcp_id, name=name, transport=transport,
            command=command, args=args or [], env=env or {},
            url=url, headers=headers or {}, description=description,
            category=category, auto_connect=auto_connect,
        )
        self._servers[mcp_id] = server
        await emit("mcp_server_added", server_id=mcp_id, name=name)
        return server

    async def connect(self, mcp_id: str) -> dict:
        """Connect to an MCP server and discover its tools.

        For SSE: sends HTTP requests to the server's URL.
        For stdio: launches the server process.

        v22 FIX: Properly auto-registers all discovered MCP tools in the
        tool REGISTRY with tracking. On reconnection, stale entries are
        cleaned up before re-registering.
        """
        server = self._servers.get(mcp_id)
        if not server:
            # Try loading from DB
            row = await db.get_mcp_server(mcp_id)
            if row:
                server = self._dict_to_server(row)
                self._servers[mcp_id] = server
            else:
                raise ValueError(f"MCP server '{mcp_id}' not found")

        try:
            discovered_tools = []

            if server.transport == "sse":
                discovered_tools = await self._connect_sse(server)
            elif server.transport == "stdio":
                discovered_tools = await self._connect_stdio(server)
            else:
                raise ValueError(f"Unknown transport: {server.transport}")

            server.connected = True
            server.status = "connected"
            server.tools = [t if isinstance(t, dict) else {"name": t} for t in discovered_tools]

            # Register discovered tools in the tool index
            for tool_info in discovered_tools:
                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name", "")
                    if tool_name:
                        self._tools[tool_name] = MCPTool(
                            name=tool_name,
                            description=tool_info.get("description", ""),
                            input_schema=tool_info.get("inputSchema", {}),
                            server_id=server.id,
                            server_name=server.name,
                        )

            # v22 FIX: Auto-register discovered MCP tools in the tool REGISTRY
            # Clean up any stale registrations from a previous connection first,
            # then register all discovered tools with proper tracking.
            await self._unregister_server_tools_from_registry(mcp_id)

            registered_names: list[str] = []
            try:
                from ..registry import REGISTRY, Tool as RegistryTool
                for tool_info in discovered_tools:
                    if isinstance(tool_info, dict):
                        original_name = tool_info.get("name", "")
                        mcp_tool_name = f"mcp_{original_name}" if original_name else ""
                        if not mcp_tool_name:
                            continue
                        # If tool already registered (e.g. from another server), skip
                        if REGISTRY.get_tool(mcp_tool_name):
                            # But still track it so we can clean up on disconnect
                            self._registry_tool_map.setdefault(mcp_id, set()).add(mcp_tool_name)
                            logger.debug(f"MCP tool already in REGISTRY, tracking: {mcp_tool_name}")
                            continue
                        _desc = tool_info.get("description", f"MCP tool from {server.name}")
                        _schema = tool_info.get("inputSchema", {"type": "object", "properties": {}})

                        # Create a proper Tool object with a bound handler
                        _sid = server.id
                        _orig_name = original_name

                        async def _mcp_tool_handler(params: dict, _server_id=_sid, _tool_name=_orig_name) -> str:
                            try:
                                result = await self.call_tool(_server_id, _tool_name, params)
                                if isinstance(result, dict):
                                    return json.dumps(result, default=str)[:8000]
                                return str(result)[:8000]
                            except Exception as exc:
                                return f"MCP tool error: {exc}"

                        mcp_tool = RegistryTool(
                            name=mcp_tool_name,
                            description=f"[MCP:{server.name}] {_desc}",
                            parameters_schema=_schema,
                            handler=_mcp_tool_handler,
                            category="MCP",
                            risk="low",
                            tags=["mcp", server.name],
                        )
                        REGISTRY.register(mcp_tool)
                        registered_names.append(mcp_tool_name)
                        logger.info(f"MCP tool registered in REGISTRY: {mcp_tool_name}")

                # Track all registered names for cleanup on disconnect
                self._registry_tool_map[mcp_id] = set(registered_names)

            except Exception as reg_exc:
                logger.warning(f"Failed to auto-register MCP tools in REGISTRY: {reg_exc}")

            # Persist state
            await db.toggle_mcp_server(mcp_id, connected=True)
            await db.update_mcp_server_status(mcp_id, "connected", server.tools)

            logger.info(f"MCP connected: {server.name} ({len(discovered_tools)} tools, {len(registered_names)} registered in REGISTRY)")
            await emit("mcp_connected", server_id=mcp_id, name=server.name, tools=server.tools)
            return {"ok": True, "server": server.to_dict(), "tools": len(discovered_tools), "registered_in_registry": registered_names}

        except Exception as e:
            server.status = "error"
            server.connected = False
            await db.update_mcp_server_status(mcp_id, "error")
            logger.error(f"MCP connect failed for {server.name}: {e}")
            await emit("mcp_error", server_id=mcp_id, error=str(e))
            raise

    async def _connect_sse(self, server: MCPServer) -> list:
        """Connect to an SSE-based MCP server via HTTP."""
        client = await self._get_client()
        base_url = server.url.rstrip("/")

        # Step 1: Initialize connection
        init_ok = False
        try:
            resp = await client.post(
                f"{base_url}/sse",
                headers={**server.headers, "Content-Type": "application/json"},
                json={"jsonrpc": "2.0", "method": "initialize", "id": 1, "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nexus-agent", "version": "8.0.0"},
                }},
            )
            if resp.status_code == 200:
                init_ok = True
            else:
                raise ConnectionError(f"SSE init failed: HTTP {resp.status_code}")
        except Exception as e:
            # Try direct tools/list endpoint as fallback
            pass

        # Step 1b: Send initialized notification (required by MCP protocol)
        if init_ok:
            try:
                await client.post(
                    f"{base_url}/sse",
                    headers={**server.headers, "Content-Type": "application/json"},
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except Exception:
                pass

        # Step 2: List tools
        try:
            resp = await client.post(
                f"{base_url}/mcp/tools/list",
                headers={**server.headers, "Content-Type": "application/json"},
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 2},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", {}).get("tools", [])
        except Exception:
            pass

        # Fallback: try common MCP HTTP patterns
        for endpoint in ["/tools", "/api/tools", "/mcp/v1/tools"]:
            try:
                resp = await client.get(
                    f"{base_url}{endpoint}",
                    headers=server.headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        return data
                    return data.get("tools", [])
            except Exception:
                continue

        return []

    async def _connect_stdio(self, server: MCPServer) -> list:
        """Connect to a stdio-based MCP server by launching its process.

        Uses proper Content-Length framing for all reads.
        """
        cmd = [server.command] + server.args
        env_vars = {**dict(os.environ), **server.env}

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env_vars,
            )
        except FileNotFoundError:
            raise ConnectionError(f"Command not found: {server.command}")

        server._process = proc
        try:
            # ── Send initialize request ───────────────────────────────────────
            init_msg = json.dumps({
                "jsonrpc": "2.0", "method": "initialize", "id": 1,
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nexus-agent", "version": "8.0.0"},
                }
            })
            proc.stdin.write(f"Content-Length: {len(init_msg)}\r\n\r\n{init_msg}".encode())
            proc.stdin.flush()

            # ── Read initialize response (proper framing) ────────────────────
            data = await asyncio.to_thread(_read_stdio_message, proc, timeout=15.0)
            if data is not None:
                # Send initialized notification (no id = notification, no response expected)
                notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
                proc.stdin.write(f"Content-Length: {len(notif)}\r\n\r\n{notif}".encode())
                proc.stdin.flush()
            else:
                logger.warning("stdio MCP: no initialize response from %s", server.command)

            # ── Drain any pending notifications before requesting tools ───────
            await asyncio.to_thread(self._drain_pending_messages, proc, timeout=2.0)

            # ── Request tools/list ────────────────────────────────────────────
            tools_msg = json.dumps({
                "jsonrpc": "2.0", "method": "tools/list", "id": 2,
                "params": {}
            })
            proc.stdin.write(f"Content-Length: {len(tools_msg)}\r\n\r\n{tools_msg}".encode())
            proc.stdin.flush()

            # ── Read tools/list response (proper framing) ─────────────────────
            tools = []
            data = await asyncio.to_thread(_read_stdio_message, proc, timeout=15.0)
            if data is not None:
                tools = data.get("result", {}).get("tools", [])
            else:
                logger.warning("stdio MCP: no tools/list response from %s", server.command)

            return tools
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            raise

    @staticmethod
    def _drain_pending_messages(proc: subprocess.Popen, timeout: float = 2.0) -> None:
        """Read and discard any pending notifications (non-response messages).

        After the initialize handshake some servers send progress/log notifications
        before the client sends its next request.  We drain them so they don't
        get interpreted as the response to the subsequent tools/list call.
        """
        import select
        import sys
        while True:
            if sys.platform == "win32":
                if not _wait_readable_win32(proc.stdout, timeout):
                    break
            else:
                ready, _, _ = select.select([proc.stdout], [], [], timeout)
                if not ready:
                    break
            # Try to read one framed message
            msg = _read_stdio_message(proc, timeout=timeout)
            if msg is None:
                break
            logger.debug("stdio MCP: drained notification: %s", msg.get("method", "?"))

    async def disconnect(self, mcp_id: str) -> dict:
        """Disconnect an MCP server.

        v22 FIX: Also unregisters all MCP tools from the tool REGISTRY
        so stale handlers don't remain after disconnection.
        """
        server = self._servers.get(mcp_id)
        if not server:
            raise ValueError(f"MCP server '{mcp_id}' not found")

        # Kill process if stdio
        if server._process:
            try:
                server._process.terminate()
                server._process = None
            except Exception:
                pass

        # Remove tools from internal MCP index
        to_remove = [name for name, tool in self._tools.items() if tool.server_id == mcp_id]
        for name in to_remove:
            del self._tools[name]

        # v22 FIX: Unregister all MCP tools from the tool REGISTRY
        unregistered = await self._unregister_server_tools_from_registry(mcp_id)

        server.connected = False
        server.status = "disconnected"
        await db.toggle_mcp_server(mcp_id, connected=False)
        await db.update_mcp_server_status(mcp_id, "disconnected")

        logger.info(f"MCP disconnected: {server.name} (unregistered {len(unregistered)} tools from REGISTRY)")
        await emit("mcp_disconnected", server_id=mcp_id, name=server.name)
        return {"ok": True, "server_id": mcp_id, "unregistered_tools": unregistered}

    async def call_tool(self, mcp_id: str, tool_name: str, arguments: dict | None = None) -> Any:
        """Call a tool on a connected MCP server."""
        server = self._servers.get(mcp_id)
        if not server or not server.connected:
            raise ConnectionError(f"MCP server '{mcp_id}' is not connected")

        if server.transport == "sse":
            return await self._call_tool_sse(server, tool_name, arguments)
        elif server.transport == "stdio":
            return await self._call_tool_stdio(server, tool_name, arguments)
        raise ValueError(f"Unknown transport: {server.transport}")

    async def _call_tool_sse(self, server: MCPServer, tool_name: str, arguments: dict | None) -> Any:
        """Call a tool on an SSE-based MCP server."""
        client = await self._get_client()
        base_url = server.url.rstrip("/")
        try:
            resp = await client.post(
                f"{base_url}/mcp/tools/call",
                headers={**server.headers, "Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0", "method": "tools/call", "id": 3,
                    "params": {"name": tool_name, "arguments": arguments or {}},
                },
            )
            if resp.status_code == 200:
                return resp.json()
            raise ConnectionError(f"Tool call failed: HTTP {resp.status_code}")
        except Exception as e:
            raise ConnectionError(f"Failed to call {tool_name}: {e}")

    async def _call_tool_stdio(self, server: MCPServer, tool_name: str, arguments: dict | None) -> Any:
        """Call a tool on a stdio-based MCP server.

        Uses proper Content-Length framing for all reads.  Drains any
        interstitial notifications before waiting for the actual response.
        """
        if not server._process or server._process.poll() is not None:
            raise ConnectionError("Server process is not running")

        msg = json.dumps({
            "jsonrpc": "2.0", "method": "tools/call", "id": 4,
            "params": {"name": tool_name, "arguments": arguments or {}},
        })
        server._process.stdin.write(f"Content-Length: {len(msg)}\r\n\r\n{msg}".encode())
        server._process.stdin.flush()

        # ── Drain notifications, then read the actual response ───────────────
        # Some servers emit progress notifications before the tool result.
        # We keep reading until we get a message with an "id" (the response)
        # or we time out.
        result = None
        deadline = time.time() + 60.0  # hard ceiling
        while time.time() < deadline:
            data = await asyncio.to_thread(_read_stdio_message, server._process, timeout=30.0)
            if data is None:
                break
            # Notifications have no "id" field; responses do.
            if "id" in data:
                if "error" in data:
                    result = {"error": data["error"]}
                else:
                    result = data.get("result", data)
                break
            # else: notification — discard and continue
            logger.debug("stdio MCP: ignored notification during tool call: %s",
                          data.get("method", "?"))

        return result or {"error": "No response from server"}

    async def test_connection(self, mcp_id: str) -> dict:
        """Test connection to an MCP server without persisting state."""
        server = self._servers.get(mcp_id)
        if not server:
            row = await db.get_mcp_server(mcp_id)
            if row:
                server = self._dict_to_server(row)
            else:
                return {"ok": False, "error": "Server not found"}
        try:
            if server.transport == "sse":
                tools = await self._connect_sse(server)
                return {"ok": True, "tools": tools, "transport": "sse"}
            elif server.transport == "stdio":
                tools = await self._connect_stdio(server)
                return {"ok": True, "tools": tools, "transport": "stdio"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": False, "error": "Unknown transport"}

    def list_servers(self) -> list[dict]:
        """Return all tracked servers."""
        return [s.to_dict() for s in self._servers.values()]

    def list_connected_servers(self) -> list[dict]:
        """Return only connected servers."""
        return [s.to_dict() for s in self._servers.values() if s.connected]

    def get_tools_for_prompt(self) -> str:
        """Generate MCP tools context for system prompt injection."""
        connected = [s for s in self._servers.values() if s.connected]
        if not connected:
            return ""
        lines = []
        for server in connected:
            tools_str = ", ".join(
                t.get("name", t) if isinstance(t, dict) else str(t)
                for t in server.tools
            ) if server.tools else "no tools discovered"
            lines.append(
                f"  - {server.name} ({server.id}): {server.description or 'MCP Server'} [{server.transport}]\n"
                f"    Tools: {tools_str}"
            )
        return (
            "# Connected MCP Servers\n"
            + "\n".join(lines)
            + "\nUse call_mcp_tool to invoke tools on these servers. "
            "MCP servers extend your capabilities with specialized tools.\n"
        )

    def get_all_tools(self) -> list[dict]:
        """Return all tools from all connected MCP servers (for tool registration)."""
        result = []
        for tool in self._tools.values():
            result.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "server_id": tool.server_id,
                "server_name": tool.server_name,
            })
        return result

    async def remove_server(self, mcp_id: str) -> dict:
        """Remove an MCP server completely.

        v22 FIX: Also unregisters all MCP tools from the tool REGISTRY.
        """
        unregistered: list[str] = []
        if mcp_id in self._servers:
            server = self._servers[mcp_id]
            if server._process:
                try:
                    server._process.terminate()
                except Exception:
                    pass
            # Remove tools from internal MCP index
            to_remove = [name for name, tool in self._tools.items() if tool.server_id == mcp_id]
            for name in to_remove:
                del self._tools[name]
            del self._servers[mcp_id]

        # v22 FIX: Unregister all MCP tools from the tool REGISTRY
        unregistered = await self._unregister_server_tools_from_registry(mcp_id)

        await db.delete_mcp_server(mcp_id)
        await emit("mcp_server_removed", server_id=mcp_id)
        logger.info(f"MCP server removed: {mcp_id} (unregistered {len(unregistered)} tools from REGISTRY)")
        return {"ok": True, "unregistered_tools": unregistered}

    async def _unregister_server_tools_from_registry(self, mcp_id: str) -> list[str]:
        """Unregister all REGISTRY tools that belong to a specific MCP server.

        v22 FIX: This ensures that when an MCP server is disconnected or removed,
        its tools are properly cleaned up from the REGISTRY so the agent doesn't
        try to call stale handlers on a disconnected server.

        Returns:
            List of tool names that were unregistered.
        """
        unregistered: list[str] = []
        tool_names = self._registry_tool_map.pop(mcp_id, set())
        if not tool_names:
            return unregistered
        try:
            from ..registry import REGISTRY
            for tool_name in tool_names:
                # Verify the tool still exists and is an MCP tool before removing
                existing = REGISTRY.get_tool(tool_name)
                if existing and existing.category == "MCP":
                    REGISTRY.unregister(tool_name)
                    unregistered.append(tool_name)
                    logger.info(f"Unregistered MCP tool from REGISTRY: {tool_name}")
                elif existing:
                    # Tool exists but is not MCP category — don't remove it
                    logger.debug(f"Skipping unregister of non-MCP tool: {tool_name}")
        except Exception as exc:
            logger.warning(f"Error unregistering MCP tools from REGISTRY: {exc}")
        return unregistered

    async def sync_registry(self) -> dict:
        """Synchronize the tool REGISTRY with the current MCP server state.

        v22: Ensures REGISTRY is consistent with which MCP servers are actually
        connected. Useful after a server restart or crash recovery.

        - Unregisters tools from servers that are no longer connected
        - Re-registers tools from connected servers if missing from REGISTRY
        - Returns a summary of changes made

        Returns:
            Dict with 'unregistered' and 'registered' lists.
        """
        result = {"unregistered": [], "registered": []}

        # Step 1: Unregister tools from servers that are no longer connected
        stale_servers = []
        for server_id in list(self._registry_tool_map.keys()):
            server = self._servers.get(server_id)
            if not server or not server.connected:
                unregistered = await self._unregister_server_tools_from_registry(server_id)
                result["unregistered"].extend(unregistered)
                if not server:
                    stale_servers.append(server_id)

        # Step 2: For connected servers, ensure their tools are in REGISTRY
        for server_id, server in self._servers.items():
            if not server.connected:
                continue
            # Check if this server's tools are tracked
            tracked = self._registry_tool_map.get(server_id, set())
            # Check MCP tools index for this server
            server_tools = [t for t in self._tools.values() if t.server_id == server_id]
            for mcp_tool in server_tools:
                reg_name = f"mcp_{mcp_tool.name}"
                if reg_name not in tracked:
                    # Tool is in MCP index but not tracked in registry map
                    # Check if it's actually in REGISTRY
                    try:
                        from ..registry import REGISTRY, Tool as RegistryTool
                        if not REGISTRY.get_tool(reg_name):
                            # Register it
                            _sid = server.id
                            _orig_name = mcp_tool.name

                            async def _make_handler(sid=_sid, orig=_orig_name):
                                async def _handler(params: dict) -> str:
                                    try:
                                        r = await self.call_tool(sid, orig, params)
                                        if isinstance(r, dict):
                                            return json.dumps(r, default=str)[:8000]
                                        return str(r)[:8000]
                                    except Exception as exc:
                                        return f"MCP tool error: {exc}"
                                return _handler

                            handler = await _make_handler()
                            new_tool = RegistryTool(
                                name=reg_name,
                                description=f"[MCP:{server.name}] {mcp_tool.description}",
                                parameters_schema=mcp_tool.input_schema if mcp_tool.input_schema else {"type": "object", "properties": {}},
                                handler=handler,
                                category="MCP",
                                risk="low",
                                tags=["mcp", server.name],
                            )
                            REGISTRY.register(new_tool)
                            self._registry_tool_map.setdefault(server_id, set()).add(reg_name)
                            result["registered"].append(reg_name)
                            logger.info(f"Sync: registered MCP tool in REGISTRY: {reg_name}")
                    except Exception as exc:
                        logger.warning(f"Sync: failed to register {reg_name}: {exc}")

        return result

    def get_registry_tool_names(self, mcp_id: str) -> list[str]:
        """Return the list of REGISTRY tool names for a given MCP server.

        v22: Useful for debugging and for the reasoning engine to know
        which REGISTRY tools are MCP-sourced.
        """
        return sorted(self._registry_tool_map.get(mcp_id, set()))

    def get_all_registry_mcp_tools(self) -> dict[str, list[str]]:
        """Return all MCP server IDs and their REGISTRY tool names.

        v22: Useful for the reasoning engine to inject MCP tool context.
        """
        return {sid: sorted(names) for sid, names in self._registry_tool_map.items() if names}

    async def auto_discover_and_connect(self, description: str) -> dict:
        """
        Auto-discover and connect to an MCP server based on a user description.
        Used for non-technical users — the agent figures out the connection.
        """
        # Common MCP server patterns
        KNOWN_SERVERS = {
            "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem"], "transport": "stdio"},
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "transport": "stdio"},
            "postgres": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres"], "transport": "stdio"},
            "sqlite": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-sqlite"], "transport": "stdio"},
            "brave-search": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-brave-search"], "transport": "stdio"},
            "google-maps": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-google-maps"], "transport": "stdio"},
            "memory": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-memory"], "transport": "stdio"},
            "puppeteer": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-puppeteer"], "transport": "stdio"},
            "slack": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-slack"], "transport": "stdio"},
            "sentry": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-sentry"], "transport": "stdio"},
        }

        low = description.lower()
        for key, server_info in KNOWN_SERVERS.items():
            if key in low or key.replace("-", " ") in low:
                mcp_id = f"mcp_auto_{key}"
                name = f"Auto: {key.replace('-', ' ').title()}"
                server = await self.add_server(
                    mcp_id=mcp_id, name=name,
                    transport=server_info["transport"],
                    command=server_info["command"],
                    args=server_info["args"],
                    description=f"Auto-discovered MCP server for {key}",
                    category="auto-discovered",
                    auto_connect=True,
                )
                try:
                    result = await self.connect(mcp_id)
                    return {"ok": True, "server": server.to_dict(), "connected": True, "tools": result.get("tools", 0)}
                except Exception as e:
                    return {"ok": True, "server": server.to_dict(), "connected": False, "error": str(e)}

        # If no known pattern matches, create a custom SSE server entry
        mcp_id = f"mcp_custom_{uuid.uuid4().hex[:8]}"
        name = f"Custom: {description[:40]}"
        return {
            "ok": False,
            "message": f"No known MCP server matched '{description}'. Please provide a server URL (SSE) or command (stdio).",
            "suggested_id": mcp_id,
            "suggested_name": name,
        }

    async def shutdown(self):
        """Clean up all resources: close HTTP client, terminate stdio processes."""
        for sid, srv in list(self._servers.items()):
            try:
                if srv.transport == "stdio" and srv._process:
                    try:
                        srv._process.terminate()
                        await asyncio.wait_for(asyncio.to_thread(srv._process.wait), timeout=5)
                    except Exception:
                        try:
                            srv._process.kill()
                        except Exception:
                            pass
            except Exception:
                pass
        self._servers.clear()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


# Singleton instance
MCP_MANAGER = MCPManager()
