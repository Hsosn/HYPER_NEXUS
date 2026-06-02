"""
MCP Tools — Agent-callable tools for MCP server management.
These allow the agent to manage MCP servers on behalf of non-technical users.
"""
from __future__ import annotations

import json
import logging
from ...tools.registry import tool
from .mcp_manager import MCP_MANAGER

logger = logging.getLogger("nexus.mcp_tools")


@tool(
    name="mcp_connect_server",
    description="Connect to an MCP (Model Context Protocol) server. Use this to connect to external tool servers that extend the agent's capabilities. Supports stdio and SSE transport.",
    category="MCP",
    risk="high",
    parameters_schema={
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": "For SSE transport: the base URL of the MCP server (e.g. http://localhost:3000). For stdio: leave empty.",
            },
            "command": {
                "type": "string",
                "description": "For stdio transport: the command to launch the MCP server (e.g. 'npx'). Leave empty for SSE.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments for the stdio command (e.g. ['-y', '@anthropic/mcp-server-filesystem']).",
            },
            "env": {
                "type": "object",
                "description": "Environment variables for the stdio process (e.g. {'API_KEY': 'xxx'}). Sensitive values are masked.",
            },
            "name": {
                "type": "string",
                "description": "A friendly name for this MCP server.",
            },
            "transport": {
                "type": "string",
                "enum": ["stdio", "sse"],
                "description": "Transport type: 'stdio' for local processes, 'sse' for remote HTTP servers. Default: auto-detect.",
            },
            "description": {
                "type": "string",
                "description": "Description of what this MCP server provides.",
            },
            "auto_connect": {
                "type": "boolean",
                "description": "Whether to auto-connect on startup. Default: true.",
            },
        },
    },
)
async def mcp_connect_server(params: dict) -> str:
    """Connect to an MCP server."""
    server_url = params.get("server_url", "")
    command = params.get("command", "")
    args = params.get("args", [])
    env = params.get("env", {})
    name = params.get("name", "")
    description = params.get("description", "")
    auto_connect = params.get("auto_connect", True)

    # Auto-detect transport
    transport = params.get("transport", "")
    if not transport:
        transport = "sse" if server_url else "stdio"

    server = await MCP_MANAGER.add_server(
        name=name or f"MCP Server",
        transport=transport,
        command=command,
        args=args,
        env=env,
        url=server_url,
        description=description,
        auto_connect=auto_connect,
    )
    try:
        result = await MCP_MANAGER.connect(server.id)
        return json.dumps({"success": True, "server_id": server.id, "server_name": server.name, **result})
    except Exception as e:
        # Rollback: remove the server entry if connect failed
        try:
            await MCP_MANAGER.remove_server(server.id)
        except Exception:
            pass
        return json.dumps({"success": False, "server_id": server.id, "error": str(e)})


@tool(
    name="mcp_disconnect_server",
    description="Disconnect from a connected MCP server.",
    category="MCP",
    parameters_schema={
        "type": "object",
        "properties": {
            "server_id": {
                "type": "string",
                "description": "The ID of the MCP server to disconnect.",
            },
        },
        "required": ["server_id"],
    },
)
async def mcp_disconnect_server(params: dict) -> str:
    """Disconnect an MCP server."""
    server_id = params["server_id"]
    try:
        result = await MCP_MANAGER.disconnect(server_id)
        return json.dumps({"success": True, **result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool(
    name="mcp_list_servers",
    description="List all configured MCP servers and their connection status.",
    category="MCP",
    parameters_schema={"type": "object", "properties": {}},
)
async def mcp_list_servers(params: dict) -> str:
    """List all MCP servers."""
    return json.dumps({"servers": MCP_MANAGER.list_servers()})


@tool(
    name="mcp_call_tool",
    description="Call a tool on a connected MCP server. Use this to invoke any tool exposed by a connected MCP server.",
    category="MCP",
    parameters_schema={
        "type": "object",
        "properties": {
            "server_id": {
                "type": "string",
                "description": "The ID of the MCP server that exposes the tool.",
            },
            "tool_name": {
                "type": "string",
                "description": "The name of the tool to call.",
            },
            "arguments": {
                "type": "object",
                "description": "The arguments to pass to the tool.",
            },
        },
        "required": ["server_id", "tool_name"],
    },
)
async def mcp_call_tool(params: dict) -> str:
    """Call a tool on an MCP server."""
    server_id = params["server_id"]
    tool_name = params["tool_name"]
    arguments = params.get("arguments", {})

    # FIX: Validate that the specific server_id is connected before calling.
    connected_servers = MCP_MANAGER.list_connected_servers()
    connected_ids = {s.get("id") for s in connected_servers}

    if server_id not in connected_ids:
        all_servers = MCP_MANAGER.list_servers()
        all_ids = {s.get("id") for s in all_servers}
        if not all_servers:
            return json.dumps({
                "success": False,
                "error": (
                    "No MCP servers are configured. "
                    "Use mcp_connect_server or mcp_auto_connect to add one. "
                    "Examples: 'filesystem', 'github', 'postgres', 'brave search'."
                ),
                "hint": "Call mcp_auto_connect with a description like 'filesystem' to get started.",
            })
        elif server_id not in all_ids:
            return json.dumps({
                "success": False,
                "error": f"MCP server '{server_id}' does not exist. Available servers: {sorted(all_ids)}",
            })
        else:
            return json.dumps({
                "success": False,
                "error": (
                    f"MCP server '{server_id}' is not connected. "
                    f"Use mcp_connect_server to reconnect it."
                ),
                "disconnected_servers": [s.get("name", s.get("id")) for s in all_servers if s.get("id") not in connected_ids],
            })

    try:
        result = await MCP_MANAGER.call_tool(server_id, tool_name, arguments)
        return json.dumps({"success": True, "result": result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool(
    name="mcp_auto_connect",
    description="Auto-discover and connect to a known MCP server by description. Use this for non-technical users who want to connect to a service. The agent will figure out the correct MCP server package and connection parameters. Examples: 'filesystem', 'github', 'postgres', 'brave search', 'google maps', 'memory', 'puppeteer', 'slack'.",
    category="MCP",
    parameters_schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of the MCP server to auto-discover (e.g. 'github', 'filesystem', 'postgres database').",
            },
        },
        "required": ["description"],
    },
)
async def mcp_auto_connect(params: dict) -> str:
    """Auto-discover and connect to an MCP server."""
    description = params["description"]
    result = await MCP_MANAGER.auto_discover_and_connect(description)
    return json.dumps(result) if isinstance(result, dict) else str(result)


@tool(
    name="mcp_remove_server",
    description="Remove an MCP server entirely from the configuration.",
    category="MCP",
    parameters_schema={
        "type": "object",
        "properties": {
            "server_id": {
                "type": "string",
                "description": "The ID of the MCP server to remove.",
            },
        },
        "required": ["server_id"],
    },
)
async def mcp_remove_server(params: dict) -> str:
    """Remove an MCP server."""
    server_id = params["server_id"]
    try:
        result = await MCP_MANAGER.remove_server(server_id)
        return json.dumps({"success": True, **result})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
