"""Tools for the agent to manage its own goals and tasks."""
from __future__ import annotations

from ..registry import tool
from ...memory import db


@tool(
    name="add_goal",
    description="Add a new goal the agent will pursue.",
    parameters_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["title"],
    },
    category="planning",
)
async def add_goal(params):
    gid = await db.add_goal(
        title=params["title"],
        description=params.get("description", ""),
        priority=int(params.get("priority", 5)),
    )
    return f"Goal #{gid} added"


@tool(
    name="list_goals",
    description="List all goals (optionally filter by status).",
    parameters_schema={
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["pending", "active", "done", "failed", "cancelled"]}},
        "required": [],
    },
    category="planning",
)
async def list_goals_tool(params):
    goals = await db.list_goals(status=params.get("status"))
    if not goals:
        return "(no goals)"
    return "\n".join(
        f"[#{g['id']} {g['status']} p{g['priority']}] {g['title']} ({(g['progress'] or 0)*100:.0f}%)"
        for g in goals
    )


@tool(
    name="update_goal_progress",
    description="Update the progress (0..1) and optionally status of a goal.",
    parameters_schema={
        "type": "object",
        "properties": {
            "goal_id": {"type": "integer"},
            "progress": {"type": "number", "minimum": 0, "maximum": 1},
            "status": {"type": "string", "enum": ["pending", "active", "done", "failed", "cancelled"]},
        },
        "required": ["goal_id"],
    },
    category="planning",
)
async def update_goal_progress(params):
    fields = {}
    if "progress" in params:
        fields["progress"] = float(params["progress"])
    if "status" in params:
        fields["status"] = params["status"]
    if not fields:
        return f"Goal {params['goal_id']}: no changes specified"
    await db.update_goal(int(params["goal_id"]), **fields)
    return f"Goal {params['goal_id']} updated"
