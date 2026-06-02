"""

Vision Loop — Autonomous Computer-Use Agent Engine.



Implements the core perception-action loop that enables NEXUS to autonomously

interact with a graphical desktop environment:



    screenshot → VISION analysis → action decision → action execution → repeat



The loop runs until the VISION determines the task is complete, an error occurs,

or the maximum iteration count is reached.

"""

from __future__ import annotations



import asyncio

import base64

import json

import logging

import re

import time

from typing import Any



from ..registry import tool

from ...events import emit

from ...virtual_computer.container import get_vm



logger = logging.getLogger("nexus.vision_loop")



# ---------------------------------------------------------------------------

# Constants

# ---------------------------------------------------------------------------



# System prompt that instructs the VISION to respond with structured JSON actions

_VISION_SYSTEM_PROMPT = """\

You are an AI computer-use agent controlling a Ubuntu 22.04 virtual desktop (1280x720).

You can see the screen via screenshots and interact using mouse and keyboard.



## Available Actions

You MUST respond with a single JSON object specifying the next action. Choose from:



1. **mousemove** — Move the mouse to coordinates

   {"action": "mousemove", "x": 640, "y": 360, "reason": "Move to button"}



2. **click** — Click the mouse (optionally after moving)

   {"action": "click", "x": 100, "y": 200, "button": "left", "reason": "Click button"}

   {"action": "click", "button": "left", "reason": "Click at current position"}



3. **type** — Type text at the current cursor position

   {"action": "type", "text": "hello world", "reason": "Type in search box"}



4. **key** — Press a key or key combination

   {"action": "key", "key": "Return", "reason": "Submit form"}

   {"action": "key", "key": "ctrl+c", "reason": "Copy text"}

   {"action": "key", "key": "alt+Tab", "reason": "Switch window"}



5. **execute** — Run a shell command inside the container

   {"action": "execute", "command": "ls -la /workspace", "reason": "List files"}



6. **done** — Task is complete

   {"action": "done", "reason": "Task completed successfully"}



## Important Rules

- Respond ONLY with the JSON object, no other text.

- The screen is 1280x720 pixels (0,0 is top-left).

- Use precise coordinates — estimate element positions from the screenshot.

- Think step by step about what you see and what needs to be done.

- If the task requires multiple steps, break it down and do one step at a time.

- If something goes wrong, try a different approach rather than giving up.

- Use the "done" action ONLY when the task is fully complete.

- Always include a brief "reason" explaining your decision.

- Common coordinates: taskbar is at the bottom (~y=700), top menu bar is at y=0-30.

- Wait briefly between actions by using small mouse movements if needed.

"""



# JSON extraction patterns

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_JSON_DIRECT_RE = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)





def _extract_json(text: str) -> dict[str, Any] | None:

    """Extract a JSON object from potentially messy LLM output."""

    # Try code block first

    m = _JSON_BLOCK_RE.search(text)

    if m:

        try:

            return json.loads(m.group(1))

        except json.JSONDecodeError:

            pass



    # Try direct JSON match

    m = _JSON_DIRECT_RE.search(text)

    if m:

        try:

            return json.loads(m.group(1))

        except json.JSONDecodeError:

            pass



    # Try parsing the whole thing

    try:

        cleaned = text.strip()

        if cleaned.startswith("{"):

            return json.loads(cleaned)

    except json.JSONDecodeError:

        pass



    return None





# ---------------------------------------------------------------------------

# Vision Loop Core

# ---------------------------------------------------------------------------



async def run_vision_loop(

    task: str,

    max_iterations: int = 20,

    model: str | None = None,

) -> dict[str, Any]:

    """Execute an autonomous vision loop to complete a GUI task.



    The loop:

    1. Takes a screenshot of the virtual display

    2. Sends it to a vision language model (VISION) with the task description

    3. Parses the VISION's response to determine the next action

    4. Executes the action (mouse, keyboard, command, etc.)

    5. Repeats until the task is done or max iterations reached



    Args:

        task: Natural language description of what to accomplish on the desktop.

        max_iterations: Maximum number of screenshot-action cycles (default 20).

        model: Optional model override for the VISION. If None, uses the default model.



    Returns:

        A dict with keys: success, iterations, actions_taken, final_message, error.

    """

    vm = get_vm()



    # Validate VM is running

    status = await vm.status()

    if status.get("status") != "running":

        return {

            "success": False,

            "iterations": 0,

            "actions_taken": [],

            "final_message": f"Virtual computer is not running. Status: {status.get('status')}",

            "error": "vm_not_running",

        }



    await emit("vision_loop_start", task=task, max_iterations=max_iterations)

    logger.info("Starting vision loop: task='%s', max_iterations=%d", task, max_iterations)



    actions_taken: list[dict[str, Any]] = []

    start_time = time.time()

    last_reason = ""

    # FIX: Track consecutive VISION parse failures for exponential backoff

    _consecutive_parse_failures = 0

    # FIX: Track recent actions to detect duplicate action loops

    _recent_actions: list[str] = []



    for iteration in range(1, max_iterations + 1):

        await emit(

            "vision_loop_iteration",

            iteration=iteration,

            max_iterations=max_iterations,

            task=task,

        )

        logger.info("Vision loop iteration %d/%d", iteration, max_iterations)



        # --- Step 1: Take screenshot ---

        try:

            png_bytes = await vm.screenshot()

            if not png_bytes:

                actions_taken.append({"iteration": iteration, "error": "screenshot_failed"})

                await emit("vision_loop_error", iteration=iteration, error="Screenshot returned empty data")

                await asyncio.sleep(2)

                continue

            screenshot_b64 = base64.b64encode(png_bytes).decode("utf-8")

        except Exception as e:

            actions_taken.append({"iteration": iteration, "error": str(e)})

            await emit("vision_loop_error", iteration=iteration, error=str(e))

            logger.error("Screenshot failed in iteration %d: %s", iteration, e)

            await asyncio.sleep(2)

            continue



        # --- Step 2: Send to VISION ---

        try:

            action_data = await _query_vision(task, screenshot_b64, iteration, model)

        except Exception as e:

            actions_taken.append({"iteration": iteration, "error": f"VISION error: {e}"})

            await emit("vision_loop_error", iteration=iteration, error=f"VISION query failed: {e}")

            logger.error("VISION query failed in iteration %d: %s", iteration, e)

            await asyncio.sleep(3)

            continue



        if action_data is None:

            _consecutive_parse_failures += 1

            actions_taken.append({"iteration": iteration, "error": "Failed to parse VISION response"})

            await emit("vision_loop_error", iteration=iteration, error="Could not parse VISION response as JSON")

            # FIX: Exponential backoff on parse failures instead of burning through iterations

            _backoff = min(2 ** _consecutive_parse_failures, 16)  # 2, 4, 8, 16s max

            logger.warning("VISION parse failure #%d, backing off %ds", _consecutive_parse_failures, _backoff)

            # After 5 consecutive parse failures, abort — something is fundamentally wrong

            if _consecutive_parse_failures >= 5:

                await emit("vision_loop_end", success=False, iterations=iteration,

                           total_time=time.time() - start_time, reason="consecutive_parse_failures")

                return {

                    "success": False,

                    "iterations": iteration,

                    "actions_taken": actions_taken,

                    "final_message": f"Aborted after {_consecutive_parse_failures} consecutive VISION parse failures. The vision model may not be producing valid JSON.",

                    "total_time": round(time.time() - start_time, 2),

                }

            await asyncio.sleep(_backoff)

            continue



        # FIX: Reset parse failure counter on successful parse

        _consecutive_parse_failures = 0



        # --- Step 3: Parse action ---

        action = action_data.get("action", "").lower().strip()

        reason = action_data.get("reason", "")

        last_reason = reason



        action_record: dict[str, Any] = {

            "iteration": iteration,

            "action": action,

            "reason": reason,

            "raw": {k: v for k, v in action_data.items() if k != "reason"},

        }



        # FIX: Duplicate action detection — if the same action with same params

        # is repeated 3+ times in a row, the VISION is stuck in a loop.

        if action != "done":

            _action_sig = f"{action}:{action_data.get('x', '')}:{action_data.get('y', '')}:{action_data.get('text', '')}:{action_data.get('key', '')}:{action_data.get('command', '')}"

            _recent_actions.append(_action_sig)

            if len(_recent_actions) > 5:

                _recent_actions.pop(0)

            # Check if the last 3 actions are identical

            if len(_recent_actions) >= 3 and len(set(_recent_actions[-3:])) == 1:

                await emit("vision_loop_end", success=False, iterations=iteration,

                           total_time=time.time() - start_time, reason="duplicate_action_loop")

                action_record["result"] = "Aborted: same action repeated 3 times"

                actions_taken.append(action_record)

                return {

                    "success": False,

                    "iterations": iteration,

                    "actions_taken": actions_taken,

                    "final_message": f"Aborted: the same action '{action}' was repeated 3 times consecutively. The VISION appears stuck. Try a different approach or increase temperature.",

                    "total_time": round(time.time() - start_time, 2),

                }

        else:

            _recent_actions.clear()  # Reset on any action type



        await emit(

            "vision_loop_action",

            iteration=iteration,

            action=action,

            reason=reason,

            details=action_data,

        )

        logger.info("Iteration %d: action=%s, reason=%s", iteration, action, reason)



        # --- Step 4: Check for done ---

        if action == "done":

            action_record["result"] = "Task completed"

            actions_taken.append(action_record)

            await emit(

                "vision_loop_end",

                success=True,

                iterations=iteration,

                total_time=time.time() - start_time,

                reason=reason,

            )

            logger.info("Vision loop completed after %d iterations: %s", iteration, reason)

            return {

                "success": True,

                "iterations": iteration,

                "actions_taken": actions_taken,

                "final_message": f"Task completed: {reason}",

                "total_time": round(time.time() - start_time, 2),

            }



        # --- Step 5: Execute action ---

        try:

            result = await _execute_action(vm, action_data)

            action_record["result"] = result

        except Exception as e:

            action_record["result"] = f"Error: {e}"

            logger.error("Action execution failed: %s", e)



        actions_taken.append(action_record)



        # Brief pause between actions to let the UI settle

        await asyncio.sleep(1.5)



    # Max iterations reached

    await emit(

        "vision_loop_end",

        success=False,

        iterations=max_iterations,

        total_time=time.time() - start_time,

        reason="max_iterations_reached",

    )

    logger.warning("Vision loop reached max iterations (%d)", max_iterations)



    return {

        "success": False,

        "iterations": max_iterations,

        "actions_taken": actions_taken,

        "final_message": f"Reached maximum iterations ({max_iterations}). Last action: {last_reason}",

        "total_time": round(time.time() - start_time, 2),

    }





# Known vision-capable model patterns

_VISION_MODEL_PATTERNS = [

    "claude-3", "claude-sonnet", "claude-opus",  # Anthropic Claude 3+

    "gpt-4o", "gpt-4-turbo", "gpt-4-vision",     # OpenAI GPT-4 Vision

    "gemini", "gemini-2",                           # Google Gemini

    "llama-4", "llama4", "scout",                   # Llama vision models

    "llava", "qwen-vl", "qwen2.5-vl", "cogvlm",    # Open-source vision models

    "vision", "glm4v", "glm-4v",                    # Generic vision suffix + GLM vision

]





async def _select_vision_model() -> str | None:

    """Select the best available vision-capable model from configuration.

    

    Checks in order:

    1. Explicit 'vision_model' setting

    2. Whether 'default_model' is vision-capable

    3. Whether 'planner_model' is vision-capable

    4. Returns None if no vision model found

    """

    from ... import config as _cfg



    # 1. Explicit setting

    vision_model = _cfg.get("vision_model", "")

    if vision_model:

        return vision_model



    # 2. Check default model

    default = _cfg.get("default_model", "")

    if default and _is_vision_model(default):

        return default



    # 3. Check planner model

    planner = _cfg.get("planner_model", "")

    if planner and _is_vision_model(planner):

        return planner



    # 4. Try to find any vision model from the model list

    try:

        from ...core.llm import list_models

        models = await list_models()

        for m in models:

            model_id = m.get("id", "") if isinstance(m, dict) else str(m)

            if _is_vision_model(model_id):

                return model_id

    except Exception:

        pass



    return None





def _is_vision_model(model_name: str) -> bool:

    """Check if a model name indicates vision capability."""

    if not model_name:

        return False

    lower = model_name.lower()

    return any(pat in lower for pat in _VISION_MODEL_PATTERNS)





async def _query_vision(

    task: str,

    screenshot_b64: str,

    iteration: int,

    model: str | None = None,

) -> dict[str, Any] | None:

    """Send a screenshot to the VISION and parse its action response."""

    from ...core.llm import complete



    # Build messages with vision content

    messages = [

        {

            "role": "system",

            "content": _VISION_SYSTEM_PROMPT,

        },

        {

            "role": "user",

            "content": [

                {

                    "type": "text",

                    "text": (

                        f"Task: {task}\n\n"

                        f"Iteration: {iteration}\n\n"

                        f"Look at the screenshot and decide the next action to take. "

                        f"Respond with ONLY a JSON object."

                    ),

                },

                {

                    "type": "image_url",

                    "image_url": {

                        "url": f"data:image/png;base64,{screenshot_b64}",

                    },

                },

            ],

        },

    ]



    # Select a vision-capable model

    if model is None:

        model = await _select_vision_model()

        if model is None:

            raise RuntimeError(

                "No vision-capable model configured. Set 'vision_model' in settings "

                "to a model that supports image inputs (e.g. anthropic/claude-sonnet-4, "

                "openai/gpt-4o, google/gemini-2.0-flash-exp). "

                "Alternatively, ensure 'default_model' supports vision."

            )



    response = await complete(

        prompt="",

        model=model,

        messages=messages,

        temperature=0.2,

        max_tokens=500,

    )



    content = response.content.strip()

    if not content:

        return None



    return _extract_json(content)





async def _execute_action(

    vm: Any,

    action_data: dict[str, Any],

) -> str:

    """Execute a parsed action on the virtual computer."""

    action = action_data.get("action", "").lower().strip()



    if action == "mousemove":

        x = int(action_data.get("x", 0))

        y = int(action_data.get("y", 0))

        x = max(0, min(1279, x))

        y = max(0, min(719, y))

        return await vm.mouse_move(x, y)



    elif action == "click":

        x = action_data.get("x")

        y = action_data.get("y")

        button_str = str(action_data.get("button", "left"))

        button_map = {"left": 1, "middle": 2, "right": 3}

        button = button_map.get(button_str, 1)

        if x is not None:

            x = max(0, min(1279, int(x)))

        if y is not None:

            y = max(0, min(719, int(y)))

        return await vm.mouse_click(x=x, y=y, button=button)



    elif action == "type":

        text = str(action_data.get("text", ""))

        if not text:

            return "Error: No text to type"

        return await vm.type_text(text)



    elif action == "key":

        key = str(action_data.get("key", ""))

        if not key:

            return "Error: No key specified"

        return await vm.press_key(key)



    elif action == "execute":

        command = str(action_data.get("command", ""))

        if not command:

            return "Error: No command specified"

        return await vm.exec_command(command, timeout=30)



    else:

        return f"Unknown action: {action}"





# ---------------------------------------------------------------------------

# Tool registration: vm_vision_loop

# ---------------------------------------------------------------------------



@tool(

    name="vm_vision_loop",

    description=(

        "Start an autonomous vision loop to complete a GUI task on the virtual computer. "

        "The agent will take screenshots, analyze them with a vision language model, "

        "and autonomously click, type, and navigate the desktop to accomplish the task. "

        "The virtual computer must be running (use vm_start first). "

        "This is the primary tool for GUI automation tasks like opening apps, browsing "

        "websites, editing documents, etc."

    ),

    parameters_schema={

        "type": "object",

        "properties": {

            "task": {

                "type": "string",

                "description": (

                    "Natural language description of the GUI task to accomplish. "

                    "Be specific about what you want done. "

                    "Example: 'Open Firefox and search for \"python tutorials\"'"

                ),

            },

            "max_iterations": {

                "type": "integer",

                "description": "Maximum number of screenshot-action cycles (default: 20)",

                "default": 20,

            },

        },

        "required": ["task"],

    },

    category="virtual_computer",

    risk="high",

    timeout=600,

    tags=["autonomous", "gui", "vision", "computer-use"],

)

async def vm_vision_loop(params: dict) -> str:

    """Tool entry point for the vision loop."""

    task = params.get("task", "").strip()

    if not task:

        return "Error: No task specified. Provide a description of what to accomplish on the desktop."



    max_iterations = int(params.get("max_iterations", 20))

    max_iterations = max(1, min(100, max_iterations))



    try:
        result = await run_vision_loop(task=task, max_iterations=max_iterations)
    except Exception as e:
        return f"[VM Vision Error] {type(e).__name__}: {e}"

    # Format output for the agent

    output_parts = [

        f"Vision Loop Result: {'SUCCESS' if result['success'] else 'INCOMPLETE'}",

        f"Iterations: {result['iterations']}",

        f"Total Time: {result.get('total_time', 'N/A')}s",

        f"Message: {result['final_message']}",

        "",

        "Actions taken:",

    ]



    for action in result.get("actions_taken", []):

        iter_num = action.get("iteration", "?")

        act = action.get("action", "?")

        reason = action.get("reason", "")

        res = action.get("result", "")

        output_parts.append(f"  [{iter_num}] {act}: {reason}")

        if res and res != f"Task completed":

            # Truncate long results

            res_short = res[:200] + "..." if len(res) > 200 else res

            output_parts.append(f"      → {res_short}")



    return "\n".join(output_parts)

