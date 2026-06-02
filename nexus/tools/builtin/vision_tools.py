"""Vision Tools — on-device image analysis using the local Florence-2 engine.

Provides the agent with the ability to independently analyze image files using
a local vision model (Microsoft Florence-2) that runs entirely on-device with
no external API calls, no third-party services, and no cloud dependency.

Registered tools:
- image_understand: Analyze an image with a text prompt via local vision model
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import tool
from ...config import BASE_DIR

_VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif", ".ico"}


def _resolve_path(file_path: str) -> Path:
    """Resolve an image file path relative to the workspace or project root.

    Args:
        file_path: Absolute path, or path relative to workspace/project root.

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If the file doesn't exist at any tried location.
    """
    path = Path(file_path)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(
                f"Image file not found: {path.resolve()}"
            )
        return path

    # Resolve relative to workspace
    workspace_candidate = BASE_DIR / "data" / "workspace" / file_path
    if workspace_candidate.exists():
        return workspace_candidate

    # Resolve relative to project root
    project_candidate = BASE_DIR / file_path
    if project_candidate.exists():
        return project_candidate

    raise FileNotFoundError(
        f"Image file not found: {file_path}\n"
        f"  Tried:\n"
        f"    {workspace_candidate.resolve()}\n"
        f"    {project_candidate.resolve()}\n\n"
        f"Provide an absolute path like `C:/path/to/image.jpg` or a "
        f"path relative to the workspace directory."
    )


@tool(
    name="image_understand",
    description=(
        "Analyze an image using a local on-device vision model (Florence-2). "
        "Reads an image file from disk and runs it through the local vision "
        "engine for detailed analysis — no API calls, no cloud, 100% offline. "
        "Can describe objects, people, text, scenes, colors, diagrams, charts, "
        "handwritten notes, UI layouts, and more. Provide a specific prompt "
        "to focus the analysis."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": (
                    "Path to the image file (PNG, JPG, JPEG, WebP, BMP). "
                    "Can be absolute, or relative to the workspace directory."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Specific question or instruction about the image. "
                    "Examples:\n"
                    "- 'What text is in this image?'\n"
                    "- 'Describe the data shown in this chart'\n"
                    "- 'Read the handwritten note'\n"
                    "- 'Identify all UI elements and their positions'\n"
                    "- 'Describe this scene in detail'\n"
                    "Default: a general description prompt."
                ),
            },
        },
        "required": ["file_path"],
    },
    category="vision",
    risk="low",
    timeout=120,
    tags=["vision", "image", "analyze", "understand", "describe", "ocr", "caption", "local"],
)
async def image_understand(params: dict) -> str:
    """Analyze an image using the local Florence-2 on-device vision model.

    The handler resolves the file path to an absolute location, then delegates
    to ``vision_local.analyze_image()`` which loads the image, runs it through
    the local Florence-2 model, and returns a text description.

    No external API calls are made — the model runs entirely on-device using
    PyTorch and HuggingFace Transformers (downloaded once on first use, cached
    locally thereafter, ~900MB disk space).

    Returns the model's analysis as a plain-text response.
    """
    # Accept multiple param aliases — the agent often uses different names
    file_path: str = (
        params.get("file_path")
        or params.get("path")
        or params.get("image")
        or params.get("image_path")
        or params.get("file")
        or ""
    )
    user_prompt: str = (
        params.get("prompt")
        or params.get("question")
        or params.get("query")
        or "Describe this image in detail."
    )

    if not file_path:
        return (
            "Error: 'file_path' (or 'path', 'image', 'image_path', 'file') "
            "is required. Provide the path to an image file on disk."
        )

    try:
        resolved = _resolve_path(file_path)
    except FileNotFoundError as e:
        return f"Error: {e}"

    ext = resolved.suffix.lower()
    if ext not in _VALID_IMAGE_EXTENSIONS:
        return (
            f"Error: Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(_VALID_IMAGE_EXTENSIONS))}."
        )

    try:
        size = resolved.stat().st_size
        if size > 20 * 1024 * 1024:
            return (
                f"Error: Image too large: {size / 1024 / 1024:.1f}MB "
                f"(max 20MB). Resize or compress the image first."
            )
    except OSError as e:
        return f"Error accessing file '{file_path}': {e}"

    try:
        from ...core.vision_local import analyze_image as local_analyze

        result = await local_analyze(str(resolved), prompt=user_prompt)
        return result
    except ImportError as e:
        return (
            f"Error: Local vision engine requires additional packages.\n"
            f"  {e}\n\n"
            f"Install missing packages with:\n"
            f"  pip install transformers einops timm accelerate"
        )
    except Exception as e:
        return (
            f"Error analyzing image with local vision model: "
            f"{type(e).__name__}: {e}"
        )
