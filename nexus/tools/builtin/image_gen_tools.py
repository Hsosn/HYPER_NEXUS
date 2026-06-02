"""
Local AI Image Generation Tool — generates images using a lightweight
HuggingFace Diffusers model (OFA-Sys/small-stable-diffusion-v0) running
entirely on the local machine with zero API calls.

The model is downloaded once on first use and cached locally in
~/.cache/huggingface/.  Subsequent invocations load from cache.

Supports:
- Fully offline image generation (no API keys, no third-party services)
- GPU acceleration if CUDA is available; otherwise runs on CPU
- Saves generated images to workspace as PNG files
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from ...config import BASE_DIR
from ...events import emit
from ..registry import tool

# ══════════════════════════════════════════════════════════════════════════════
# Model configuration
# ══════════════════════════════════════════════════════════════════════════════

# Default model — lightweight (~306M params), designed for edge/CPU inference.
# Change this constant to swap to any HuggingFace StableDiffusion model ID
# (e.g. "runwayml/stable-diffusion-v1-5", "segmind/tiny-sd", etc.)
_MODEL_ID = "OFA-Sys/small-stable-diffusion-v0"

# Inference defaults (CPU-friendly: DPMSolverMultistepScheduler does well with few steps)
_DEFAULT_STEPS = 8       # quality/speed trade-off (4-10 recommended on CPU)
_DEFAULT_SIZE = 384      # CPU-friendly default (512 available via size param)
_VALID_SIZES = [256, 384, 512, 640, 768]

# ── Lazy-loaded pipeline singleton ─────────────────────────────────────────
_pipeline: Any = None      # StableDiffusionPipeline
_pipeline_lock = asyncio.Lock()
_pipeline_thread_lock = threading.Lock()  # guards concurrent pipe.__call__
_pipeline_model_id: str | None = None


async def _get_pipeline(model_id: str = _MODEL_ID) -> Any:
    """Return a cached StableDiffusionPipeline, loading it on first call.

    The pipeline is created in a thread (since model loading is CPU-bound)
    and reused for all subsequent generation calls.
    """
    global _pipeline, _pipeline_model_id

    if _pipeline is not None and _pipeline_model_id == model_id:
        return _pipeline

    async with _pipeline_lock:
        # Double-check after acquiring lock
        if _pipeline is not None and _pipeline_model_id == model_id:
            return _pipeline

        def _load() -> Any:
            import torch
            from diffusers import StableDiffusionPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            print(
                f"[image_gen] Loading model '{model_id}' on {device} "
                f"(dtype={dtype}) — this may take a moment on first run..."
            )

            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,       # local use; no content filter overhead
                requires_safety_checker=False,
            )
            pipe = pipe.to(device)

            # Enable memory-efficient attention if available
            if device == "cpu":
                try:
                    pipe.enable_attention_slicing()
                except Exception:
                    pass

            print(f"[image_gen] Model '{model_id}' loaded on {device}")
            return pipe

        _pipeline = await asyncio.to_thread(_load)
        _pipeline_model_id = model_id
        return _pipeline


def _get_workspace() -> Path:
    """Get the workspace directory for saving generated images."""
    workspace = BASE_DIR / "data" / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    images_dir = workspace / "generated_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def _closest_size(requested: int) -> int:
    """Round the requested size to the nearest valid size."""
    return min(_VALID_SIZES, key=lambda x: abs(x - requested))


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: Generate Image (local model, no API)
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="generate_image",
    description=(
        "Generate an AI image from a text prompt using a fully local model.  "
        "The model runs entirely on this machine — no API keys, no third-party "
        "services, works fully offline.  The image is saved to the workspace "
        "as a PNG file and a base64 preview is returned."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed text prompt describing the image to generate.",
            },
            "negative_prompt": {
                "type": "string",
                "description": "Things to avoid in the image (e.g. 'blurry, low quality, deformed').",
            },
            "size": {
                "type": "integer",
                "description": (
                    "Image size in pixels (width=height).  "
                    f"Valid values: {_VALID_SIZES}.  Default: {_DEFAULT_SIZE}."
                ),
            },
            "num_inference_steps": {
                "type": "integer",
                "description": (
                    "Number of denoising steps (higher = better quality but slower).  "
                    f"Default: {_DEFAULT_STEPS}.  On CPU, 4-10 gives good quality in 2-5 minutes."
                ),
            },
            "save_to_workspace": {
                "type": "boolean",
                "description": "Whether to save the image to workspace (default: true).",
            },
        },
        "required": ["prompt"],
    },
    category="media",
    risk="medium",
    timeout=300,
)
async def generate_image(params: dict) -> str:
    """Generate an image from a text prompt using a local model."""
    prompt = params.get("prompt", "").strip()
    if not prompt:
        return "Error: No prompt provided. Describe the image you want to generate."

    negative_prompt = params.get("negative_prompt", "").strip() or None
    size = params.get("size", _DEFAULT_SIZE)
    steps = params.get("num_inference_steps", _DEFAULT_STEPS)
    save_to_workspace = params.get("save_to_workspace", True)

    # Clamp / validate parameters (CPU-friendly limits)
    size = _closest_size(size)
    steps = max(3, min(30, steps))  # DPMSolver works well at low step counts

    await emit("image_gen_start", prompt=prompt[:100], size=size, steps=steps)
    start_time = time.time()

    try:
        pipe = await _get_pipeline()
    except ImportError as e:
        return (
            "Error: Missing required dependency.\n\n"
            "Run:  pip install diffusers\n\n"
            f"Details: {e}"
        )
    except Exception as e:
        return (
            f"Error: Failed to load the local image generation model.\n\n"
            f"Model: {_MODEL_ID}\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Check your internet connection (first run downloads the model), "
            f"or try another model by changing the _MODEL_ID constant."
        )

    try:
        # Run inference in a thread (blocking operation)
        def _infer() -> bytes:
            import torch

            # Build call kwargs
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "height": size,
                "width": size,
                "num_inference_steps": steps,
            }
            if negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt

            # Thread lock: PyTorch models aren't safe for concurrent forward passes
            with _pipeline_thread_lock, torch.no_grad():
                result = pipe(**call_kwargs)
                image = result.images[0]

            buf = BytesIO()
            image.save(buf, format="PNG")
            return buf.getvalue()

        image_bytes = await asyncio.to_thread(_infer)

        # Base64 for preview
        b64_data: str = base64.b64encode(image_bytes).decode("utf-8")
        elapsed = round(time.time() - start_time, 2)

        # Save to workspace
        file_path = None
        if save_to_workspace:
            workspace = _get_workspace()
            timestamp = int(time.time())
            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:50])
            filename = f"gen_{timestamp}_{safe_prompt.strip()}.png"
            file_path = workspace / filename
            file_path.write_bytes(image_bytes)

        # Build result text
        device = getattr(pipe, "_execution_device", None) or pipe.device if hasattr(pipe, "device") else "?"
        device_str = str(device)

        result_lines = [
            "## Image Generated (Local Model)",
            f"- **Model**: {_MODEL_ID}",
            f"- **Device**: {device_str}",
            f"- **Size**: {size}×{size}",
            f"- **Steps**: {steps}",
            f"- **Prompt**: {prompt[:200]}{'...' if len(prompt) > 200 else ''}",
            f"- **Time**: {elapsed}s",
        ]
        if negative_prompt:
            result_lines.append(f"- **Negative prompt**: {negative_prompt[:200]}")

        # File size
        result_lines.append(f"- **Image size**: {len(image_bytes) / 1024:.1f} KB")

        if file_path:
            result_lines.append(f"- **File**: `{file_path}`")

        # Base64 preview (truncated for display, full data emitted as event)
        _preview_len = 500
        preview = b64_data[:_preview_len]
        preview_note = f" (showing first {_preview_len} chars)" if len(b64_data) > _preview_len else ""
        result_lines.append(f"\nBase64 preview{preview_note}:\n`{preview}`")

        await emit(
            "image_gen_complete",
            model_id=_MODEL_ID,
            size=size,
            steps=steps,
            elapsed=elapsed,
            file_path=str(file_path) if file_path else None,
        )
        return "\n".join(result_lines)

    except Exception as e:
        return f"Error: Image generation failed — {type(e).__name__}: {e}"
