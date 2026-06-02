"""Local Vision Engine — on-device image analysis using Florence-2.

Uses Microsoft's Florence-2 (230M / 770M params) via HuggingFace Transformers.
Runs entirely on-device — no external API calls, no third-party services,
no Ollama. Requires PyTorch, Transformers, Pillow, einops, and timm.

The model is downloaded on first use (~900 MB) and cached locally thereafter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded singleton model
_model = None
_processor = None


_FLORENCE_TASK_PROMPTS = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression": "<REFERRING_EXPRESSION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "region_to_ocr": "<REGION_TO_OCR>",
}

# Keywords that suggest the user wants OCR / text extraction
_OCR_HINTS = [
    "text", "ocr", "read", "character", "letter", "word",
    "handwriting", "handwritten", "transcribe", "extract text",
    "what does it say", "what is written",
]


def _detect_task_type(user_prompt: str) -> str:
    """Detect the best Florence-2 task prompt from the user's request.

    Returns one of:
      - "ocr"                  — text extraction
      - "more_detailed_caption"  — comprehensive scene description
      - "vqa"                  — open-ended visual question answering
      - "detailed_caption"     — standard detailed description (default)
    """
    prompt_lower = user_prompt.lower()

    # Check for OCR hints
    if any(hint in prompt_lower for hint in _OCR_HINTS):
        return "ocr"

    # Check for detailed description requests
    detailed_hints = [
        "detailed", "describe in detail", "thorough", "comprehensive",
        "tell me everything", "what all", "full description",
    ]
    if any(hint in prompt_lower for hint in detailed_hints):
        return "more_detailed_caption"

    # Check for VQA — any specific question about the image content
    # that isn't covered by OCR or detailed description above.
    # Keywords that indicate the user is asking about something specific
    # (not just a general "describe this"):
    vqa_hints = [
        "what color", "what shape", "how many", "where is",
        "is there", "are there", "do you see", "can you see",
        "what is", "what are", "who is", "tell me about",
        "explain", "count", "identify", "locate", "find",
        "compare", "difference", "similar",
    ]
    # Only treat as "generic describe" when the prompt is a simple
    # "describe this image" variant — not focused questions like
    # "describe the colors" or "describe what text is in this image"
    generic_describe_phrases = [
        "describe this image",
        "describe the image",
        "describe what you see",
        "describe the picture",
        "describe this picture",
        "what do you see",
        "what can you see",
    ]
    is_generic_describe = any(phrase in prompt_lower for phrase in generic_describe_phrases)
    if any(hint in prompt_lower for hint in vqa_hints) and not is_generic_describe:
        return "vqa"

    # Default to detailed caption
    return "detailed_caption"


def _build_florence_prompt(user_prompt: str, task_type: str | None = None) -> str:
    """Build the Florence-2 task prompt.

    Florence-2 uses special tokens like `<CAPTION>`, `<DETAILED_CAPTION>`,
    `<OCR>`, etc. to specify the task. These task tokens must be the *only*
    token in the text — the processor will assert this. For open-ended VQA,
    just pass the user's question directly as the prompt.

    Returns the full prompt string to pass to the model.
    """
    if task_type is None:
        task_type = _detect_task_type(user_prompt)

    task_token = _FLORENCE_TASK_PROMPTS.get(task_type)

    if task_token:
        # Task tokens (like <OCR>, <DETAILED_CAPTION>) MUST be the only
        # text in the prompt — the Florence-2 processor enforces this.
        # The model generates the rest from the image alone.
        return task_token

    # Fallback: open-ended VQA — pass user's question as the prompt
    return user_prompt


def _ensure_model():
    """Lazy-load the Florence-2 model and processor.

    Downloads on first call, caches in HuggingFace cache directory.
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
        )
    except ImportError:
        raise ImportError(
            "Local vision requires `transformers`. Install with:\n"
            "  pip install transformers"
        )

    model_id = "microsoft/Florence-2-base"
    logger.info(f"Loading local vision model: {model_id} (first load downloads ~900MB)")

    try:
        # Patch the root PretrainedConfig class BEFORE loading the model.
        # Newer transformers versions (>=5.x) removed forced_bos_token_id
        # from PretrainedConfig, but Florence-2's cached custom config
        # (Florence2LanguageConfig, which inherits from PretrainedConfig)
        # still references self.forced_bos_token_id in its __init__.
        # We add the missing attribute at the root so all subclasses
        # inherit it automatically.
        from transformers import PretrainedConfig
        for attr in ("forced_bos_token_id", "forced_eos_token_id"):
            if not hasattr(PretrainedConfig, attr):
                setattr(PretrainedConfig, attr, None)

        # Patch RobertaTokenizer for transformers >=5.x compatibility.
        # Florence-2 processor internally loads a RobertaTokenizer.  Newer
        # transformers versions removed the `additional_special_tokens`
        # instance attribute; patch it onto the class so the model's custom
        # processing code doesn't crash.
        try:
            from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
            if not hasattr(RobertaTokenizer, "additional_special_tokens"):
                RobertaTokenizer.additional_special_tokens = []
        except ImportError:
            pass

        _processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        import torch
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        _model.eval()
        logger.info("Local vision model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Florence-2: {e}")
        raise


def _generate_and_parse(image: Image.Image, florence_prompt: str) -> str:
    """Run Florence-2 generation on a PIL image and parse the result.

    Shared helper used by both ``analyze_image`` (file path) and
    ``analyze_image_base64`` (raw base64 data).

    Args:
        image: PIL Image in RGB mode.
        florence_prompt: The Florence-2 task token (e.g. ``<DETAILED_CAPTION>``)
            or the VQA question string.

    Returns:
        Cleaned text description from the model.
    """
    # Process inputs
    inputs = _processor(
        text=florence_prompt,
        images=image,
        return_tensors="pt",
    )

    # Generate
    try:
        generated_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            use_cache=False,
        )
    except Exception as e:
        logger.error(f"Florence-2 generation failed: {e}")
        raise RuntimeError(f"Local vision model failed to generate: {e}")

    # Decode
    generated_text = _processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    # Clean up the output
    parsed_answer = _processor.post_process_generation(
        generated_text,
        task=florence_prompt,
        image_size=(image.width, image.height),
    )

    # Extract the actual answer text
    if isinstance(parsed_answer, dict):
        for key in ("generated_text", "answer", "caption", "text"):
            if key in parsed_answer:
                answer = parsed_answer[key]
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
        for v in parsed_answer.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return str(parsed_answer)

    if isinstance(generated_text, str):
        for prefix in (
            florence_prompt,
            "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>",
            "<OCR>", "<OD>",
        ):
            if generated_text.startswith(prefix):
                generated_text = generated_text[len(prefix):].strip()
        return generated_text.strip()

    return str(generated_text)


async def analyze_image(
    file_path: str | Path,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze an image file from disk using the local Florence-2 vision model.

    Args:
        file_path: Path to the image file (PNG, JPG, JPEG, WebP, BMP).
        prompt: Natural language question or instruction about the image.

    Returns:
        Text description / analysis from the model.
    """
    from PIL import Image

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Load image
    image = Image.open(path).convert("RGB")
    logger.debug(f"Loaded image: {path} ({image.size})")

    # Load model (lazy, cached after first call)
    _ensure_model()

    # Build Florence-2 prompt
    task_type = _detect_task_type(prompt)
    florence_prompt = _build_florence_prompt(prompt, task_type)
    logger.debug(f"Florence-2 task: {task_type}, prompt: {florence_prompt[:80]}...")

    return _generate_and_parse(image, florence_prompt)


async def analyze_image_base64(
    b64_data: str,
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze a base64-encoded image using the local Florence-2 vision model.

    Useful for images that come from the WebUI chat (encoded as base64 strings)
    or from other sources where no file path is available.

    Args:
        b64_data: Base64-encoded image data (raw base64, without any
            ``data:image/...`` prefix).
        prompt: Natural language question or instruction about the image.

    Returns:
        Text description / analysis from the model.
    """
    from io import BytesIO
    from PIL import Image

    try:
        img_bytes = base64.b64decode(b64_data)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {e}")

    logger.debug(f"Loaded base64 image: {image.size}")

    # Load model (lazy, cached after first call)
    _ensure_model()

    # Build Florence-2 prompt
    task_type = _detect_task_type(prompt)
    florence_prompt = _build_florence_prompt(prompt, task_type)
    logger.debug(f"Florence-2 task: {task_type}, prompt: {florence_prompt[:80]}...")

    return _generate_and_parse(image, florence_prompt)


def unload_model():
    """Unload the model from memory to free resources."""
    global _model, _processor
    _model = None
    _processor = None
    import gc
    gc.collect()
    logger.info("Local vision model unloaded.")
