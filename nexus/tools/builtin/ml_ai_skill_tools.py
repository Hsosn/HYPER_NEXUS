"""
ML/AI Skill Tools — Bridges the 13 ml_ai_skills into the AgentNexus tool registry.

Each skill is a self-contained PyTorch/ML module in the ``ml_ai_skills/`` directory.
This file exposes every skill as a ``@tool``-decorated function that:

1. Dynamically imports the skill's Python module
2. Delegates the requested ``action`` with optional ``params``
3. Falls back gracefully when the module or its dependencies are missing

Optimization features:
- Thread-safe module import cache with TTL-based refresh
- Parallel skill execution support (asyncio.gather compatible)
- Early-exit for unavailable modules
- Comprehensive error handling with dependency hints
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from ..registry import tool

logger = logging.getLogger("nexus.ml_ai_skill_tools")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ml_ai_skills"

if _SKILLS_DIR.is_dir() and str(_SKILLS_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILLS_DIR))
    logger.info("Added ml_ai_skills directory to sys.path: %s", _SKILLS_DIR)

# ---------------------------------------------------------------------------
# Enhanced module cache with TTL
# ---------------------------------------------------------------------------

_SKILL_MODULE_CACHE: dict[str, tuple[float, Any]] = {}
_MODULE_CACHE_TTL = 300.0  # 5 minutes


def _get_skill_module(skill_name: str) -> Any:
    """Get a cached skill module, reloading if TTL expired."""
    now = time.monotonic()
    entry = _SKILL_MODULE_CACHE.get(skill_name)
    if entry and (now - entry[0]) < _MODULE_CACHE_TTL:
        return entry[1]
    # Import fresh
    module = importlib.import_module(f"{skill_name}.{skill_name}")
    _SKILL_MODULE_CACHE[skill_name] = (now, module)
    # Evict oldest if cache too large
    if len(_SKILL_MODULE_CACHE) > 32:
        oldest = sorted(_SKILL_MODULE_CACHE.items(), key=lambda x: x[1][0])[:4]
        for k, _ in oldest:
            del _SKILL_MODULE_CACHE[k]
    return module


# ---------------------------------------------------------------------------
# Shared parameter schema
# ---------------------------------------------------------------------------

_SKILL_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "description": (
                "The skill action to perform. Common actions include 'build', 'train', "
                "'evaluate', 'analyze', 'design', 'apply', 'generate', 'export'. "
                "Refer to the skill's SKILL.md for the full list of supported actions."
            ),
        },
        "params": {
            "type": "object",
            "description": "Key-value parameters for the action.",
            "additionalProperties": True,
        },
    },
    "required": ["action"],
}


# ---------------------------------------------------------------------------
# Dynamic skill caller helper (optimized)
# ---------------------------------------------------------------------------

async def _call_skill(skill_name: str, action: str, params: dict[str, Any] | None = None) -> str:
    """Dynamically import and call a skill module with caching + early exit."""
    params = params or {}

    # Early dependency check
    torch_available = False
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        pass

    # Try to import the skill module (cached with TTL)
    try:
        module = _get_skill_module(skill_name)
    except ImportError as exc:
        if not torch_available:
            return json.dumps({
                "status": "unavailable",
                "skill": skill_name,
                "action": action,
                "error": f"Skill '{skill_name}' requires PyTorch (not installed). Install with: pip install torch",
                "missing_dependency": "torch",
            })
        return json.dumps({
            "status": "unavailable",
            "skill": skill_name,
            "action": action,
            "error": f"Skill module '{skill_name}' could not be imported: {exc}.",
        })

    # Try run_action (preferred entry point)
    run_action_fn = getattr(module, "run_action", None)
    if callable(run_action_fn):
        try:
            result = run_action_fn(action, params)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return json.dumps({
                "status": "error",
                "skill": skill_name,
                "action": action,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-800:],
            })

    # Fallback: introspect module and show available API
    public_names = [name for name in dir(module) if not name.startswith("_") and not name.isupper()]
    classes = [name for name in public_names if isinstance(getattr(module, name, None), type)]
    functions = [name for name in public_names if callable(getattr(module, name, None)) and not isinstance(getattr(module, name), type)]

    result: dict[str, Any] = {
        "status": "available",
        "skill": skill_name,
        "action": action,
        "message": f"Skill '{skill_name}' loaded successfully.",
        "available_classes": sorted(classes),
        "available_functions": sorted(functions),
    }

    # Try calling action as a function if it matches
    if action in functions:
        try:
            fn = getattr(module, action)
            import inspect
            sig = inspect.signature(fn)
            result["matched_function"] = action
            result["function_signature"] = str(sig)
            param_count = len(sig.parameters)
            if param_count == 0:
                call_result = fn()
                result["result"] = str(call_result) if call_result is not None else "None"
                result["status"] = "success"
            elif param_count == 1:
                call_result = fn(params)
                result["result"] = str(call_result) if call_result is not None else "None"
                result["status"] = "success"
        except Exception as exc:
            result["call_error"] = f"{type(exc).__name__}: {exc}"

    if action in classes:
        cls = getattr(module, action)
        import inspect
        try:
            sig = inspect.signature(cls.__init__)
            result["matched_class"] = action
            result["class_signature"] = str(sig)
        except (ValueError, TypeError):
            result["matched_class"] = action

    return json.dumps(result, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Transformer Architect
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_transformer_architect",
    description=(
        "Transformer Architect — Design and build advanced transformer architectures. "
        "Supports GPT-2 (decoder-only), BERT (encoder-only), T5 (encoder-decoder), "
        "and modern LLaMA-style blocks with GQA, MQA, RoPE, SwiGLU, RMSNorm, KV-Cache. "
        "Actions: 'build', 'attention', 'export'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["transformer", "attention", "gpt2", "bert", "t5", "llama"],
    timeout=120,
)
async def ml_transformer_architect(params: dict) -> str:
    return await _call_skill("transformer_architect", params.get("action", "build"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Model Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_model_optimizer",
    description=(
        "Model Optimizer — Optimize ML models through pruning, quantization, "
        "knowledge distillation, and hyperparameter search. Supports unstructured/structured "
        "pruning, dynamic int8 + QAT quantization, teacher-student distillation, LR range test. "
        "Actions: 'tune', 'prune', 'quantize', 'distill', 'lr_finder'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["optimization", "pruning", "quantization", "distillation"],
    timeout=180,
)
async def ml_model_optimizer(params: dict) -> str:
    return await _call_skill("model_optimizer", params.get("action", "tune"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: Deep Learning Trainer
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_deep_learning_trainer",
    description=(
        "Deep Learning Trainer — Complete PyTorch training engine with CNN, MLP, and "
        "Transformer architectures. Features AMP (bf16/fp16), gradient accumulation, "
        "warmup cosine LR, early stopping, torch.compile, gradient checkpointing. "
        "Actions: 'train', 'evaluate'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["training", "deep_learning", "cnn", "mlp", "transformer"],
    timeout=300,
)
async def ml_deep_learning_trainer(params: dict) -> str:
    return await _call_skill("deep_learning_trainer", params.get("action", "train"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: CV Workbench
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_cv_workbench",
    description=(
        "CV Workbench — Computer vision tools using PyTorch and torchvision. "
        "Supports classification (CNN, ResNet, EfficientNet, ViT), detection "
        "(Faster R-CNN, SSD, YOLO), segmentation (U-Net, FCN, DeepLabV3), "
        "and augmentation (MixUp, CutMix, AutoAugment, RandAugment). "
        "Actions: 'classify', 'detect', 'segment', 'augment', 'preprocess'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["cv", "vision", "classification", "detection", "segmentation"],
    timeout=180,
)
async def ml_cv_workbench(params: dict) -> str:
    return await _call_skill("cv_workbench", params.get("action", "classify"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: RLHF Lab
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_rlhf_lab",
    description=(
        "RLHF Lab — LLM alignment methods. Implements DPO, KTO, ORPO, SimPO, "
        "Constitutional AI (self-critique/revision), and SafetyEvaluator. "
        "Actions: 'sft', 'dpo', 'kto', 'orpo', 'simpo', 'safety_eval'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["rlhf", "alignment", "dpo", "kto", "orpo", "ppo", "safety"],
    timeout=300,
)
async def ml_rlhf_lab(params: dict) -> str:
    return await _call_skill("rlhf_lab", params.get("action", "dpo"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6: LLM Trainer
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_llm_trainer",
    description=(
        "LLM Trainer — Train GPT-style language models. Full GPT from scratch with "
        "multi-head causal attention, LoRA/QLoRA (4-bit NF4), PPO RLHF, AMP, "
        "torch.compile, DDP/FSDP, DeepSpeed ZeRO. "
        "Actions: 'train_gpt', 'lora_finetune', 'rlhf', 'export'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["llm", "gpt", "lora", "qlora", "rlhf", "deepspeed"],
    timeout=300,
)
async def ml_llm_trainer(params: dict) -> str:
    return await _call_skill("llm_trainer", params.get("action", "train_gpt"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7: Distributed Training
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_distributed_training",
    description=(
        "Distributed Training — Scale training across GPUs. DDP, FSDP with auto-wrap "
        "and CPU offload, DeepSpeed ZeRO 1/2/3, pipeline parallelism, elastic training. "
        "Actions: 'ddp', 'fsdp', 'deepspeed_config', 'pipeline', 'compile'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["distributed", "ddp", "fsdp", "deepspeed", "pipeline"],
    timeout=300,
)
async def ml_distributed_training(params: dict) -> str:
    return await _call_skill("distributed_training", params.get("action", "ddp"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8: Data Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_data_pipeline",
    description=(
        "Data Pipeline — End-to-end data processing. Preprocessing (imputation, "
        "outlier detection, encoding), feature engineering (polynomial, ratio, log), "
        "stratified splitting, and PyTorch DataLoader creation. "
        "Actions: 'preprocess', 'features', 'split', 'dataloader'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["data", "pipeline", "preprocessing", "feature_engineering"],
    timeout=120,
)
async def ml_data_pipeline(params: dict) -> str:
    return await _call_skill("data_pipeline", params.get("action", "preprocess"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9: RL Lab
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_rl_lab",
    description=(
        "RL Lab — Reinforcement learning with PyTorch. DQN, REINFORCE (with baseline), "
        "A2C (with GAE), and PPO. Supports GridWorld, CartPole, MountainCar, LunarLander. "
        "Actions: 'env', 'agent', 'train', 'evaluate'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["rl", "reinforcement_learning", "dqn", "ppo", "a2c"],
    timeout=300,
)
async def ml_rl_lab(params: dict) -> str:
    return await _call_skill("rl_lab", params.get("action", "agent"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10: NLP Workbench
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_nlp_workbench",
    description=(
        "NLP Workbench — Natural language processing. Text classification (RNN, BiLSTM, "
        "CNN-text, Transformer), sentiment analysis, tokenization, embeddings "
        "(HuggingFace, sentence-transformers), NER with HF pipeline. "
        "Actions: 'classify', 'sentiment', 'tokenize', 'embed', 'ner'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["nlp", "text", "classification", "sentiment", "embeddings", "ner"],
    timeout=120,
)
async def ml_nlp_workbench(params: dict) -> str:
    return await _call_skill("nlp_workbench", params.get("action", "classify"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 11: GAN Studio
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_gan_studio",
    description=(
        "GAN Studio — Generative Adversarial Networks. DCGAN, WGAN-GP (gradient penalty), "
        "CGAN (conditional), CycleGAN (unpaired translation), StyleGAN (AdaIN). "
        "Includes FID, IS, LPIPS evaluation metrics. "
        "Actions: 'design', 'train', 'generate', 'evaluate'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["gan", "generative", "dcgan", "wgan", "cyclegan", "stylegan"],
    timeout=300,
)
async def ml_gan_studio(params: dict) -> str:
    return await _call_skill("gan_studio", params.get("action", "design"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 12: Neural Architect
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_neural_architect",
    description=(
        "Neural Architect — Design, analyze, benchmark PyTorch architectures. "
        "CustomTransformerEncoder, ResNet (bottleneck), U-Net, LSTMWithAttention, "
        "VAE (beta-VAE with KL annealing). Includes param counts and inference benchmarks. "
        "Actions: 'design', 'analyze', 'benchmark'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["architecture", "resnet", "unet", "vae", "lstm", "transformer"],
    timeout=120,
)
async def ml_neural_architect(params: dict) -> str:
    return await _call_skill("neural_architect", params.get("action", "design"), params.get("params", {}))


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 13: PEFT Fine-Tuning
# ═══════════════════════════════════════════════════════════════════════════════

@tool(
    name="ml_peft_finetuning",
    description=(
        "PEFT Fine-Tuning — Parameter-efficient fine-tuning from scratch. "
        "LoRA (ΔW = B@A with merging), QLoRA (4-bit NF4), Adapter layers (bottleneck), "
        "Prefix Tuning (KV prefixes), Prompt Tuning (soft tokens), BitFit (bias-only), IA3. "
        "Actions: 'apply', 'analyze', 'merge'."
    ),
    parameters_schema=_SKILL_PARAMS_SCHEMA,
    risk="medium",
    category="ml_ai",
    tags=["peft", "lora", "qlora", "adapter", "prefix_tuning", "prompt_tuning"],
    timeout=120,
)
async def ml_peft_finetuning(params: dict) -> str:
    return await _call_skill("peft_finetuning", params.get("action", "apply"), params.get("params", {}))
