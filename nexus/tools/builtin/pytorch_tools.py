"""
PyTorch / ML-AI Tools — bridges PyTorch, torchvision, scikit-learn, and advanced ML
capabilities into the AgentNexus framework.

Tools registered:
- pt_model_create          : Build neural network models (sequential, custom nn.Module)
- pt_model_train           : Train models with configurable hyperparameters, mixed precision,
                              early stopping, LR scheduling, gradient clipping
- pt_model_predict         : Run inference / predictions on trained models (torch.no_grad)
- pt_model_evaluate        : Evaluate model with metrics (loss, accuracy, classification_report)
- pt_model_save            : Save model as .pt / .pth (state_dict or full model)
- pt_model_load            : Load a previously saved model
- pt_model_info            : Get model architecture, parameter count, layer details
- pt_data_preprocess       : Data preprocessing pipelines (normalize, encode, split, DataLoader)
- pt_data_augment          : Data augmentation for images (torchvision.transforms)
- pt_transfer_learning     : Set up transfer learning from torchvision pretrained models
- pt_model_visualize       : Generate model summary and architecture visualization
- pt_hyperparameter_tune   : Hyperparameter search (grid, random, Bayesian-inspired)
- pt_text_classification   : End-to-end text classification pipeline
- pt_image_classification  : End-to-end image classification pipeline
- pt_model_convert         : Convert to ONNX / TorchScript for deployment
- pt_training_monitor      : Monitor training progress, detect overfitting

Requires: torch, scikit-learn, numpy
Optional: torchvision (for pretrained CV models and image augmentation), onnx, onnxruntime
Graceful degradation if PyTorch is not installed.
"""
from __future__ import annotations

import copy
import io
import itertools
import json
import math
import os
import random
import struct
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...config import BASE_DIR
from ..registry import tool

# ── Import-time detection ─────────────────────────────────────────────────────

_TORCH_AVAILABLE = False
_TORCHVISION_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_NP_AVAILABLE = False
_ONNX_AVAILABLE = False
_IMPORT_ERRORS: list[str] = []

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"numpy: {e}")

try:
    import sklearn
    _SKLEARN_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"scikit-learn: {e}")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    _TORCH_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"torch: {e}")

try:
    import torchvision
    import torchvision.transforms as T
    import torchvision.models as tv_models
    _TORCHVISION_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERRORS.append(f"torchvision: {e}")

try:
    import onnx
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ── Workspace helper ──────────────────────────────────────────────────────────

def _get_workspace() -> Path:
    """Get the workspace directory for saving models/data."""
    base = BASE_DIR / "data" / "workspace"
    base.mkdir(parents=True, exist_ok=True)
    ml_dir = base / "ml_models"
    ml_dir.mkdir(parents=True, exist_ok=True)
    return ml_dir


def _numpy_safe(obj: Any) -> Any:
    """Convert numpy / torch types to JSON-serializable Python types."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _torch_safe(obj: Any) -> Any:
    """Convert PyTorch tensors to Python native types for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, (torch.nn.Parameter,)):
        return obj.detach().cpu().item()
    return _numpy_safe(obj)


def _check_available() -> str | None:
    """Return error message if ML libraries are not available."""
    missing: list[str] = []
    if not _NP_AVAILABLE:
        missing.append("numpy")
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if missing:
        return (
            f"ML libraries not available: {', '.join(missing)}. "
            f"Errors: {'; '.join(_IMPORT_ERRORS)}. "
            f"Install with: pip install torch scikit-learn numpy"
        )
    return None


def _check_sklearn() -> str | None:
    """Return error message if scikit-learn is not available."""
    if not _SKLEARN_AVAILABLE:
        return "scikit-learn is not installed. Install with: pip install scikit-learn"
    return None


def _check_torchvision() -> str | None:
    """Return error message if torchvision is not available."""
    if not _TORCHVISION_AVAILABLE:
        return "torchvision is not installed. Install with: pip install torchvision"
    return None


# ── In-memory model registry (session-scoped) ────────────────────────────────

_MODEL_REGISTRY: dict[str, Any] = {}
"""Stores trained models keyed by model_name for cross-tool usage.
Each entry is a dict with keys: 'model' (nn.Module), 'optimizer', 'loss_fn',
'metrics', 'loss_name', 'input_shape', 'num_classes', 'optimizer_name'.
"""

_TRAINING_HISTORY: dict[str, dict] = {}


# ── Custom PyTorch Modules ───────────────────────────────────────────────────

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM wrapper compatible with nn.Sequential."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dropout: float = 0.0, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


class ResidualBlock(nn.Module):
    """Simple residual block: linear -> bn -> relu -> linear -> skip connection."""

    def __init__(self, in_features: int, out_features: int | None = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.downsample: Optional[nn.Linear] = None
        if in_features != out_features:
            self.downsample = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ── Model wrapper stored in registry ─────────────────────────────────────────

@dataclass
class ModelWrapper:
    """Wraps a PyTorch model with training metadata for the registry."""
    model: nn.Module
    optimizer: Optional[optim.Optimizer] = None
    loss_fn: Optional[nn.Module] = None
    loss_name: str = "cross_entropy"
    optimizer_name: str = "adam"
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])
    input_shape: Optional[tuple[int, ...]] = None
    num_classes: int = 10
    device: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "loss_name": self.loss_name,
            "optimizer_name": self.optimizer_name,
            "metrics": self.metrics,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "device": self.device,
        }


def _get_device() -> str:
    """Return the best available device string."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    if _TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _get_activation(name: str) -> Optional[nn.Module]:
    """Map activation name string to PyTorch module."""
    mapping: dict[str, nn.Module] = {
        "relu": nn.ReLU(inplace=True),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
        "elu": nn.ELU(inplace=True),
        "selu": nn.SELU(inplace=True),
        "gelu": nn.GELU(),
        "softmax": nn.Softmax(dim=-1),
        "log_softmax": nn.LogSoftmax(dim=-1),
        "linear": None,
        "none": None,
    }
    return mapping.get(name.lower(), nn.ReLU(inplace=True))


def _build_optimizer(name: str, model_params, lr: float = 1e-3,
                     weight_decay: float = 0.0) -> optim.Optimizer:
    """Create optimizer from name string."""
    name_lower = name.lower().strip()
    if name_lower == "adam":
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "adamw":
        return optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name_lower == "rmsprop":
        return optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "adagrad":
        return optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
    else:
        return optim.Adam(model_params, lr=lr, weight_decay=weight_decay)


def _build_loss(name: str) -> nn.Module:
    """Create loss function from name string."""
    name_lower = name.lower().strip()
    if name_lower in ("cross_entropy", "categorical_crossentropy"):
        return nn.CrossEntropyLoss()
    elif name_lower == "binary_crossentropy" or name_lower == "bce":
        return nn.BCEWithLogitsLoss()
    elif name_lower == "mse" or name_lower == "mean_squared_error":
        return nn.MSELoss()
    elif name_lower == "mae" or name_lower == "l1":
        return nn.L1Loss()
    elif name_lower == "huber":
        return nn.SmoothL1Loss()
    elif name_lower == "nll":
        return nn.NLLLoss()
    else:
        return nn.CrossEntropyLoss()


def _auto_detect_loss(num_classes: int) -> str:
    """Auto-detect loss function based on number of classes."""
    if num_classes > 2:
        return "cross_entropy"
    elif num_classes == 1:
        return "binary_crossentropy"
    elif num_classes == 0:
        return "mse"
    return "cross_entropy"


def _simple_tokenizer(texts: list[str], vocab_size: int = 10000,
                      max_length: int = 200) -> tuple[list[list[int]], dict[str, int]]:
    """Simple word-level tokenizer for text data."""
    from collections import Counter
    word_counts: Counter[str] = Counter()
    for text in texts:
        words = str(text).lower().split()
        word_counts.update(words)

    most_common = word_counts.most_common(vocab_size - 2)
    word_index: dict[str, int] = {"<PAD>": 0, "<OOV>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word_index[word] = idx

    sequences: list[list[int]] = []
    for text in texts:
        words = str(text).lower().split()
        seq = [word_index.get(w, 1) for w in words]
        seq = seq[:max_length]
        seq += [0] * (max_length - len(seq))
        sequences.append(seq)

    return sequences, word_index


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: Create Model
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_create",
    description=(
        "Create a neural network model using PyTorch (nn.Sequential / custom nn.Module). "
        "Provide layers as a JSON array with type, units, activation, and optional parameters. "
        "Supports dense, conv2d, lstm, gru, dropout, batch_norm, flatten, max_pool, "
        "global_avg_pool, embedding, bidirectional_lstm, attention, and residual_block layers. "
        "Returns model summary with parameter counts."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Unique name for the model (used for later train/predict/save)",
            },
            "model_type": {
                "type": "string",
                "enum": ["sequential", "functional"],
                "description": "Architecture type (default: sequential)",
            },
            "input_shape": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Input shape tuple, e.g. [28, 28, 1] or [100] or [3, 224, 224]",
            },
            "num_classes": {
                "type": "integer",
                "description": "Number of output classes (0 for regression)",
            },
            "layers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": (
                                "Layer type: dense, conv2d, lstm, gru, dropout, batch_norm, "
                                "flatten, max_pool, global_avg_pool, embedding, "
                                "bidirectional_lstm, attention, residual_block"
                            ),
                        },
                        "units": {"type": "integer"},
                        "filters": {"type": "integer"},
                        "kernel_size": {"type": "array", "items": {"type": "integer"}},
                        "strides": {"type": "array", "items": {"type": "integer"}},
                        "activation": {"type": "string"},
                        "dropout_rate": {"type": "number"},
                        "return_sequences": {"type": "boolean"},
                        "num_heads": {"type": "integer"},
                        "key_dim": {"type": "integer"},
                        "input_dim": {"type": "integer"},
                        "padding": {"type": "string"},
                    },
                    "required": ["type"],
                },
                "description": "Array of layer definitions. Example: [{'type':'dense','units':128,'activation':'relu'}]",
            },
            "optimizer": {
                "type": "string",
                "description": "Optimizer: adam, adamw, sgd, rmsprop, adagrad (default: adam)",
            },
            "loss": {
                "type": "string",
                "description": "Loss function: cross_entropy, binary_crossentropy, mse, mae, huber (default: auto)",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of metrics: accuracy, precision, recall, auc (default: [accuracy])",
            },
            "learning_rate": {
                "type": "number",
                "description": "Initial learning rate (default: 0.001)",
            },
            "weight_decay": {
                "type": "number",
                "description": "Weight decay for optimizer (default: 0.0)",
            },
        },
        "required": ["model_name"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "neural_network", "model", "deep_learning", "nn_module"],
    timeout=60,
    version="1.0",
)
async def pt_model_create(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "unnamed_model")
        model_type: str = params.get("model_type", "sequential")
        input_shape: list[int] | None = params.get("input_shape")
        num_classes: int = params.get("num_classes", 10)
        layer_defs: list[dict] = params.get("layers", [])
        optimizer_name: str = params.get("optimizer", "adam")
        loss_name: str = params.get("loss", "")
        metric_names: list[str] = params.get("metrics", ["accuracy"])
        lr: float = float(params.get("learning_rate", 0.001))
        weight_decay: float = float(params.get("weight_decay", 0.0))

        # ── Default layers if none provided ──
        if not layer_defs:
            needs_flatten = input_shape is not None and len(input_shape) > 1
            out_act = "softmax" if num_classes > 2 else ("sigmoid" if num_classes == 1 else "linear")
            layer_defs = [
                {"type": "flatten" if needs_flatten else "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "dropout_rate": 0.3},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "dropout_rate": 0.2},
                {"type": "dense", "units": num_classes if num_classes > 1 else 1, "activation": out_act},
            ]

        # ── Build PyTorch layers ──
        pytorch_layers: list[nn.Module] = []
        last_out_features: int = 1
        current_spatial: tuple[int, ...] = tuple(input_shape) if input_shape else (1,)
        # Track dimensions for LSTM input_size
        _lstm_input_size: int | None = None

        for i, layer_def in enumerate(layer_defs):
            ltype: str = layer_def.get("type", "dense").lower().strip()

            if ltype == "dense":
                units: int = layer_def.get("units", 64)
                activation_name: str = layer_def.get("activation", "relu")
                linear = nn.Linear(last_out_features, units)
                pytorch_layers.append(linear)
                act = _get_activation(activation_name)
                if act is not None:
                    pytorch_layers.append(act)
                last_out_features = units

            elif ltype == "conv2d":
                in_channels = current_spatial[0] if len(current_spatial) >= 3 else 3
                filters = layer_def.get("filters", 32)
                kernel_size = tuple(layer_def.get("kernel_size", [3, 3]))
                strides = tuple(layer_def.get("strides", [1, 1]))
                padding = layer_def.get("padding", "same")
                activation_name = layer_def.get("activation", "relu")

                if padding == "same":
                    pad_h = (kernel_size[0] - 1) // 2
                    pad_w = (kernel_size[1] - 1) // 2
                    padding_val = (pad_h, pad_w)
                elif isinstance(padding, (list, tuple)):
                    padding_val = tuple(padding)
                else:
                    padding_val = 0

                conv = nn.Conv2d(in_channels, filters, kernel_size,
                                 stride=strides, padding=padding_val)
                pytorch_layers.append(conv)
                act = _get_activation(activation_name)
                if act is not None:
                    pytorch_layers.append(act)
                # Update spatial tracking
                h = (current_spatial[1] + 2 * padding_val[0] - kernel_size[0]) // strides[0] + 1
                w = (current_spatial[2] + 2 * padding_val[1] - kernel_size[1]) // strides[1] + 1
                current_spatial = (filters, h, w)
                last_out_features = filters * h * w

            elif ltype == "lstm":
                hidden_size = layer_def.get("units", 64)
                return_seq = layer_def.get("return_sequences", False)
                dropout_rate = layer_def.get("dropout_rate", 0.0)
                input_size = last_out_features if _lstm_input_size is None else _lstm_input_size
                lstm = nn.LSTM(input_size, hidden_size, batch_first=True,
                               dropout=dropout_rate)
                pytorch_layers.append(lstm)
                _lstm_input_size = hidden_size
                if not return_seq:
                    last_out_features = hidden_size

            elif ltype == "gru":
                hidden_size = layer_def.get("units", 64)
                dropout_rate = layer_def.get("dropout_rate", 0.0)
                input_size = last_out_features if _lstm_input_size is None else _lstm_input_size
                gru = nn.GRU(input_size, hidden_size, batch_first=True,
                             dropout=dropout_rate)
                pytorch_layers.append(gru)
                _lstm_input_size = hidden_size
                last_out_features = hidden_size

            elif ltype == "bidirectional_lstm":
                hidden_size = layer_def.get("units", 64)
                return_seq = layer_def.get("return_sequences", False)
                dropout_rate = layer_def.get("dropout_rate", 0.0)
                input_size = last_out_features if _lstm_input_size is None else _lstm_input_size
                bi_lstm = BidirectionalLSTM(input_size, hidden_size, dropout=dropout_rate)
                pytorch_layers.append(bi_lstm)
                _lstm_input_size = hidden_size * 2
                if not return_seq:
                    last_out_features = hidden_size * 2

            elif ltype == "dropout":
                rate = layer_def.get("dropout_rate", 0.3)
                pytorch_layers.append(nn.Dropout(p=rate))

            elif ltype == "batch_norm":
                # Choose 1d or 2d based on context
                if len(current_spatial) >= 3 and current_spatial[0] > 1:
                    pytorch_layers.append(nn.BatchNorm2d(current_spatial[0]))
                else:
                    pytorch_layers.append(nn.BatchNorm1d(last_out_features))

            elif ltype == "flatten":
                pytorch_layers.append(nn.Flatten())
                # After flatten, set last_out_features
                if len(current_spatial) >= 3:
                    last_out_features = current_spatial[0] * current_spatial[1] * current_spatial[2]
                    current_spatial = (last_out_features,)
                _lstm_input_size = None  # reset

            elif ltype == "max_pool":
                pool_size = tuple(layer_def.get("kernel_size", [2, 2]))
                pytorch_layers.append(nn.MaxPool2d(kernel_size=pool_size))
                if len(current_spatial) >= 3:
                    h = current_spatial[1] // pool_size[0]
                    w = current_spatial[2] // pool_size[1]
                    current_spatial = (current_spatial[0], h, w)
                    last_out_features = current_spatial[0] * h * w

            elif ltype == "global_avg_pool":
                pytorch_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                if len(current_spatial) >= 3:
                    last_out_features = current_spatial[0]
                    current_spatial = (current_spatial[0],)

            elif ltype == "embedding":
                vocab_dim = layer_def.get("input_dim", 10000)
                embed_dim = layer_def.get("units", 128)
                pytorch_layers.append(nn.Embedding(vocab_dim, embed_dim))
                last_out_features = embed_dim * (input_shape[0] if input_shape else 200)
                _lstm_input_size = embed_dim

            elif ltype == "attention":
                num_heads = layer_def.get("num_heads", 4)
                embed_dim = layer_def.get("key_dim", 64) * num_heads
                attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                pytorch_layers.append(attn)
                last_out_features = embed_dim

            elif ltype == "residual_block":
                units = layer_def.get("units", 64)
                res_block = ResidualBlock(last_out_features, units)
                pytorch_layers.append(res_block)
                last_out_features = units

            else:
                warnings.warn(f"Unknown layer type: {ltype}, skipping")

        # ── Build nn.Sequential model ──
        if not pytorch_layers:
            return "[PyTorch Error] No valid layers were constructed."

        model = nn.Sequential(*pytorch_layers)

        # ── Auto-detect loss ──
        if not loss_name:
            loss_name = _auto_detect_loss(num_classes)

        loss_fn = _build_loss(loss_name)

        # ── Build optimizer ──
        optimizer = _build_optimizer(optimizer_name, model.parameters(),
                                     lr=lr, weight_decay=weight_decay)

        # ── Register ──
        device = _get_device()
        wrapper = ModelWrapper(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_name=loss_name,
            optimizer_name=optimizer_name,
            metrics=metric_names,
            input_shape=tuple(input_shape) if input_shape else None,
            num_classes=num_classes,
            device=device,
        )
        _MODEL_REGISTRY[model_name] = wrapper

        # ── Collect summary ──
        total_params, trainable_params = _count_parameters(model)
        layer_summaries: list[dict] = []
        for idx, layer in enumerate(model):
            lparams = sum(p.numel() for p in layer.parameters())
            layer_summaries.append({
                "index": idx,
                "type": layer.__class__.__name__,
                "output_shape": str(tuple(layer.output_shape) if hasattr(layer, 'output_shape') else "dynamic"),
                "params": lparams,
                "trainable": all(p.requires_grad for p in layer.parameters()) if list(layer.parameters()) else True,
            })

        result = {
            "model_name": model_name,
            "model_type": model_type,
            "total_parameters": total_params,
            "total_layers": len(list(model.modules())),
            "trainable_parameters": trainable_params,
            "optimizer": optimizer_name,
            "loss": loss_name,
            "metrics": metric_names,
            "layers": layer_summaries,
            "device": device,
            "status": "created",
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: Train Model
# ══════════════════════════════════════════════════════════════════════════════

def _compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute classification accuracy from model output and targets."""
    with torch.no_grad():
        if output.ndim == 2 and output.shape[1] > 1:
            preds = output.argmax(dim=1)
        elif output.ndim == 2 and output.shape[1] == 1:
            preds = (output.squeeze() > 0.0).long()
        else:
            preds = (output > 0.5).long()
        if target.ndim == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.numel()


@tool(
    name="pt_model_train",
    description=(
        "Train a previously created PyTorch model using a manual training loop. "
        "Features: DataLoader support, mixed precision (AMP), gradient clipping, "
        "early stopping with patience, learning rate scheduling (ReduceLROnPlateau, "
        "CosineAnnealingLR, OneCycleLR), validation evaluation per epoch, "
        "and full epoch-by-epoch history tracking."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {
                "type": "string",
                "description": "Name of the model to train (must be created first with pt_model_create)",
            },
            "X_train": {
                "type": "array",
                "description": "Training features (2D array). Can be nested JSON array.",
            },
            "y_train": {
                "type": "array",
                "description": "Training labels (1D or 2D array).",
            },
            "X_val": {
                "type": "array",
                "description": "Validation features (optional).",
            },
            "y_val": {
                "type": "array",
                "description": "Validation labels (optional).",
            },
            "epochs": {"type": "integer", "description": "Number of training epochs (default: 10)"},
            "batch_size": {"type": "integer", "description": "Batch size (default: 32)"},
            "validation_split": {"type": "number", "description": "Fraction of training data for validation (0.0-1.0, default: 0.2)"},
            "learning_rate": {"type": "number", "description": "Override learning rate (default: from model creation)"},
            "early_stopping": {"type": "boolean", "description": "Enable early stopping (default: true)"},
            "patience": {"type": "integer", "description": "Early stopping patience (default: 3)"},
            "class_weights": {
                "type": "object",
                "description": "Optional class weights dict, e.g. {'0': 1.0, '1': 2.0}",
            },
            "gradient_clip": {"type": "number", "description": "Max gradient norm for clipping (0.0 = disabled, default: 0.0)"},
            "mixed_precision": {"type": "boolean", "description": "Enable AMP mixed precision training (default: true)"},
            "lr_scheduler": {
                "type": "string",
                "enum": ["none", "reduce_on_plateau", "cosine_annealing", "one_cycle"],
                "description": "Learning rate scheduler (default: reduce_on_plateau)",
            },
        },
        "required": ["model_name", "X_train", "y_train"],
    },
    risk="medium",
    category="ml_ai",
    tags=["pytorch", "training", "deep_learning", "amp", "mixed_precision"],
    timeout=300,
    version="1.0",
)
async def pt_model_train(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found. Create it first with pt_model_create."

        model: nn.Module = wrapper.model
        loss_fn = wrapper.loss_fn
        optimizer = wrapper.optimizer

        # ── Parse data ──
        X_train_np = np.array(params["X_train"], dtype=np.float32)
        y_train_np = np.array(params["y_train"], dtype=np.float32)

        # Handle label shapes for cross_entropy (needs long tensors)
        if wrapper.loss_name in ("cross_entropy", "nll"):
            if y_train_np.ndim == 2 and y_train_np.shape[1] > 1:
                y_train_tensor = torch.from_numpy(y_train_np.argmax(axis=1)).long()
            else:
                y_train_tensor = torch.from_numpy(y_train_np.flatten()).long()
        else:
            y_train_tensor = torch.from_numpy(y_train_np).float()

        X_train_tensor = torch.from_numpy(X_train_np).float()
        # Flatten if needed for dense layers
        if X_train_tensor.ndim > 2:
            X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)

        X_val_tensor = None
        y_val_tensor = None
        if params.get("X_val") and params.get("y_val"):
            X_val_np = np.array(params["X_val"], dtype=np.float32)
            y_val_np = np.array(params["y_val"], dtype=np.float32)
            X_val_tensor = torch.from_numpy(X_val_np).float()
            if X_val_tensor.ndim > 2:
                X_val_tensor = X_val_tensor.view(X_val_tensor.size(0), -1)
            if wrapper.loss_name in ("cross_entropy", "nll"):
                if y_val_np.ndim == 2 and y_val_np.shape[1] > 1:
                    y_val_tensor = torch.from_numpy(y_val_np.argmax(axis=1)).long()
                else:
                    y_val_tensor = torch.from_numpy(y_val_np.flatten()).long()
            else:
                y_val_tensor = torch.from_numpy(y_val_np).float()

        # ── Split for validation if needed ──
        val_split: float = float(params.get("validation_split", 0.2))
        if X_val_tensor is None and val_split > 0:
            split_idx = int(len(X_train_tensor) * (1 - val_split))
            X_val_tensor = X_train_tensor[split_idx:]
            y_val_tensor = y_train_tensor[split_idx:]
            X_train_tensor = X_train_tensor[:split_idx]
            y_train_tensor = y_train_tensor[:split_idx]

        # ── Training config ──
        epochs: int = int(params.get("epochs", 10))
        batch_size: int = int(params.get("batch_size", 32))
        early_stop: bool = params.get("early_stopping", True)
        patience: int = int(params.get("patience", 3))
        grad_clip: float = float(params.get("gradient_clip", 0.0))
        use_amp: bool = params.get("mixed_precision", True) and _get_device() == "cuda"
        scheduler_type: str = params.get("lr_scheduler", "reduce_on_plateau")

        lr_override = params.get("learning_rate")
        if lr_override:
            for pg in optimizer.param_groups:
                pg['lr'] = float(lr_override)

        # ── DataLoaders ──
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, drop_last=False, num_workers=0)

        val_loader = None
        if X_val_tensor is not None and y_val_tensor is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, drop_last=False, num_workers=0)

        # ── LR Scheduler ──
        scheduler = None
        if scheduler_type == "reduce_on_plateau" and val_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5,
                patience=max(patience // 2, 1), min_lr=1e-7,
            )
        elif scheduler_type == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-7,
            )
        elif scheduler_type == "one_cycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=optimizer.param_groups[0]['lr'],
                epochs=epochs, steps_per_epoch=len(train_loader),
            )

        # ── Mixed precision ──
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # ── Device ──
        device = torch.device(_get_device())
        model.to(device)

        # ── Training loop ──
        history: dict[str, list[float]] = {
            "loss": [], "accuracy": [],
            "val_loss": [], "val_accuracy": [],
            "learning_rate": [],
        }
        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        early_stopped = False

        for epoch in range(epochs):
            # --- Train ---
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(batch_x)
                        loss = loss_fn(output, batch_y)
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(batch_x)
                    loss = loss_fn(output, batch_y)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                running_correct += _compute_accuracy(output.detach(), batch_y) * batch_x.size(0)
                running_total += batch_x.size(0)

            epoch_loss = running_loss / max(running_total, 1)
            epoch_acc = running_correct / max(running_total, 1)
            history["loss"].append(round(epoch_loss, 6))
            history["accuracy"].append(round(epoch_acc, 4))
            history["learning_rate"].append(optimizer.param_groups[0]['lr'])

            # --- Validate ---
            val_loss_val = 0.0
            val_acc_val = 0.0
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_running_correct = 0
                val_running_total = 0

                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                vout = model(vx)
                                vloss = loss_fn(vout, vy)
                        else:
                            vout = model(vx)
                            vloss = loss_fn(vout, vy)
                        val_running_loss += vloss.item() * vx.size(0)
                        val_running_correct += _compute_accuracy(vout, vy) * vx.size(0)
                        val_running_total += vx.size(0)

                val_loss_val = val_running_loss / max(val_running_total, 1)
                val_acc_val = val_running_correct / max(val_running_total, 1)
                history["val_loss"].append(round(val_loss_val, 6))
                history["val_accuracy"].append(round(val_acc_val, 4))

                # Early stopping check
                if early_stop and val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                elif early_stop:
                    epochs_no_improve += 1

                # Step scheduler
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss_val)
                    else:
                        scheduler.step()

            # Early stopping break
            if early_stop and epochs_no_improve >= patience:
                early_stopped = True
                break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        model.cpu()

        # ── Store history ──
        _TRAINING_HISTORY[model_name] = history

        # ── Build result ──
        final_loss = history["loss"][-1] if history["loss"] else 0.0
        final_metrics = {"accuracy": history["accuracy"][-1] if history["accuracy"] else 0.0}
        val_results: dict[str, float] = {}
        if history["val_loss"]:
            val_results["val_loss"] = history["val_loss"][-1]
            val_results["val_accuracy"] = history["val_accuracy"][-1]

        result = {
            "model_name": model_name,
            "epochs_completed": len(history["loss"]),
            "epochs_requested": epochs,
            "final_loss": round(final_loss, 6),
            "final_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
            "validation_results": {k: round(v, 4) for k, v in val_results.items()},
            "early_stopped": early_stopped,
            "samples_trained": len(X_train_tensor),
            "device_used": str(device),
            "mixed_precision": use_amp,
            "gradient_clipping": grad_clip if grad_clip > 0 else None,
            "lr_scheduler": scheduler_type if scheduler else "none",
            "history_preview": {k: vals[-5:] for k, vals in history.items()},
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: Predict
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_predict",
    description=(
        "Run inference/predictions with a trained PyTorch model using torch.no_grad(). "
        "Accepts input data as arrays. Returns predictions, class labels (for classification), "
        "and confidence scores."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the trained model"},
            "X_input": {"type": "array", "description": "Input data for prediction (2D array)"},
            "return_probabilities": {"type": "boolean", "description": "Return raw probabilities (default: true)"},
            "batch_size": {"type": "integer", "description": "Batch size for inference (default: 32)"},
        },
        "required": ["model_name", "X_input"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "inference", "prediction", "deep_learning"],
    timeout=120,
    version="1.0",
)
async def pt_model_predict(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        model: nn.Module = wrapper.model
        X_input = np.array(params["X_input"], dtype=np.float32)
        batch_size: int = int(params.get("batch_size", 32))
        return_probs: bool = params.get("return_probabilities", True)

        model.eval()
        device = torch.device(_get_device())
        model.to(device)

        X_tensor = torch.from_numpy(X_input).float()
        if X_tensor.ndim > 2:
            X_tensor = X_tensor.view(X_tensor.size(0), -1)

        all_outputs: list[torch.Tensor] = []
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                all_outputs.append(output.cpu())

        predictions = torch.cat(all_outputs, dim=0)
        model.cpu()

        # Apply softmax for multi-class probabilities
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            prob_tensor = torch.softmax(predictions, dim=-1)
        else:
            prob_tensor = torch.sigmoid(predictions)

        prob_np = prob_tensor.numpy()
        num_samples = len(X_input)

        # Class predictions
        class_predictions = None
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            class_predictions = prob_np.argmax(axis=-1).tolist()
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            class_predictions = (prob_np.flatten() > 0.5).astype(int).tolist()

        # Confidence scores
        confidence = None
        if predictions.ndim == 2 and predictions.shape[1] > 1:
            confidence = prob_np.max(axis=-1).tolist()
        elif predictions.ndim == 2 and predictions.shape[1] == 1:
            confidence = prob_np.flatten().tolist()

        result: dict[str, Any] = {
            "model_name": model_name,
            "num_predictions": num_samples,
            "prediction_shape": list(predictions.shape),
            "class_predictions": class_predictions[:50] if class_predictions is not None else None,
            "confidence_scores": [round(c, 4) for c in confidence[:50]] if confidence is not None else None,
        }

        if return_probs and num_samples <= 20:
            result["probabilities"] = [[round(float(p), 4) for p in row] for row in prob_np]

        if confidence is not None:
            conf_arr = np.array(confidence[:num_samples])
            result["confidence_stats"] = {
                "mean": round(float(np.mean(conf_arr)), 4),
                "min": round(float(np.min(conf_arr)), 4),
                "max": round(float(np.max(conf_arr)), 4),
                "std": round(float(np.std(conf_arr)), 4),
            }

        if class_predictions is not None:
            unique, counts = np.unique(class_predictions[:num_samples], return_counts=True)
            result["class_distribution"] = {str(int(u)): int(c) for u, c in zip(unique, counts)}

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: Evaluate Model
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_evaluate",
    description=(
        "Evaluate a trained PyTorch model on test data. Returns loss, accuracy, and all "
        "configured metrics. Can compute confusion matrix, classification report, and "
        "per-class metrics using scikit-learn."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the trained model"},
            "X_test": {"type": "array", "description": "Test features"},
            "y_test": {"type": "array", "description": "Test labels"},
            "detailed_report": {"type": "boolean", "description": "Include detailed classification report (default: false)"},
            "batch_size": {"type": "integer", "description": "Batch size for evaluation (default: 32)"},
        },
        "required": ["model_name", "X_test", "y_test"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "evaluation", "metrics", "deep_learning"],
    timeout=120,
    version="1.0",
)
async def pt_model_evaluate(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        model: nn.Module = wrapper.model
        loss_fn = wrapper.loss_fn

        X_test_np = np.array(params["X_test"], dtype=np.float32)
        y_test_np = np.array(params["y_test"], dtype=np.float32)
        batch_size: int = int(params.get("batch_size", 32))
        detailed: bool = params.get("detailed_report", False)

        X_test_tensor = torch.from_numpy(X_test_np).float()
        if X_test_tensor.ndim > 2:
            X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), -1)

        if wrapper.loss_name in ("cross_entropy", "nll"):
            if y_test_np.ndim == 2 and y_test_np.shape[1] > 1:
                y_test_tensor = torch.from_numpy(y_test_np.argmax(axis=1)).long()
            else:
                y_test_tensor = torch.from_numpy(y_test_np.flatten()).long()
        else:
            y_test_tensor = torch.from_numpy(y_test_np).float()

        model.eval()
        device = torch.device(_get_device())
        model.to(device)

        dataset = TensorDataset(X_test_tensor, y_test_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        all_outputs: list[np.ndarray] = []

        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(device)
                by = by.to(device)
                output = model(bx)
                loss = loss_fn(output, by)
                total_loss += loss.item() * bx.size(0)
                total_correct += _compute_accuracy(output, by) * bx.size(0)
                total_samples += bx.size(0)
                all_outputs.append(output.cpu().numpy())

        model.cpu()

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_correct / max(total_samples, 1)
        all_predictions = np.concatenate(all_outputs, axis=0)

        eval_results: dict[str, float] = {
            "loss": round(avg_loss, 6),
            "accuracy": round(avg_acc, 4),
        }

        result: dict[str, Any] = {
            "model_name": model_name,
            "test_samples": len(X_test_np),
            "metrics": eval_results,
        }

        # Detailed classification report
        if detailed:
            sk_err = _check_sklearn()
            if sk_err:
                result["sklearn_warning"] = sk_err
            else:
                try:
                    from sklearn.metrics import (
                        classification_report, confusion_matrix,
                        precision_score, recall_score, f1_score,
                        accuracy_score,
                    )
                    if all_predictions.ndim == 2 and all_predictions.shape[1] > 1:
                        y_pred = all_predictions.argmax(axis=-1)
                    elif all_predictions.ndim == 2 and all_predictions.shape[1] == 1:
                        y_pred = (all_predictions.flatten() > 0.5).astype(int)
                    else:
                        y_pred = (all_predictions > 0.5).astype(int)

                    if y_test_np.ndim == 2 and y_test_np.shape[1] > 1:
                        y_true = y_test_np.argmax(axis=-1)
                    else:
                        y_true = y_test_np.flatten().astype(int)

                    cm = confusion_matrix(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

                    result["classification_report"] = report
                    result["confusion_matrix"] = _numpy_safe(cm)
                    result["summary_metrics"] = {
                        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                        "weighted_precision": round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                        "weighted_recall": round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                        "weighted_f1": round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
                    }
                except Exception as e:
                    result["classification_error"] = str(e)

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5: Save Model
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_save",
    description=(
        "Save a trained PyTorch model to disk. Supports full model (.pt), "
        "state_dict only (.pth), and TorchScript (.pt) formats."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the trained model"},
            "format": {
                "type": "string",
                "enum": ["full_model", "state_dict", "torchscript"],
                "description": "Save format (default: full_model)",
            },
            "custom_filename": {"type": "string", "description": "Custom filename (optional)"},
        },
        "required": ["model_name"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "save", "export", "model_persistence"],
    timeout=120,
    version="1.0",
)
async def pt_model_save(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        save_format: str = params.get("format", "full_model")
        workspace = _get_workspace()
        custom_name: str = params.get("custom_filename", model_name)
        model: nn.Module = wrapper.model

        if save_format == "full_model":
            filepath = workspace / f"{custom_name}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': wrapper.optimizer.state_dict() if wrapper.optimizer else None,
                'loss_name': wrapper.loss_name,
                'optimizer_name': wrapper.optimizer_name,
                'metrics': wrapper.metrics,
                'input_shape': wrapper.input_shape,
                'num_classes': wrapper.num_classes,
            }, str(filepath))
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            result = {
                "model_name": model_name,
                "format": "full_model",
                "path": str(filepath),
                "size_mb": round(size_mb, 2),
                "status": "saved",
            }

        elif save_format == "state_dict":
            filepath = workspace / f"{custom_name}_state_dict.pth"
            torch.save(model.state_dict(), str(filepath))
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            result = {
                "model_name": model_name,
                "format": "state_dict",
                "path": str(filepath),
                "size_mb": round(size_mb, 2),
                "status": "saved",
            }

        elif save_format == "torchscript":
            model.eval()
            filepath = workspace / f"{custom_name}_scripted.pt"
            try:
                example_input = torch.randn(1, *wrapper.input_shape) if wrapper.input_shape else torch.randn(1, 100)
                if example_input.ndim > 2:
                    example_input = example_input.view(1, -1)
                scripted = torch.jit.trace(model, example_input)
                scripted.save(str(filepath))
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                result = {
                    "model_name": model_name,
                    "format": "torchscript",
                    "path": str(filepath),
                    "size_mb": round(size_mb, 2),
                    "status": "saved",
                }
            except Exception as te:
                # Fallback to save
                filepath2 = workspace / f"{custom_name}_scripted.pt"
                torch.save(model, str(filepath2))
                result = {
                    "model_name": model_name,
                    "format": "torchscript_fallback",
                    "path": str(filepath2),
                    "note": f"TorchScript trace failed ({te}), saved full model instead",
                    "status": "saved",
                }

        # Save training history
        history = _TRAINING_HISTORY.get(model_name)
        if history:
            hist_path = workspace / f"{custom_name}_history.json"
            hist_path.write_text(json.dumps(history, indent=2))
            result["history_file"] = str(hist_path)

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6: Load Model
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_load",
    description=(
        "Load a previously saved PyTorch model from disk. Supports .pt (full model) "
        "and .pth (state_dict) formats. Registers the loaded model for use with "
        "other pt_* tools."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name to register the model under"},
            "filepath": {"type": "string", "description": "Path to the saved model file (.pt or .pth)"},
            "format": {
                "type": "string",
                "enum": ["full_model", "state_dict", "auto"],
                "description": "Format (default: auto-detect by extension)",
            },
        },
        "required": ["model_name", "filepath"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "load", "import", "model"],
    timeout=120,
    version="1.0",
)
async def pt_model_load(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        filepath = Path(params.get("filepath", ""))
        fmt: str = params.get("format", "auto")

        if not filepath.exists():
            return f"File not found: {filepath}"

        if fmt == "auto":
            if filepath.suffix == ".pth":
                fmt = "state_dict"
            else:
                fmt = "full_model"

        if fmt == "state_dict":
            checkpoint = torch.load(str(filepath), map_location='cpu', weights_only=False)
            # Try to load with model wrapper metadata if available
            # For state_dict only, we need the model architecture first
            # We store a minimal wrapper
            model_state = checkpoint if isinstance(checkpoint, dict) and 'state_dict' not in checkpoint else checkpoint
            wrapper = ModelWrapper(
                model=nn.Sequential(),  # placeholder
                loss_name="cross_entropy",
                optimizer_name="adam",
            )
            wrapper.model.load_state_dict(model_state)  # type: ignore[assignment]
            _MODEL_REGISTRY[model_name] = wrapper

            total_params, trainable_params = _count_parameters(wrapper.model)
            result = {
                "model_name": model_name,
                "format": "state_dict",
                "source": str(filepath),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "status": "loaded",
                "note": "Model architecture must be recreated before using. Load the full model format when possible.",
            }
        else:
            checkpoint = torch.load(str(filepath), map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full model checkpoint with metadata
                loss_name = checkpoint.get('loss_name', 'cross_entropy')
                optimizer_name = checkpoint.get('optimizer_name', 'adam')
                metrics = checkpoint.get('metrics', ['accuracy'])
                input_shape = checkpoint.get('input_shape')
                num_classes = checkpoint.get('num_classes', 10)

                # Create a wrapper - the model itself needs to exist
                # Store checkpoint for later model reconstruction
                wrapper = ModelWrapper(
                    model=nn.Sequential(),
                    loss_name=loss_name,
                    optimizer_name=optimizer_name,
                    metrics=metrics,
                    input_shape=input_shape,
                    num_classes=num_classes,
                )
                wrapper.model.load_state_dict(checkpoint['model_state_dict'])  # type: ignore[assignment]

                optimizer = _build_optimizer(optimizer_name, wrapper.model.parameters())
                if checkpoint.get('optimizer_state_dict'):
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                wrapper.optimizer = optimizer

                _MODEL_REGISTRY[model_name] = wrapper

                total_params, trainable_params = _count_parameters(wrapper.model)
                result = {
                    "model_name": model_name,
                    "format": "full_model",
                    "source": str(filepath),
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "optimizer": optimizer_name,
                    "loss": loss_name,
                    "status": "loaded",
                }
            elif isinstance(checkpoint, nn.Module):
                # Directly saved model
                wrapper = ModelWrapper(model=checkpoint)
                _MODEL_REGISTRY[model_name] = wrapper
                total_params, trainable_params = _count_parameters(wrapper.model)
                result = {
                    "model_name": model_name,
                    "format": "full_model",
                    "source": str(filepath),
                    "total_parameters": total_params,
                    "status": "loaded",
                }
            else:
                return f"Unrecognized checkpoint format in {filepath}"

        # Load training history
        stem = filepath.stem.replace("_state_dict", "").replace("_scripted", "")
        hist_path = filepath.parent / f"{stem}_history.json"
        if hist_path.exists():
            _TRAINING_HISTORY[model_name] = json.loads(hist_path.read_text())
            result["training_history_loaded"] = True

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7: Model Info
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_info",
    description=(
        "Get detailed information about a PyTorch model: architecture, parameter count, "
        "layer details, device, optimizer info, and training history if available."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the model"},
        },
        "required": ["model_name"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "info", "architecture", "model"],
    timeout=30,
    version="1.0",
)
async def pt_model_info(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            registered = list(_MODEL_REGISTRY.keys())
            return f"Model '{model_name}' not found. Registered models: {registered}"

        model: nn.Module = wrapper.model

        layer_details: list[dict] = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Sequential):
                continue
            params_count = sum(p.numel() for p in layer.parameters())
            trainable = all(p.requires_grad for p in layer.parameters()) if list(layer.parameters()) else True
            info: dict[str, Any] = {
                "name": name,
                "type": layer.__class__.__name__,
                "param_count": params_count,
                "trainable": trainable,
            }
            # Extra info for specific layers
            if hasattr(layer, 'out_features'):
                info["out_features"] = layer.out_features  # type: ignore[attr-defined]
            if hasattr(layer, 'out_channels'):
                info["out_channels"] = layer.out_channels  # type: ignore[attr-defined]
            if hasattr(layer, 'kernel_size'):
                info["kernel_size"] = tuple(layer.kernel_size) if hasattr(layer.kernel_size, '__iter__') else layer.kernel_size  # type: ignore[attr-defined]
            layer_details.append(info)

        total_params, trainable_params = _count_parameters(model)
        non_trainable = total_params - trainable_params

        # Optimizer info
        opt_info: dict[str, Any] = {
            "type": wrapper.optimizer_name,
        }
        if wrapper.optimizer:
            opt_info["learning_rate"] = wrapper.optimizer.param_groups[0].get('lr', 'N/A')

        # Training history
        history = _TRAINING_HISTORY.get(model_name)

        result: dict[str, Any] = {
            "model_name": model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable,
            "num_layers": len(list(model.modules())),
            "layers": layer_details,
            "optimizer": opt_info,
            "loss": wrapper.loss_name,
            "metrics": wrapper.metrics,
            "device": wrapper.device or _get_device(),
            "has_training_history": history is not None,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() if _TORCH_AVAILABLE else False,
        }

        if history:
            result["history_summary"] = {
                k: f"{len(v)} epochs, final: {v[-1]:.4f}" for k, v in history.items()
            }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 8: Data Preprocess
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_data_preprocess",
    description=(
        "Preprocess data for PyTorch ML: normalize, standardize, encode labels, split datasets, "
        "reshape tensors, handle missing values, and create DataLoader pipelines."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["normalize", "standardize", "train_test_split", "label_encode",
                         "one_hot_encode", "reshape", "handle_missing", "create_dataloader"],
                "description": "Preprocessing operation",
            },
            "data": {"type": "array", "description": "Input data (1D or 2D array)"},
            "labels": {"type": "array", "description": "Labels for supervised operations"},
            "test_size": {"type": "number", "description": "Test split fraction (default: 0.2)"},
            "random_state": {"type": "integer", "description": "Random seed (default: 42)"},
            "target_shape": {"type": "array", "items": {"type": "integer"}, "description": "Target shape for reshape"},
            "num_classes": {"type": "integer", "description": "Number of classes for one-hot encoding"},
            "batch_size": {"type": "integer", "description": "Batch size for DataLoader (default: 32)"},
            "shuffle": {"type": "boolean", "description": "Shuffle data (default: true)"},
        },
        "required": ["operation", "data"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "preprocessing", "data", "dataloader", "pipeline"],
    timeout=120,
    version="1.0",
)
async def pt_data_preprocess(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        operation: str = params.get("operation", "normalize")
        data = np.array(params["data"], dtype=np.float64)

        if operation == "normalize":
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val > 1e-10:
                normalized = (data - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(data)
            result = {
                "operation": "normalize",
                "shape": list(data.shape),
                "min_before": round(float(min_val), 6),
                "max_before": round(float(max_val), 6),
                "min_after": round(float(np.min(normalized)), 6),
                "max_after": round(float(np.max(normalized)), 6),
                "data_preview": _numpy_safe(normalized[:10].tolist()),
            }

        elif operation == "standardize":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std_safe = np.where(std > 1e-10, std, 1.0)
            standardized = (data - mean) / std_safe
            result = {
                "operation": "standardize",
                "shape": list(data.shape),
                "mean": _numpy_safe(mean[:5].tolist()),
                "std": _numpy_safe(std_safe[:5].tolist()),
                "data_preview": _numpy_safe(standardized[:10].tolist()),
            }

        elif operation == "train_test_split":
            sk_err = _check_sklearn()
            if sk_err:
                return sk_err
            from sklearn.model_selection import train_test_split
            labels = np.array(params.get("labels", []), dtype=np.float64) if params.get("labels") else None
            test_size = float(params.get("test_size", 0.2))
            random_state = int(params.get("random_state", 42))

            if labels is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    data, labels, test_size=test_size, random_state=random_state,
                    stratify=labels if test_size < 0.5 and len(np.unique(labels)) > 1 else None,
                )
                result = {
                    "operation": "train_test_split",
                    "X_train_shape": list(X_train.shape),
                    "X_test_shape": list(X_test.shape),
                    "y_train_shape": list(y_train.shape),
                    "y_test_shape": list(y_test.shape),
                    "test_size": test_size,
                }
            else:
                X_train, X_test = train_test_split(data, test_size=test_size, random_state=random_state)
                result = {
                    "operation": "train_test_split",
                    "X_train_shape": list(X_train.shape),
                    "X_test_shape": list(X_test.shape),
                    "test_size": test_size,
                }

        elif operation == "label_encode":
            sk_err = _check_sklearn()
            if sk_err:
                return sk_err
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            encoded = le.fit_transform(data.flatten().astype(str))
            result = {
                "operation": "label_encode",
                "classes": le.classes_.tolist(),
                "encoded_preview": encoded[:20].tolist(),
                "num_classes": len(le.classes_),
            }

        elif operation == "one_hot_encode":
            labels_arr = np.array(params.get("labels", params.get("data", [])))
            num_classes = int(params.get("num_classes", len(np.unique(labels_arr))))
            encoded = np.eye(num_classes)[labels_arr.astype(int)]
            result = {
                "operation": "one_hot_encode",
                "input_shape": list(labels_arr.shape),
                "output_shape": list(encoded.shape),
                "num_classes": num_classes,
                "preview": _numpy_safe(encoded[:5].tolist()),
            }

        elif operation == "reshape":
            target_shape = tuple(params.get("target_shape", [-1]))
            reshaped = data.reshape(target_shape)
            result = {
                "operation": "reshape",
                "input_shape": list(data.shape),
                "output_shape": list(reshaped.shape),
                "total_elements": int(data.size),
            }

        elif operation == "handle_missing":
            clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            nan_count = int(np.isnan(data).sum()) if not np.issubdtype(data.dtype, np.integer) else 0
            result = {
                "operation": "handle_missing",
                "input_shape": list(data.shape),
                "missing_values_filled": nan_count,
                "data_preview": _numpy_safe(clean[:5].tolist()),
            }

        elif operation == "create_dataloader":
            batch_size = int(params.get("batch_size", 32))
            shuffle = params.get("shuffle", True)
            labels_np = np.array(params.get("labels", []), dtype=np.float32) if params.get("labels") else None

            X_tensor = torch.from_numpy(data.astype(np.float32))
            if X_tensor.ndim > 2:
                X_tensor = X_tensor.view(X_tensor.size(0), -1)

            if labels_np is not None:
                y_tensor = torch.from_numpy(labels_np)
                dataset = TensorDataset(X_tensor, y_tensor)
                num_samples = len(X_tensor)
            else:
                dataset = TensorDataset(X_tensor)
                num_samples = len(X_tensor)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            num_batches = len(loader)

            result = {
                "operation": "create_dataloader",
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_samples": num_samples,
                "num_batches": num_batches,
                "feature_shape": list(X_tensor.shape[1:]) if X_tensor.ndim > 1 else [X_tensor.shape[-1]],
                "has_labels": labels_np is not None,
            }

        else:
            return f"Unknown operation: {operation}"

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 9: Data Augment
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_data_augment",
    description=(
        "Apply data augmentation to image data using torchvision.transforms. "
        "Supports random horizontal/vertical flip, rotation, affine transforms, "
        "color jitter, grayscale, random erasing, Gaussian blur, and normalization. "
        "Returns the transforms pipeline configuration."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "data_type": {
                "type": "string",
                "enum": ["image", "text"],
                "description": "Type of data to augment",
            },
            "augmentations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of augmentations. Image: random_flip_h, random_flip_v, "
                             "random_rotation, random_affine, color_jitter, random_grayscale, "
                             "random_erasing, gaussian_blur, random_crop, center_crop, "
                             "normalize, random_perspective.",
            },
            "num_augmented": {"type": "integer", "description": "Number of augmented samples to generate (default: 5)"},
            "rotation_range": {"type": "number", "description": "Rotation range in degrees (default: 20)"},
            "brightness": {"type": "number", "description": "Brightness jitter factor (default: 0.2)"},
            "contrast": {"type": "number", "description": "Contrast jitter factor (default: 0.2)"},
        },
        "required": ["data_type", "augmentations"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "data_augmentation", "torchvision", "transforms", "images"],
    timeout=60,
    version="1.0",
)
async def pt_data_augment(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        data_type: str = params.get("data_type", "image")
        augmentations: list[str] = params.get("augmentations", ["random_flip_h", "random_rotation"])
        rotation: float = float(params.get("rotation_range", 20))
        brightness: float = float(params.get("brightness", 0.2))
        contrast: float = float(params.get("contrast", 0.2))

        if data_type == "image":
            tv_err = _check_torchvision()
            if tv_err:
                return tv_err

            transforms_list: list[Any] = []
            aug_descriptions: list[dict] = []

            for aug in augmentations:
                if aug == "random_flip_h":
                    transforms_list.append(T.RandomHorizontalFlip(p=0.5))
                    aug_descriptions.append({"name": "RandomHorizontalFlip", "params": {"p": 0.5}})
                elif aug == "random_flip_v":
                    transforms_list.append(T.RandomVerticalFlip(p=0.5))
                    aug_descriptions.append({"name": "RandomVerticalFlip", "params": {"p": 0.5}})
                elif aug == "random_rotation":
                    transforms_list.append(T.RandomRotation(degrees=rotation))
                    aug_descriptions.append({"name": "RandomRotation", "params": {"degrees": rotation}})
                elif aug == "random_affine":
                    transforms_list.append(T.RandomAffine(degrees=rotation, translate=(0.1, 0.1)))
                    aug_descriptions.append({"name": "RandomAffine", "params": {"degrees": rotation, "translate": [0.1, 0.1]}})
                elif aug == "color_jitter":
                    transforms_list.append(T.ColorJitter(brightness=brightness, contrast=contrast,
                                                        saturation=brightness, hue=brightness * 0.1))
                    aug_descriptions.append({"name": "ColorJitter", "params": {
                        "brightness": brightness, "contrast": contrast}})
                elif aug == "random_grayscale":
                    transforms_list.append(T.RandomGrayscale(p=0.1))
                    aug_descriptions.append({"name": "RandomGrayscale", "params": {"p": 0.1}})
                elif aug == "random_erasing":
                    transforms_list.append(T.RandomErasing(p=0.5))
                    aug_descriptions.append({"name": "RandomErasing", "params": {"p": 0.5}})
                elif aug == "gaussian_blur":
                    transforms_list.append(T.GaussianBlur(kernel_size=3))
                    aug_descriptions.append({"name": "GaussianBlur", "params": {"kernel_size": 3}})
                elif aug == "random_crop":
                    transforms_list.append(T.RandomResizedCrop(size=224, scale=(0.8, 1.0)))
                    aug_descriptions.append({"name": "RandomResizedCrop", "params": {"size": 224}})
                elif aug == "center_crop":
                    transforms_list.append(T.CenterCrop(size=224))
                    aug_descriptions.append({"name": "CenterCrop", "params": {"size": 224}})
                elif aug == "normalize":
                    transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225]))
                    aug_descriptions.append({"name": "Normalize", "params": {
                        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}})
                elif aug == "random_perspective":
                    transforms_list.append(T.RandomPerspective(distortion_scale=0.2, p=0.5))
                    aug_descriptions.append({"name": "RandomPerspective", "params": {"distortion_scale": 0.2}})

            pipeline = T.Compose(transforms_list) if transforms_list else None

            result = {
                "data_type": "image",
                "augmentations": aug_descriptions,
                "num_transforms": len(transforms_list),
                "pipeline_built": pipeline is not None,
                "pipeline_repr": str(pipeline) if pipeline else None,
            }

        elif data_type == "text":
            aug_map = {
                "synonym_replace": "Replace words with synonyms",
                "random_delete": "Randomly delete words with probability p",
                "random_swap": "Randomly swap two words in sentence",
                "random_insert": "Insert random word at random position",
            }
            descriptions = {a: aug_map.get(a, "Unknown") for a in augmentations}
            result = {
                "data_type": "text",
                "augmentations": descriptions,
                "note": "Text augmentation is applied during data pipeline creation. "
                        "Use pt_data_preprocess with create_dataloader for batch processing.",
            }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 10: Transfer Learning
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_transfer_learning",
    description=(
        "Set up a transfer learning model using PyTorch torchvision pretrained backbones. "
        "Supports resnet18, resnet50, mobilenet_v2, efficientnet_b0, vgg16, densenet121, "
        "wide_resnet50_2, inception_v3. Configures trainable layers, custom classification "
        "head, and fine-tuning settings."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name for the transfer learning model"},
            "backbone": {
                "type": "string",
                "enum": ["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0",
                         "vgg16", "densenet121", "wide_resnet50_2", "inception_v3", "custom"],
                "description": "Pretrained backbone model",
            },
            "input_shape": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Input shape [C, H, W] (default: [3, 224, 224])",
            },
            "num_classes": {"type": "integer", "description": "Number of output classes (default: 10)"},
            "fine_tune_layers": {"type": "integer", "description": "Number of top layers to unfreeze (0=all frozen, -1=all trainable)"},
            "dropout_rate": {"type": "number", "description": "Dropout rate in classification head (default: 0.5)"},
            "pretrained": {"type": "boolean", "description": "Use pretrained weights (default: true)"},
            "learning_rate": {"type": "number", "description": "Learning rate (default: 0.001)"},
        },
        "required": ["model_name", "backbone", "num_classes"],
    },
    risk="medium",
    category="ml_ai",
    tags=["pytorch", "transfer_learning", "pretrained", "torchvision", "deep_learning", "cv"],
    timeout=120,
    version="1.0",
)
async def pt_transfer_learning(params: dict) -> str:
    err = _check_available()
    if err:
        return err
    tv_err = _check_torchvision()
    if tv_err:
        return tv_err

    try:
        model_name: str = params.get("model_name", "transfer_model")
        backbone_name: str = params.get("backbone", "resnet18")
        input_shape: list[int] = params.get("input_shape", [3, 224, 224])
        num_classes: int = int(params.get("num_classes", 10))
        fine_tune_layers: int = int(params.get("fine_tune_layers", 0))
        dropout_rate: float = float(params.get("dropout_rate", 0.5))
        pretrained: bool = params.get("pretrained", True)
        lr: float = float(params.get("learning_rate", 0.001))

        # ── Backbone map ──
        backbone_map: dict[str, Any] = {
            "resnet18": tv_models.resnet18,
            "resnet50": tv_models.resnet50,
            "mobilenet_v2": tv_models.mobilenet_v2,
            "efficientnet_b0": tv_models.efficientnet_b0,
            "vgg16": tv_models.vgg16,
            "densenet121": tv_models.densenet121,
            "wide_resnet50_2": tv_models.wide_resnet50_2,
            "inception_v3": tv_models.inception_v3,
        }

        if backbone_name not in backbone_map:
            return f"Unknown backbone: {backbone_name}. Available: {list(backbone_map.keys())}"

        weights_arg = "IMAGENET1K_V1" if pretrained else None
        try:
            base_model = backbone_map[backbone_name](weights=weights_arg)
        except Exception as load_err:
            try:
                base_model = backbone_map[backbone_name](pretrained=pretrained)
            except Exception:
                return f"Failed to load {backbone_name}: {load_err}"

        # ── Freeze layers ──
        if fine_tune_layers == 0:
            for param in base_model.parameters():
                param.requires_grad = False
        elif fine_tune_layers > 0:
            # Freeze all first, then unfreeze top N layers
            for param in base_model.parameters():
                param.requires_grad = False
            layers_list = list(base_model.children())
            for layer in layers_list[-fine_tune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        # fine_tune_layers == -1 means all trainable (default)

        # ── Replace classification head ──
        # Get the in_features of the final layer
        if hasattr(base_model, 'fc'):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif hasattr(base_model, 'classifier'):
            if isinstance(base_model.classifier, nn.Sequential):
                in_features = base_model.classifier[-1].in_features
            else:
                in_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, 'head'):
            in_features = base_model.head.in_features
            base_model.head = nn.Identity()
        else:
            # Fallback: try to get from last Linear layer
            in_features = 512
            for m in reversed(list(base_model.modules())):
                if isinstance(m, nn.Linear):
                    in_features = m.out_features
                    break

        # Build new model with classification head
        class TransferModel(nn.Module):
            def __init__(self, backbone: nn.Module, in_feat: int, n_classes: int, drop: float) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Dropout(p=drop),
                    nn.Linear(in_feat, 256),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=drop * 0.5),
                    nn.Linear(256, n_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                if features.ndim > 2:
                    features = features.mean(dim=tuple(range(1, features.ndim)))
                return self.head(features)

        model = TransferModel(base_model, in_features, num_classes, dropout_rate)

        # ── Optimizer ──
        optimizer = _build_optimizer("adam", model.parameters(), lr=lr)
        loss_name = "cross_entropy" if num_classes > 2 else ("binary_crossentropy" if num_classes == 1 else "mse")
        loss_fn = _build_loss(loss_name)

        # ── Register ──
        wrapper = ModelWrapper(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_name=loss_name,
            optimizer_name="adam",
            input_shape=tuple(input_shape),
            num_classes=num_classes,
        )
        _MODEL_REGISTRY[model_name] = wrapper

        total_params, trainable_params = _count_parameters(model)
        frozen_params = total_params - trainable_params

        result = {
            "model_name": model_name,
            "backbone": backbone_name,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "fine_tune_layers": fine_tune_layers,
            "pretrained": pretrained,
            "learning_rate": lr,
            "loss": loss_name,
            "classification_head": ["dropout", "linear(256)", "relu", "batch_norm", "dropout", f"linear({num_classes})"],
            "status": "created",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 11: Model Visualize
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_visualize",
    description=(
        "Generate a text-based visualization of PyTorch model architecture with layer "
        "connectivity, parameter counts, and training curves (ASCII art). Provides "
        "a structured summary of the model topology."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the model"},
            "include_training_curves": {"type": "boolean", "description": "Include ASCII training curves (default: true)"},
            "max_curve_points": {"type": "integer", "description": "Max points for curves (default: 30)"},
        },
        "required": ["model_name"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "visualization", "architecture", "plot"],
    timeout=30,
    version="1.0",
)
async def pt_model_visualize(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        model: nn.Module = wrapper.model

        # ── Build architecture diagram ──
        lines: list[str] = [
            f"{'=' * 70}",
            f" PyTorch Model: {model_name}",
            f"{'=' * 70}",
            f"  Type: {model.__class__.__name__}",
            f"  Device: {wrapper.device or _get_device()}",
            "",
        ]

        total_params = 0
        idx = 0
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Sequential, ModelWrapper)):
                continue
            params_count = sum(p.numel() for p in layer.parameters())
            total_params += params_count
            trainable = "T" if all(p.requires_grad for p in layer.parameters()) else "F"
            ltype = layer.__class__.__name__[:25]

            extra = ""
            if hasattr(layer, 'out_features'):
                extra = f" out={layer.out_features}"  # type: ignore[attr-defined]
            elif hasattr(layer, 'out_channels'):
                extra = f" out_ch={layer.out_channels}"  # type: ignore[attr-defined]
            elif hasattr(layer, 'kernel_size'):
                extra = f" k={layer.kernel_size}"  # type: ignore[attr-defined]

            lines.append(f"  [{idx:2d}] {name[:40].ljust(40)} | {ltype:25s} | {trainable} | params: {params_count:>12,}{extra}")
            if idx < sum(1 for _ in model.named_modules()) - 2:
                lines.append(f"       {'|':^40}      |")
            idx += 1

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params

        lines.append(f"\n{'=' * 70}")
        lines.append(f"  Total parameters:  {total_params:>15,}")
        lines.append(f"  Trainable:         {trainable_params:>15,}")
        lines.append(f"  Non-trainable:     {non_trainable:>15,}")
        lines.append(f"  Model size (est):  {total_params * 4 / (1024**2):>13.2f} MB (float32)")
        lines.append(f"{'=' * 70}")

        # ── Training curves ──
        if params.get("include_training_curves", True):
            history = _TRAINING_HISTORY.get(model_name)
            if history:
                max_pts: int = int(params.get("max_curve_points", 30))
                n_epochs = len(history.get("loss", []))
                lines.append(f"\n--- Training Curves ({n_epochs} epochs) ---\n")
                for metric_name, values in history.items():
                    if metric_name == "learning_rate":
                        continue
                    if len(values) < 2:
                        continue
                    step = max(1, len(values) // max_pts)
                    sampled = values[::step]
                    if len(sampled) < 2:
                        continue

                    min_v, max_v = min(sampled), max(sampled)
                    v_range = max_v - min_v if max_v - min_v > 1e-10 else 1.0
                    width = 40

                    curve: list[str] = []
                    for v in sampled:
                        bar_len = int(((v - min_v) / v_range) * width)
                        curve.append(f"{'#' * bar_len}{'-' * (width - bar_len)}")

                    lines.append(f"  {metric_name}:")
                    lines.append(f"  max: {max_v:.4f} | min: {min_v:.4f}")
                    for j, bar in enumerate(curve[:15]):
                        epoch_num = j * step
                        val_str = f"{sampled[j]:.4f}" if j < len(sampled) else "  -  "
                        lines.append(f"  E{epoch_num:3d} [{val_str}] |{bar}|")

        return "\n".join(lines)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 12: Hyperparameter Tune
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_hyperparameter_tune",
    description=(
        "Perform hyperparameter search for a PyTorch model. Supports grid search and "
        "random search. Tests different combinations of learning rates, batch sizes, "
        "weight decay, and gradient clipping values. Trains a copy of the model for "
        "a few epochs per trial and reports the best configuration."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Base model name"},
            "X_train": {"type": "array", "description": "Training features"},
            "y_train": {"type": "array", "description": "Training labels"},
            "search_type": {
                "type": "string",
                "enum": ["random", "grid"],
                "description": "Search strategy (default: random)",
            },
            "num_trials": {"type": "integer", "description": "Number of trials (default: 5)"},
            "epochs_per_trial": {"type": "integer", "description": "Epochs per trial (default: 3)"},
            "param_space": {
                "type": "object",
                "description": "Parameter search space. Keys: learning_rate, batch_size, weight_decay, "
                             "gradient_clip. Values: arrays of values to try.",
            },
        },
        "required": ["model_name", "X_train", "y_train"],
    },
    risk="medium",
    category="ml_ai",
    tags=["pytorch", "hyperparameter", "tuning", "optimization", "grid_search", "random_search"],
    timeout=600,
    version="1.0",
)
async def pt_hyperparameter_tune(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        X_train_np = np.array(params["X_train"], dtype=np.float32)
        y_train_np = np.array(params["y_train"], dtype=np.float32)
        search_type: str = params.get("search_type", "random")
        num_trials: int = int(params.get("num_trials", 5))
        epochs: int = int(params.get("epochs_per_trial", 3))

        # Prepare tensors
        X_tensor = torch.from_numpy(X_train_np).float()
        if X_tensor.ndim > 2:
            X_tensor = X_tensor.view(X_tensor.size(0), -1)

        if wrapper.loss_name in ("cross_entropy", "nll"):
            if y_train_np.ndim == 2 and y_train_np.shape[1] > 1:
                y_tensor = torch.from_numpy(y_train_np.argmax(axis=1)).long()
            else:
                y_tensor = torch.from_numpy(y_train_np.flatten()).long()
        else:
            y_tensor = torch.from_numpy(y_train_np).float()

        # Default parameter space
        default_space: dict[str, list] = {
            "learning_rate": [0.001, 0.0005, 0.0001, 0.01, 0.005],
            "batch_size": [16, 32, 64, 128],
        }
        param_space: dict[str, list] = params.get("param_space", default_space)

        # Generate trials
        trials: list[dict[str, Any]] = []
        if search_type == "grid":
            keys = list(param_space.keys())
            value_lists = list(param_space.values())
            for combo in itertools.product(*value_lists):
                trials.append(dict(zip(keys, combo)))
            trials = trials[:num_trials]
        else:
            for _ in range(num_trials):
                trial: dict[str, Any] = {}
                for k, v in param_space.items():
                    trial[k] = random.choice(v) if isinstance(v, list) else v
                trials.append(trial)

        # Run trials
        results: list[dict[str, Any]] = []
        best_score = -float('inf')
        best_params: dict[str, Any] | None = None

        split_idx = int(len(X_tensor) * 0.8)
        X_train_t, X_val_t = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train_t, y_val_t = y_tensor[:split_idx], y_tensor[split_idx:]

        for i, trial_params in enumerate(trials):
            try:
                # Clone model architecture
                model_copy = copy.deepcopy(wrapper.model)
                lr_val = float(trial_params.get("learning_rate", 0.001))
                batch_size = int(trial_params.get("batch_size", 32))

                opt = _build_optimizer(wrapper.optimizer_name, model_copy.parameters(), lr=lr_val)
                loss_fn = wrapper.loss_fn

                train_ds = TensorDataset(X_train_t, y_train_t)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_ds = TensorDataset(X_val_t, y_val_t)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                device = torch.device(_get_device())
                model_copy.to(device)

                best_val_loss = float('inf')
                for _epoch in range(epochs):
                    model_copy.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        opt.zero_grad()
                        out = model_copy(bx)
                        loss = loss_fn(out, by)
                        loss.backward()
                        opt.step()

                    model_copy.eval()
                    val_loss_sum = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            vout = model_copy(vx)
                            vloss = loss_fn(vout, vy)
                            val_loss_sum += vloss.item() * vx.size(0)
                            val_count += vx.size(0)
                    val_loss = val_loss_sum / max(val_count, 1)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                score = -best_val_loss
                results.append({
                    "trial": i + 1,
                    "params": trial_params,
                    "val_loss": round(float(best_val_loss), 4),
                    "score": round(float(score), 4),
                })

                if score > best_score:
                    best_score = score
                    best_params = trial_params
                    _MODEL_REGISTRY[f"{model_name}_best_tuned"] = ModelWrapper(
                        model=model_copy,
                        optimizer=opt,
                        loss_fn=loss_fn,
                        loss_name=wrapper.loss_name,
                        optimizer_name=wrapper.optimizer_name,
                        metrics=wrapper.metrics,
                        input_shape=wrapper.input_shape,
                        num_classes=wrapper.num_classes,
                    )

                model_copy.cpu()

            except Exception as trial_err:
                results.append({
                    "trial": i + 1,
                    "params": trial_params,
                    "error": str(trial_err),
                })

        results.sort(key=lambda x: x.get("score", -999), reverse=True)

        result = {
            "model_name": model_name,
            "search_type": search_type,
            "total_trials": num_trials,
            "successful_trials": len([r for r in results if "error" not in r]),
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "best_model_name": f"{model_name}_best_tuned",
            "trials": results,
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 13: Text Classification Pipeline
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_text_classification",
    description=(
        "End-to-end text classification pipeline using PyTorch. Builds a text "
        "classification model with embedding layer, processes text data, trains, and "
        "evaluates. Supports architectures: simple_rnn, bidirectional_lstm, cnn_text, "
        "transformer. Suitable for sentiment analysis, topic classification, intent detection."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Model name"},
            "texts": {"type": "array", "items": {"type": "string"}, "description": "Training texts"},
            "labels": {"type": "array", "items": {"type": "integer"}, "description": "Training labels (integers)"},
            "vocab_size": {"type": "integer", "description": "Vocabulary size (default: 10000)"},
            "max_length": {"type": "integer", "description": "Max sequence length (default: 200)"},
            "embedding_dim": {"type": "integer", "description": "Embedding dimension (default: 128)"},
            "num_classes": {"type": "integer", "description": "Number of classes (default: auto-detect)"},
            "epochs": {"type": "integer", "description": "Training epochs (default: 5)"},
            "model_architecture": {
                "type": "string",
                "enum": ["simple_rnn", "bidirectional_lstm", "cnn_text", "transformer"],
                "description": "Model architecture (default: bidirectional_lstm)",
            },
        },
        "required": ["model_name", "texts", "labels"],
    },
    risk="medium",
    category="ml_ai",
    tags=["pytorch", "nlp", "text_classification", "sentiment", "deep_learning"],
    timeout=300,
    version="1.0",
)
async def pt_text_classification(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "text_classifier")
        texts: list[str] = params.get("texts", [])
        labels_np = np.array(params.get("labels", []), dtype=np.int64)
        vocab_size: int = int(params.get("vocab_size", 10000))
        max_length: int = int(params.get("max_length", 200))
        embedding_dim: int = int(params.get("embedding_dim", 128))
        num_classes: int = int(params.get("num_classes", len(np.unique(labels_np))))
        epochs: int = int(params.get("epochs", 5))
        architecture: str = params.get("model_architecture", "bidirectional_lstm")

        if len(texts) == 0:
            return "No training texts provided."

        # ── Tokenize ──
        sequences, word_index = _simple_tokenizer(texts, vocab_size, max_length)
        X_tensor = torch.from_numpy(np.array(sequences, dtype=np.int64)).long()
        y_tensor = torch.from_numpy(labels_np).long()

        # ── Define model ──
        class TextClassifier(nn.Module):
            def __init__(self, vocab_sz: int, embed_sz: int, max_len: int,
                         n_classes: int, arch: str) -> None:
                super().__init__()
                self.arch = arch
                self.embedding = nn.Embedding(vocab_sz, embed_sz, padding_idx=0)

                if arch == "simple_rnn":
                    self.rnn = nn.RNN(embed_sz, 64, batch_first=True)
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(64, n_classes),
                    )
                elif arch == "bidirectional_lstm":
                    self.lstm = nn.LSTM(embed_sz, 64, batch_first=True,
                                        bidirectional=True, dropout=0.1)
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(64 * 2, 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Linear(64, n_classes),
                    )
                elif arch == "cnn_text":
                    self.conv = nn.Conv1d(embed_sz, 128, kernel_size=5, padding=2)
                    self.classifier = nn.Sequential(
                        nn.AdaptiveMaxPool1d(1),
                        nn.Flatten(),
                        nn.Dropout(0.3),
                        nn.Linear(128, n_classes),
                    )
                elif arch == "transformer":
                    self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_sz) * 0.02)
                    self.attn = nn.MultiheadAttention(embed_sz, num_heads=4, batch_first=True)
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(embed_sz),
                        nn.Dropout(0.3),
                        nn.Linear(embed_sz, n_classes),
                    )
                else:
                    self.classifier = nn.Linear(embed_sz, n_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embedding(x)  # (B, L, E)
                if self.arch == "simple_rnn":
                    _, hidden = self.rnn(x)
                    out = hidden[-1]
                elif self.arch == "bidirectional_lstm":
                    _, (hidden, _) = self.lstm(x)
                    out = torch.cat([hidden[-2], hidden[-1]], dim=1)
                elif self.arch == "cnn_text":
                    x = x.transpose(1, 2)  # (B, E, L)
                    x = torch.relu(self.conv(x))
                    out = x
                elif self.arch == "transformer":
                    x = x + self.pos_embed[:, :x.size(1), :]
                    attn_out, _ = self.attn(x, x, x)
                    out = attn_out.mean(dim=1)
                else:
                    out = x.mean(dim=1)
                return self.classifier(out)

        model = TextClassifier(vocab_size, embedding_dim, max_length, num_classes, architecture)
        total_params, trainable_params = _count_parameters(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        device = torch.device(_get_device())
        model.to(device)

        # Split
        split_idx = int(len(X_tensor) * 0.8)
        X_tr, X_va = X_tensor[:split_idx].to(device), X_tensor[split_idx:].to(device)
        y_tr, y_va = y_tensor[:split_idx].to(device), y_tensor[split_idx:].to(device)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_ds = TensorDataset(X_va, y_va)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # Train
        hist: dict[str, list[float]] = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = loss_fn(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * bx.size(0)
                correct += (out.argmax(dim=1) == by).sum().item()
                total += bx.size(0)

            epoch_loss = running_loss / max(total, 1)
            epoch_acc = correct / max(total, 1)
            hist["loss"].append(round(epoch_loss, 6))
            hist["accuracy"].append(round(epoch_acc, 4))

            # Validate
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vout = model(vx)
                    vloss = loss_fn(vout, vy)
                    val_loss_sum += vloss.item() * vx.size(0)
                    val_correct += (vout.argmax(dim=1) == vy).sum().item()
                    val_total += vx.size(0)

            val_loss_val = val_loss_sum / max(val_total, 1)
            val_acc_val = val_correct / max(val_total, 1)
            hist["val_loss"].append(round(val_loss_val, 6))
            hist["val_accuracy"].append(round(val_acc_val, 4))

            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 2:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.cpu()

        _MODEL_REGISTRY[model_name] = ModelWrapper(
            model=model, optimizer=optimizer, loss_fn=loss_fn,
            loss_name="cross_entropy", optimizer_name="adam",
            input_shape=(max_length,), num_classes=num_classes,
        )
        _TRAINING_HISTORY[model_name] = hist

        result = {
            "model_name": model_name,
            "architecture": architecture,
            "vocab_size": vocab_size,
            "tokenizer_word_count": len(word_index),
            "num_classes": num_classes,
            "training_samples": len(texts),
            "epochs_completed": len(hist["loss"]),
            "final_accuracy": round(hist["accuracy"][-1], 4),
            "final_val_accuracy": round(hist["val_accuracy"][-1], 4),
            "total_parameters": total_params,
            "status": "trained",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 14: Image Classification Pipeline
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_image_classification",
    description=(
        "End-to-end image classification pipeline using PyTorch. Builds a CNN model "
        "for image data with architectures: simple_cnn, deep_cnn, resnet_style. "
        "Supports data augmentation via torchvision.transforms. The model is created "
        "and registered for training with pt_model_train."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Model name"},
            "num_classes": {"type": "integer", "description": "Number of image classes"},
            "image_size": {"type": "array", "items": {"type": "integer"}, "description": "Image dimensions [C, H, W] (default: [3, 64, 64])"},
            "architecture": {
                "type": "string",
                "enum": ["simple_cnn", "deep_cnn", "resnet_style"],
                "description": "Architecture type (default: simple_cnn)",
            },
            "data_augmentation": {"type": "boolean", "description": "Enable data augmentation (default: true)"},
        },
        "required": ["model_name", "num_classes"],
    },
    risk="medium",
    category="ml_ai",
    tags=["pytorch", "computer_vision", "image_classification", "cnn", "deep_learning"],
    timeout=300,
    version="1.0",
)
async def pt_image_classification(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "image_classifier")
        num_classes: int = int(params.get("num_classes", 10))
        image_size: list[int] = params.get("image_size", [3, 64, 64])
        architecture: str = params.get("architecture", "simple_cnn")
        use_aug: bool = params.get("data_augmentation", True)

        channels = image_size[0] if len(image_size) >= 3 else 3
        h = image_size[1] if len(image_size) >= 2 else 64

        if architecture == "simple_cnn":
            model = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * (h // 8) * (h // 8), 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        elif architecture == "deep_cnn":
            layers_list: list[nn.Module] = [nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(inplace=True)]
            current_c = 32
            for _ in range(3):
                layers_list.extend([
                    nn.Conv2d(current_c, current_c * 2, 3, padding=1),
                    nn.BatchNorm2d(current_c * 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                ])
                current_c *= 2
            layers_list.extend([
                nn.Flatten(),
                nn.Linear(current_c * (h // 8) * (h // 8), 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            ])
            model = nn.Sequential(*layers_list)

        elif architecture == "resnet_style":
            class ResNetStyle(nn.Module):
                def __init__(self, in_ch: int, n_classes: int) -> None:
                    super().__init__()
                    self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
                    self.bn1 = nn.BatchNorm2d(32)
                    self.block1 = self._make_block(32, 64)
                    self.block2 = self._make_block(64, 128)
                    self.block3 = self._make_block(128, 256)
                    self.pool = nn.AdaptiveAvgPool2d(1)
                    self.fc = nn.Linear(256, n_classes)
                    self.relu = nn.ReLU(inplace=True)

                def _make_block(self, in_c: int, out_c: int) -> nn.Sequential:
                    return nn.Sequential(
                        nn.Conv2d(in_c, out_c, 3, padding=1),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_c, out_c, 3, padding=1),
                        nn.BatchNorm2d(out_c),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x = self.relu(self.bn1(self.conv1(x)))
                    x = self.block1(x)
                    x = self.block2(x)
                    x = self.block3(x)
                    x = self.pool(x)
                    x = x.view(x.size(0), -1)
                    return self.fc(x)

            model = ResNetStyle(channels, num_classes)
        else:
            return f"Unknown architecture: {architecture}. Use simple_cnn, deep_cnn, or resnet_style."

        total_params, trainable_params = _count_parameters(model)
        loss_name = "cross_entropy" if num_classes > 2 else ("binary_crossentropy" if num_classes == 1 else "mse")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = _build_loss(loss_name)

        wrapper = ModelWrapper(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_name=loss_name,
            optimizer_name="adam",
            metrics=["accuracy"],
            input_shape=tuple(image_size),
            num_classes=num_classes,
        )
        _MODEL_REGISTRY[model_name] = wrapper

        result = {
            "model_name": model_name,
            "architecture": architecture,
            "input_shape": image_size,
            "num_classes": num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "data_augmentation": use_aug,
            "status": "created",
            "note": "Train this model using pt_model_train with your image data. "
                   "Flatten image data to (N, C*H*W) before passing to pt_model_train.",
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 15: Model Convert
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_model_convert",
    description=(
        "Convert a PyTorch model to deployment-ready formats: ONNX, TorchScript (trace), "
        "or TorchScript (script). Provides file sizes and compatibility info."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the model to convert"},
            "target_format": {
                "type": "string",
                "enum": ["onnx", "torchscript_trace", "torchscript_script"],
                "description": "Target conversion format",
            },
            "output_name": {"type": "string", "description": "Output filename (optional)"},
            "opset_version": {"type": "integer", "description": "ONNX opset version (default: 14)"},
            "dynamic_axes": {"type": "boolean", "description": "Enable dynamic axes for ONNX (default: true)"},
        },
        "required": ["model_name", "target_format"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "onnx", "torchscript", "conversion", "export", "deployment"],
    timeout=120,
    version="1.0",
)
async def pt_model_convert(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        wrapper = _MODEL_REGISTRY.get(model_name)
        if wrapper is None:
            return f"Model '{model_name}' not found."

        model: nn.Module = wrapper.model
        target: str = params.get("target_format", "onnx")
        workspace = _get_workspace()
        output_name: str = params.get("output_name", model_name)

        model.eval()

        # Create example input for tracing
        if wrapper.input_shape:
            example_input = torch.randn(1, *wrapper.input_shape)
        else:
            example_input = torch.randn(1, 100)

        if example_input.ndim > 2:
            example_input = example_input.view(1, -1)

        if target == "onnx":
            if not _ONNX_AVAILABLE:
                return "ONNX is not installed. Install with: pip install onnx onnxruntime"

            filepath = workspace / f"{output_name}.onnx"
            opset: int = int(params.get("opset_version", 14))
            dynamic: bool = params.get("dynamic_axes", True)

            dynamic_axes_spec = None
            if dynamic:
                dynamic_axes_spec = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

            torch.onnx.export(
                model,
                example_input,
                str(filepath),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes_spec,
            )

            # Validate ONNX model
            onnx_model = onnx.load(str(filepath))
            onnx.checker.check_model(onnx_model)

            size_kb = os.path.getsize(filepath) / 1024
            result = {
                "model_name": model_name,
                "target_format": "onnx",
                "output_path": str(filepath),
                "size_kb": round(size_kb, 2),
                "opset_version": opset,
                "dynamic_axes": dynamic,
                "onnx_valid": True,
                "status": "exported",
                "usage": f"Load with: onnxruntime.InferenceSession('{filepath}')",
            }

        elif target == "torchscript_trace":
            filepath = workspace / f"{output_name}_traced.pt"
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
                traced.save(str(filepath))

            size_kb = os.path.getsize(filepath) / 1024
            result = {
                "model_name": model_name,
                "target_format": "torchscript_trace",
                "output_path": str(filepath),
                "size_kb": round(size_kb, 2),
                "status": "exported",
                "usage": f"Load with: torch.jit.load('{filepath}')",
            }

        elif target == "torchscript_script":
            filepath = workspace / f"{output_name}_scripted.pt"
            try:
                scripted = torch.jit.script(model)
                scripted.save(str(filepath))
                size_kb = os.path.getsize(filepath) / 1024
                result = {
                    "model_name": model_name,
                    "target_format": "torchscript_script",
                    "output_path": str(filepath),
                    "size_kb": round(size_kb, 2),
                    "status": "exported",
                    "usage": f"Load with: torch.jit.load('{filepath}')",
                }
            except Exception as script_err:
                # Fallback to trace
                filepath2 = workspace / f"{output_name}_traced_fallback.pt"
                with torch.no_grad():
                    traced = torch.jit.trace(model, example_input)
                    traced.save(str(filepath2))
                result = {
                    "model_name": model_name,
                    "target_format": "torchscript_trace_fallback",
                    "output_path": str(filepath2),
                    "note": f"TorchScript script failed ({script_err}), used trace instead",
                    "status": "exported",
                }
        else:
            return f"Unknown target format: {target}. Use: onnx, torchscript_trace, torchscript_script"

        total_params, _ = _count_parameters(model)
        result["model_parameters"] = total_params

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 16: Training Monitor
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_training_monitor",
    description=(
        "Analyze PyTorch training history and detect issues: overfitting, underfitting, "
        "vanishing gradients, learning rate problems, and unstable training. Provides "
        "recommendations for improvement based on training curves analysis."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "model_name": {"type": "string", "description": "Name of the trained model"},
        },
        "required": ["model_name"],
    },
    risk="low",
    category="ml_ai",
    tags=["pytorch", "monitoring", "overfitting", "analysis", "training"],
    timeout=30,
    version="1.0",
)
async def pt_training_monitor(params: dict) -> str:
    err = _check_available()
    if err:
        return err

    try:
        model_name: str = params.get("model_name", "")
        history = _TRAINING_HISTORY.get(model_name)
        if not history:
            return f"No training history found for '{model_name}'. Train the model first."

        issues: list[str] = []
        recommendations: list[str] = []

        # ── Analyze loss curves ──
        train_loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])

        if train_loss and val_loss:
            if len(val_loss) > 3:
                loss_diff = [t - v for t, v in zip(train_loss, val_loss)]

                # Overfitting detection
                if val_loss[-1] > val_loss[-3] and train_loss[-1] < train_loss[-3]:
                    gap = val_loss[-1] - train_loss[-1]
                    issues.append(
                        f"Overfitting detected: val_loss ({val_loss[-1]:.4f}) > "
                        f"train_loss ({train_loss[-1]:.4f}), gap: {gap:.4f}"
                    )
                    recommendations.append(
                        "Consider: more dropout, data augmentation, early stopping, "
                        "weight decay, or reduce model complexity"
                    )

                # Underfitting detection
                if train_loss[-1] > 0.5:
                    issues.append(f"Possible underfitting: final train_loss is high ({train_loss[-1]:.4f})")
                    recommendations.append(
                        "Consider: more epochs, larger model, or reduce regularization"
                    )

                # Unstable training
                if len(val_loss) > 2:
                    loss_variance = float(np.var(val_loss[-5:])) if len(val_loss) >= 5 else float(np.var(val_loss))
                    if loss_variance > 0.01:
                        issues.append(f"Unstable training: loss variance is {loss_variance:.6f}")
                        recommendations.append(
                            "Consider: lower learning rate, gradient clipping, or batch normalization"
                        )

                # Plateau detection (vanishing gradients)
                if len(train_loss) > 5:
                    recent_improvement = train_loss[-5] - train_loss[-1]
                    if recent_improvement < 0.001:
                        issues.append(
                            f"Plateau detected: loss improved only {recent_improvement:.6f} "
                            f"in last 5 epochs"
                        )
                        recommendations.append(
                            "Consider: learning rate scheduling (CosineAnnealing, OneCycleLR), "
                            "different optimizer, or model architecture change"
                        )

        # ── Analyze accuracy curves ──
        train_acc = history.get("accuracy", [])
        val_acc = history.get("val_accuracy", [])

        if train_acc and val_acc:
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]
            acc_gap = final_train_acc - final_val_acc

            if acc_gap > 0.1:
                issues.append(
                    f"Large accuracy gap: train ({final_train_acc:.4f}) - "
                    f"val ({final_val_acc:.4f}) = {acc_gap:.4f}"
                )
                recommendations.append(
                    "Consider: regularization, dropout, weight decay, or more training data"
                )

            if final_val_acc < 0.5:
                issues.append(f"Low validation accuracy ({final_val_acc:.4f})")
                recommendations.append(
                    "Consider: different architecture, pretrained features, or check data quality"
                )

        # ── Check best epoch ──
        if "val_loss" in history and len(history["val_loss"]) > 1:
            best_epoch = int(np.argmin(history["val_loss"]))
            last_epoch = len(history["val_loss"]) - 1
            if best_epoch < last_epoch:
                issues.append(
                    f"Best val_loss was at epoch {best_epoch}, not the final epoch {last_epoch}"
                )
                recommendations.append("Ensure early stopping with best weight restoration is enabled")

        # ── Check learning rate ──
        lr_history = history.get("learning_rate", [])
        if lr_history and len(set(lr_history)) > 1:
            min_lr = min(lr_history)
            max_lr = max(lr_history)
            recommendations.append(
                f"Learning rate varied from {max_lr:.6f} to {min_lr:.6f} (scheduler active)"
            )

        if not issues:
            issues.append("No issues detected")
            recommendations.append("Training looks healthy!")

        result = {
            "model_name": model_name,
            "epochs_trained": len(train_loss),
            "issues_found": len(issues) - 1 if issues == ["No issues detected"] else len(issues),
            "issues": issues,
            "recommendations": recommendations,
            "training_summary": {
                "final_train_loss": round(train_loss[-1], 6) if train_loss else None,
                "final_val_loss": round(val_loss[-1], 6) if val_loss else None,
                "best_val_loss": round(min(val_loss), 6) if val_loss else None,
                "final_train_acc": round(train_acc[-1], 4) if train_acc else None,
                "final_val_acc": round(val_acc[-1], 4) if val_acc else None,
                "final_lr": lr_history[-1] if lr_history else None,
            },
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"[PyTorch Error] {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: PyTorch Capabilities Info
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_ml_info",
    description=(
        "Get information about available PyTorch ML/AI capabilities: PyTorch version, "
        "available tools, supported architectures, pretrained models, GPU/CUDA/MPS "
        "availability, and feature flags."
    ),
    parameters_schema={"type": "object", "properties": {}},
    risk="low",
    category="ml_ai",
    tags=["pytorch", "info", "capabilities", "ml", "ai"],
    timeout=10,
    version="1.0",
)
async def pt_ml_info(_params: dict) -> str:
    try:
        torch_version = torch.__version__ if _TORCH_AVAILABLE else None
        torchvision_version = torchvision.__version__ if _TORCHVISION_AVAILABLE else None
        sklearn_version = sklearn.__version__ if _SKLEARN_AVAILABLE else None
        np_version = np.__version__ if _NP_AVAILABLE else None
        onnx_version = onnx.__version__ if _ONNX_AVAILABLE else None

        # Device info
        device_info: dict[str, Any] = {"device": "cpu"}
        if _TORCH_AVAILABLE:
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            device_info = {
                "cuda_available": cuda_available,
                "mps_available": mps_available,
                "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                "recommended_device": _get_device(),
            }

        registered_models = list(_MODEL_REGISTRY.keys())
        models_with_history = [name for name, hist in _TRAINING_HISTORY.items() if hist]

        result: dict[str, Any] = {
            "pytorch": {
                "available": _TORCH_AVAILABLE,
                "version": torch_version,
                "cuda_compiled": torch.version.cuda if _TORCH_AVAILABLE else None,
            },
            "torchvision": {
                "available": _TORCHVISION_AVAILABLE,
                "version": torchvision_version,
            },
            "scikit_learn": {
                "available": _SKLEARN_AVAILABLE,
                "version": sklearn_version,
            },
            "numpy": {
                "available": _NP_AVAILABLE,
                "version": np_version,
            },
            "onnx": {
                "available": _ONNX_AVAILABLE,
                "version": onnx_version,
            },
            "device": device_info,
            "tools": [
                "pt_model_create", "pt_model_train", "pt_model_predict",
                "pt_model_evaluate", "pt_model_save", "pt_model_load",
                "pt_model_info", "pt_data_preprocess", "pt_data_augment",
                "pt_transfer_learning", "pt_model_visualize", "pt_hyperparameter_tune",
                "pt_text_classification", "pt_image_classification",
                "pt_model_convert", "pt_training_monitor", "pt_ml_info",
            ],
            "registered_models": registered_models,
            "models_with_training_history": models_with_history,
            "capabilities": {
                "neural_networks": True,
                "convolutional_networks": True,
                "recurrent_networks": True,
                "transformers": True,
                "transfer_learning": _TORCHVISION_AVAILABLE,
                "data_augmentation": _TORCHVISION_AVAILABLE,
                "hyperparameter_tuning": True,
                "mixed_precision_training": _TORCH_AVAILABLE and torch.cuda.is_available(),
                "gradient_clipping": True,
                "lr_scheduling": True,
                "model_export": {
                    "onnx": _ONNX_AVAILABLE,
                    "torchscript_trace": True,
                    "torchscript_script": True,
                    "state_dict": True,
                    "full_model": True,
                },
                "text_classification": True,
                "image_classification": True,
                "training_monitoring": True,
            },
            "supported_layers": {
                "feedforward": ["dense (Linear)", "dropout", "batch_norm"],
                "convolutional": ["conv2d (Conv2d)", "max_pool (MaxPool2d)", "global_avg_pool (AdaptiveAvgPool2d)", "residual_block"],
                "recurrent": ["lstm (LSTM)", "gru (GRU)", "bidirectional_lstm"],
                "attention": ["attention (MultiheadAttention)"],
                "embedding": ["embedding (Embedding)"],
            },
            "supported_pretrained_backbones": [
                "resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0",
                "vgg16", "densenet121", "wide_resnet50_2", "inception_v3",
            ] if _TORCHVISION_AVAILABLE else [],
            "advanced_features": [
                "Mixed Precision (AMP)",
                "Gradient Clipping",
                "Learning Rate Scheduling (ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR)",
                "Early Stopping with best weight restoration",
                "TorchScript export",
                "ONNX export with validation",
                "Model parallelism basics",
            ],
        }

        if _IMPORT_ERRORS:
            result["import_errors"] = _IMPORT_ERRORS

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"[ML Info Error] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL: PyTorch Status (always available, no torch required)
# ══════════════════════════════════════════════════════════════════════════════

@tool(
    name="pt_status",
    description="Check PyTorch ML/AI library availability. Shows what's installed and what's missing.",
    parameters_schema={"type": "object", "properties": {}, "required": []},
    category="ml_ai",
    risk="low",
    timeout=5,
)
async def pt_status(_params: dict) -> str:
    lines = ["## PyTorch ML/AI Library Status"]
    lines.append(f"- numpy: {'OK installed' if _NP_AVAILABLE else 'NOT installed'}")
    lines.append(f"- scikit-learn: {'OK installed' if _SKLEARN_AVAILABLE else 'NOT installed'}")
    lines.append(f"- torch: {'OK installed' if _TORCH_AVAILABLE else 'NOT installed'}")
    lines.append(f"- torchvision: {'OK installed' if _TORCHVISION_AVAILABLE else 'NOT installed'}")
    lines.append(f"- onnx: {'OK installed' if _ONNX_AVAILABLE else 'NOT installed'}")
    if _IMPORT_ERRORS:
        lines.append(f"\nErrors:\n" + "\n".join(f"  - {e}" for e in _IMPORT_ERRORS))
    if _TORCH_AVAILABLE:
        lines.append(f"\nPyTorch version: {torch.__version__}")
        lines.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        lines.append(f"MPS available: {mps}")
        lines.append(f"\nAvailable tools: 17 (full PyTorch suite)")
    else:
        lines.append("\nOnly pt_status is available. Install PyTorch for full ML capabilities:")
        lines.append("  pip install torch torchvision scikit-learn numpy")
    return "\n".join(lines)
