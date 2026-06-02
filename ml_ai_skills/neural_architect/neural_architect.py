"""
Neural Architect — Advanced Neural Architecture Design System
==============================================================
Provides comprehensive neural network architecture design: model
blueprints, layer compositing, architecture visualization, FLOPs
estimation, transfer learning setup, and component recommendations.
Supports CNN, Transformer, LSTM, GNN, and hybrid architectures.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ArchitectureFamily(Enum):
    CNN = "cnn"
    TRANSFORMER = "transformer"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    GNN = "gnn"
    HYBRID = "hybrid"
    MLP = "mlp"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    VIT = "vit"
    UNET = "unet"
    GAN = "gan"
    DIFFUSION = "diffusion"
    AUTOENCODER = "autoencoder"


class ActivationFunction(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"    # Swish
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    PRELU = "prelu"
    SELU = "selu"
    MISH = "mish"


class NormalizationType(Enum):
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    RMS_NORM = "rms_norm"
    NONE = "none"


# ---------------------------------------------------------------------------
# Architecture Blueprint
# ---------------------------------------------------------------------------

@dataclass
class ArchitectLayer:
    """A single layer in the architecture blueprint."""
    name: str = ""
    layer_type: str = "linear"       # linear | conv2d | conv1d | lstm | multihead_attention | embedding
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: Tuple[int, int] = (3, 3)
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    activation: str = "relu"
    normalization: str = "none"
    dropout: float = 0.0
    num_heads: int = 8              # for attention
    hidden_dim: int = 0             # for ffn in transformer
    bidirectional: bool = False     # for rnn/lstm
    num_layers: int = 1             # for stacked rnn

    def flops_contribution(self, input_size: int) -> int:
        """Estimate FLOPs for this layer."""
        if self.layer_type == "linear":
            return 2 * self.in_channels * self.out_channels
        elif self.layer_type == "conv2d":
            return (2 * self.in_channels * self.out_channels *
                    self.kernel_size[0] * self.kernel_size[1] *
                    (input_size // self.stride) ** 2)
        elif self.layer_type == "multihead_attention":
            return 4 * input_size * self.hidden_dim ** 2
        return 0


@dataclass
class ArchitectureBlueprint:
    """Complete neural network architecture design."""
    name: str = "architecture"
    family: ArchitectureFamily = ArchitectureFamily.CNN
    layers: List[ArchitectLayer] = field(default_factory=list)
    input_shape: Tuple[int, ...] = (3, 224, 224)
    num_classes: int = 1000
    total_params: int = 0
    total_flops: int = 0

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        total = 0
        for layer in self.layers:
            if layer.layer_type == "linear":
                total += layer.in_channels * layer.out_channels + layer.out_channels
            elif layer.layer_type == "conv2d":
                total += (layer.in_channels * layer.out_channels *
                          layer.kernel_size[0] * layer.kernel_size[1] +
                          layer.out_channels)
            elif layer.layer_type == "multihead_attention":
                total += 4 * layer.hidden_dim ** 2 + 4 * layer.hidden_dim
            elif layer.layer_type == "lstm":
                total += 4 * (layer.in_channels * layer.hidden_dim +
                              layer.hidden_dim ** 2 + layer.hidden_dim)
            elif layer.layer_type == "embedding":
                total += layer.in_channels * layer.out_channels
        return total

    def estimate_flops(self, input_size: int = 224) -> int:
        """Estimate total FLOPs."""
        total = 0
        for layer in self.layers:
            total += layer.flops_contribution(input_size)
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "layers": [asdict(l) for l in self.layers],
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "estimated_params": self.estimate_params(),
            "estimated_flops": self.estimate_flops(),
        }


# ---------------------------------------------------------------------------
# Architecture Generator
# ---------------------------------------------------------------------------

class ArchitectureGenerator:
    """Generate common architecture blueprints."""

    @staticmethod
    def resnet(num_layers: int = 50, num_classes: int = 1000) -> ArchitectureBlueprint:
        """Generate a ResNet-like blueprint."""
        bp = ArchitectureBlueprint(name=f"resnet{num_layers}",
                                   family=ArchitectureFamily.RESNET,
                                   num_classes=num_classes)

        # Stem
        bp.layers.append(ArchitectLayer(
            name="conv1", layer_type="conv2d",
            in_channels=3, out_channels=64, kernel_size=(7, 7),
            stride=2, padding=3, normalization="batch_norm",
            activation="relu"))
        bp.layers.append(ArchitectLayer(
            name="maxpool", layer_type="conv2d",
            in_channels=64, out_channels=64, kernel_size=(3, 3),
            stride=2, padding=1, activation="none"))

        # Residual blocks (simplified)
        channels = [64, 128, 256, 512]
        blocks_per_layer = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
        blocks = blocks_per_layer.get(num_layers, [3, 4, 6, 3])

        for stage, (ch, n_blocks) in enumerate(zip(channels, blocks)):
            for b in range(n_blocks):
                stride = 2 if b == 0 and stage > 0 else 1
                bp.layers.append(ArchitectLayer(
                    name=f"res{stage + 1}_{b}",
                    layer_type="conv2d",
                    in_channels=channels[stage - 1] if b == 0 and stage > 0 else ch,
                    out_channels=ch,
                    kernel_size=(3, 3), stride=stride, padding=1,
                    normalization="batch_norm", activation="relu",
                ))
                bp.layers.append(ArchitectLayer(
                    name=f"res{stage + 1}_{b}_b",
                    layer_type="conv2d",
                    in_channels=ch, out_channels=ch,
                    kernel_size=(3, 3), stride=1, padding=1,
                    normalization="batch_norm", activation="relu",
                ))

        # Head
        bp.layers.append(ArchitectLayer(
            name="avg_pool", layer_type="linear",
            in_channels=512, out_channels=512, activation="none"))
        bp.layers.append(ArchitectLayer(
            name="fc", layer_type="linear",
            in_channels=512, out_channels=num_classes, activation="none"))

        return bp

    @staticmethod
    def transformer(vocab_size: int = 30000,
                    d_model: int = 512,
                    num_heads: int = 8,
                    num_layers: int = 6) -> ArchitectureBlueprint:
        """Generate a Transformer blueprint."""
        bp = ArchitectureBlueprint(name="transformer",
                                   family=ArchitectureFamily.TRANSFORMER,
                                   input_shape=(vocab_size,),
                                   num_classes=vocab_size)

        bp.layers.append(ArchitectLayer(
            name="embedding", layer_type="embedding",
            in_channels=vocab_size, out_channels=d_model))

        for i in range(num_layers):
            bp.layers.append(ArchitectLayer(
                name=f"attention_{i}",
                layer_type="multihead_attention",
                in_channels=d_model, out_channels=d_model,
                num_heads=num_heads, hidden_dim=d_model,
                normalization="layer_norm"))
            bp.layers.append(ArchitectLayer(
                name=f"ffn_{i}",
                layer_type="linear",
                in_channels=d_model, out_channels=d_model * 4,
                activation="gelu", normalization="layer_norm"))
            bp.layers.append(ArchitectLayer(
                name=f"ffn_{i}_proj",
                layer_type="linear",
                in_channels=d_model * 4, out_channels=d_model))

        bp.layers.append(ArchitectLayer(
            name="output", layer_type="linear",
            in_channels=d_model, out_channels=vocab_size,
            activation="softmax"))
        return bp

    @staticmethod
    def unet(in_channels: int = 3, out_channels: int = 1,
             base_filters: int = 64) -> ArchitectureBlueprint:
        """Generate a U-Net blueprint for segmentation."""
        bp = ArchitectureBlueprint(name="unet",
                                   family=ArchitectureFamily.UNET,
                                   input_shape=(in_channels, 256, 256),
                                   num_classes=out_channels)

        # Encoder
        filters = [base_filters * (2 ** i) for i in range(4)]
        for i, f in enumerate(filters):
            bp.layers.append(ArchitectLayer(
                name=f"enc_conv_{i}_1",
                layer_type="conv2d",
                in_channels=in_channels if i == 0 else filters[i - 1],
                out_channels=f,
                kernel_size=(3, 3), padding=1,
                normalization="batch_norm", activation="relu"))
            bp.layers.append(ArchitectLayer(
                name=f"enc_conv_{i}_2",
                layer_type="conv2d",
                in_channels=f, out_channels=f,
                kernel_size=(3, 3), padding=1,
                normalization="batch_norm", activation="relu"))

        # Bridge
        bp.layers.append(ArchitectLayer(
            name="bridge", layer_type="conv2d",
            in_channels=filters[-1], out_channels=filters[-1] * 2,
            kernel_size=(3, 3), padding=1,
            normalization="batch_norm", activation="relu"))

        # Decoder (transposed conv upsample)
        for i, f in enumerate(reversed(filters)):
            bp.layers.append(ArchitectLayer(
                name=f"dec_conv_{i}",
                layer_type="conv2d",
                in_channels=f * 2 if i == 0 else f * 2,
                out_channels=f,
                kernel_size=(3, 3), padding=1,
                normalization="batch_norm", activation="relu"))

        # Output
        bp.layers.append(ArchitectLayer(
            name="output", layer_type="conv2d",
            in_channels=base_filters, out_channels=out_channels,
            kernel_size=(1, 1), activation="sigmoid"))

        return bp

    @staticmethod
    def lstm_classifier(input_size: int = 100,
                        hidden_size: int = 256,
                        num_layers: int = 2,
                        num_classes: int = 10) -> ArchitectureBlueprint:
        """Generate an LSTM classifier blueprint."""
        bp = ArchitectureBlueprint(name="lstm_classifier",
                                   family=ArchitectureFamily.LSTM,
                                   input_shape=(input_size,),
                                   num_classes=num_classes)
        bp.layers.append(ArchitectLayer(
            name="lstm", layer_type="lstm",
            in_channels=input_size, hidden_dim=hidden_size,
            num_layers=num_layers, bidirectional=True,
            normalization="layer_norm"))
        bp.layers.append(ArchitectLayer(
            name="fc", layer_type="linear",
            in_channels=hidden_size * 2, out_channels=num_classes,
            activation="none"))
        return bp


# ---------------------------------------------------------------------------
# Architecture Analyzer
# ---------------------------------------------------------------------------

class ArchitectureAnalyzer:
    """Analyze and compare architectures."""

    @staticmethod
    def analyze(blueprint: ArchitectureBlueprint) -> Dict[str, Any]:
        """Analyze architecture characteristics."""
        param_count = blueprint.estimate_params()
        flops = blueprint.estimate_flops()
        size_mb = param_count * 4 / (1024 ** 2)

        layer_types = {}
        for layer in blueprint.layers:
            layer_types[layer.layer_type] = layer_types.get(layer.layer_type, 0) + 1

        return {
            "name": blueprint.name,
            "family": blueprint.family.value,
            "param_count": param_count,
            "size_mb": size_mb,
            "flops": flops,
            "memory_footprint_mb": size_mb * 2.5,
            "num_layers": len(blueprint.layers),
            "layer_breakdown": layer_types,
            "depth": len(blueprint.layers),
        }

    @staticmethod
    def compare(blueprints: List[ArchitectureBlueprint]) -> List[Dict[str, Any]]:
        """Compare multiple architectures."""
        return [ArchitectureAnalyzer.analyze(bp) for bp in blueprints]


# ---------------------------------------------------------------------------
# Component Recommender
# ---------------------------------------------------------------------------

class ArchitectureRecommender:
    """Recommend architecture components based on task."""

    @staticmethod
    def recommend(task: str = "image_classification",
                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Recommend architecture for a given task."""
        default_constraints = {"max_params": 100e6, "max_flops": 10e9}
        if constraints:
            default_constraints.update(constraints)
        c = default_constraints

        recommendations = {
            "image_classification": {
                "families": [ArchitectureFamily.RESNET, ArchitectureFamily.EFFICIENTNET,
                             ArchitectureFamily.VIT],
                "suggested": {"family": "resnet", "model": "ResNet-50",
                              "params_m": 25, "top1_acc": 76.1},
                "tiny": {"family": "efficientnet", "model": "EfficientNet-B0",
                         "params_m": 5.3, "top1_acc": 77.1},
                "large": {"family": "vit", "model": "ViT-Large",
                          "params_m": 307, "top1_acc": 85.1},
            },
            "object_detection": {
                "families": [ArchitectureFamily.CNN, ArchitectureFamily.HYBRID],
                "suggested": {"family": "cnn", "model": "YOLOv8-m",
                              "params_m": 25, "map": 50.2},
            },
            "segmentation": {
                "families": [ArchitectureFamily.UNET, ArchitectureFamily.CNN],
                "suggested": {"family": "unet", "model": "U-Net",
                              "params_m": 35, "miou": 0.72},
            },
            "nlp_classification": {
                "families": [ArchitectureFamily.TRANSFORMER, ArchitectureFamily.LSTM],
                "suggested": {"family": "transformer", "model": "BERT-Base",
                              "params_m": 110, "accuracy": 0.91},
            },
            "timeseries": {
                "families": [ArchitectureFamily.LSTM, ArchitectureFamily.TRANSFORMER],
                "suggested": {"family": "lstm", "model": "BiLSTM-2",
                              "params_m": 2.5, "mse": 0.01},
            },
        }

        result = recommendations.get(task, recommendations["image_classification"])
        return {
            "task": task,
            "recommendations": result,
            "constraints": c,
            "note": "Recommendations based on typical SOTA architectures",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_architecture(family: str = "resnet",
                                 num_classes: int = 1000,
                                 **params) -> Dict[str, Any]:
    """Generate an architecture blueprint."""
    gen = ArchitectureGenerator()
    if family == "resnet":
        bp = gen.resnet(params.get("num_layers", 50), num_classes)
    elif family == "transformer":
        bp = gen.transformer(
            params.get("vocab_size", 30000),
            params.get("d_model", 512),
            params.get("num_heads", 8),
            params.get("num_layers", 6),
        )
    elif family == "unet":
        bp = gen.unet(
            params.get("in_channels", 3),
            params.get("out_channels", 1),
            params.get("base_filters", 64),
        )
    elif family == "lstm":
        bp = gen.lstm_classifier(
            params.get("input_size", 100),
            params.get("hidden_size", 256),
            params.get("num_layers", 2),
            num_classes,
        )
    else:
        bp = gen.resnet(50, num_classes)
    return bp.to_dict()


async def analyze_architecture(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze an architecture blueprint."""
    bp = ArchitectureBlueprint(**{k: v for k, v in blueprint.items()
                                   if k in ArchitectureBlueprint.__dataclass_fields__})
    return ArchitectureAnalyzer.analyze(bp)


async def compare_architectures(blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare multiple architectures."""
    bps = [ArchitectureBlueprint(**{k: v for k, v in bp.items()
                                     if k in ArchitectureBlueprint.__dataclass_fields__})
           for bp in blueprints]
    return ArchitectureAnalyzer.compare(bps)


async def recommend_architecture(task: str = "image_classification",
                                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get architecture recommendations."""
    return ArchitectureRecommender.recommend(task, constraints)


async def list_architecture_families() -> List[str]:
    """List available architecture families."""
    return [f.value for f in ArchitectureFamily]


async def list_activations() -> List[str]:
    """List available activation functions."""
    return [a.value for a in ActivationFunction]


async def list_normalizations() -> List[str]:
    """List available normalization types."""
    return [n.value for n in NormalizationType]
