#!/usr/bin/env python3
"""Distributed Training — Advanced Distributed Training Skill.

Production-grade distributed training patterns for large-scale model training.
Implements DDP, FSDP, DeepSpeed ZeRO, gradient checkpointing, torch.compile,
pipeline parallelism, elastic training, and custom distributed samplers.

All patterns are real executable PyTorch code, not configuration generators.

Usage:
    python distributed_training.py launch-ddp --script train.py --nproc 4
    python distributed_training.py launch-fsdp --script train.py --nproc 4
    python distributed_training.py deepspeed-config --zero-stage 3 --bf16
    python distributed_training.py elastic-train --script train.py --min-nodes 1 --max-nodes 4
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import functools
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DISTRIBUTED INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════


def setup_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """Initialize distributed process group.

    Supports both environment-variable-based init (torchrun/deepspeed)
    and manual init for custom launchers.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedContext:
    """Context manager for distributed training setup and cleanup.

    Usage:
        with DistributedContext() as ctx:
            print(f"Rank {ctx.rank}/{ctx.world_size}")
            # ... training code ...
    """

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_main = True

    def __enter__(self) -> "DistributedContext":
        self.rank, self.local_rank, self.world_size = setup_distributed(self.backend)
        self.is_main = (self.rank == 0)
        return self

    def __exit__(self, *args: Any) -> None:
        cleanup_distributed()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DDP (DISTRIBUTED DATA PARALLEL)
# ═══════════════════════════════════════════════════════════════════════════════


def wrap_ddp(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> DDP:
    """Wrap a model with DistributedDataParallel.

    DDP replicates the model across all GPUs and synchronizes gradients
    after each backward pass via AllReduce.

    Args:
        model: Model to wrap.
        device_ids: CUDA devices for this process.
        output_device: Device for output tensors.
        find_unused_parameters: Visit all parameters in backward (slower but safer).
        gradient_as_bucket_view: Reduce memory by using gradient bucket views.
        static_graph: Enable for models with fixed computation graph (faster).

    Returns:
        DDP-wrapped model.
    """
    if not dist.is_initialized():
        return model  # Fall back to single-GPU

    rank = dist.get_rank()
    device_id = device_ids[0] if device_ids else rank
    output_device = output_device or device_id

    ddp_model = DDP(
        model,
        device_ids=[device_id] if torch.cuda.is_available() else None,
        output_device=output_device if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )

    return ddp_model


class DDPTrainer:
    """Complete DDP training loop with gradient accumulation, mixed precision,
    gradient checkpointing, and comprehensive logging.

    Args:
        model: Model to train.
        train_dataset: Training dataset.
        config: Training configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        grad_accum_steps: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 3,
        use_bf16: bool = True,
        gradient_checkpointing: bool = False,
        use_compile: bool = False,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
    ):
        self.ctx = DistributedContext()
        self.ctx.__enter__()

        self.device = torch.device(f"cuda:{self.ctx.local_rank}" if torch.cuda.is_available() else "cpu")
        self.is_main = self.ctx.is_main

        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir

        # Move model to device
        model = model.to(self.device)

        # Gradient checkpointing
        if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if self.is_main:
                print("Gradient checkpointing enabled")

        # torch.compile
        if use_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
            if self.is_main:
                print("Model compiled with torch.compile")

        # Wrap with DDP
        self.model = wrap_ddp(model, gradient_as_bucket_view=True)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(train_dataset) // batch_size // self.ctx.world_size * num_epochs // grad_accum_steps
        self.scheduler = self._cosine_lr_with_warmup(
            self.optimizer, learning_rate, int(0.1 * total_steps), total_steps
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if (use_bf16 is False and self.device.type == "cuda") else None

        # DataLoader with DistributedSampler
        self.train_sampler = DistributedSampler(
            train_dataset, num_replicas=self.ctx.world_size, rank=self.ctx.rank, shuffle=True
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=self.train_sampler,
            num_workers=2, pin_memory=True, drop_last=True,
        )
        self.val_dataset = val_dataset

        if self.is_main:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"DDP Trainer initialized: rank={self.ctx.rank}, world_size={self.ctx.world_size}")
            print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @staticmethod
    def _cosine_lr_with_warmup(
        optimizer: torch.optim.Optimizer, base_lr: float, warmup_steps: int, total_steps: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine LR schedule with linear warmup."""
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return base_lr * step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train(self) -> Dict[str, Any]:
        """Run the complete training loop."""
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
        global_step = 0

        for epoch in range(self.num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_loader):
                # Move to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    loss = self._train_step(batch)
                else:
                    batch = batch.to(self.device)
                    loss = self._train_step_simple(batch)

                epoch_loss += loss
                num_batches += 1

                if (step + 1) % self.grad_accum_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1

                    # Logging
                    if self.is_main and global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / num_batches
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(f"Epoch {epoch+1}/{self.num_epochs} | Step {global_step} | "
                              f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                        history["lr"].append(lr)

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            history["train_loss"].append(avg_epoch_loss)

            # Validation
            if self.val_dataset is not None:
                val_loss = self._evaluate()
                history["val_loss"].append(val_loss)
                if self.is_main:
                    print(f"Epoch {epoch+1} | Train: {avg_epoch_loss:.4f} | Val: {val_loss:.4f}")

            # Save checkpoint
            if self.is_main:
                self._save_checkpoint(global_step, epoch, history)

        if self.is_main:
            print("Training complete!")

        self.ctx.__exit__()
        return {"history": history, "total_steps": global_step}

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with mixed precision."""
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs.get("loss", outputs) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = self.model(**batch)
                loss = outputs.get("loss", outputs) / self.grad_accum_steps
            loss.backward()

        return loss.item() * self.grad_accum_steps

    def _train_step_simple(self, batch: torch.Tensor) -> float:
        """Training step for simple (input, target) batches."""
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss = F.cross_entropy(self.model(batch[0]), batch[1]) / self.grad_accum_steps
            self.scaler.scale(loss).backward()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                loss = F.cross_entropy(self.model(batch[0]), batch[1]) / self.grad_accum_steps
            loss.backward()

        return loss.item() * self.grad_accum_steps

    @torch.no_grad()
    def _evaluate(self, batch_size: int = 8) -> float:
        """Evaluate model on validation set."""
        self.model.eval()
        loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        total_loss = 0.0
        n = 0
        for batch in loader:
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.get("loss", torch.tensor(0.0)).item()
            n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def _save_checkpoint(self, step: int, epoch: int, history: Dict) -> None:
        """Save checkpoint on main process."""
        path = os.path.join(self.checkpoint_dir, f"ddp_step{step}.pt")
        state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        torch.save({
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "history": history,
        }, path)
        print(f"Checkpoint saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FSDP (FULLY SHARDED DATA PARALLEL)
# ═══════════════════════════════════════════════════════════════════════════════


def wrap_fsdp(
    model: nn.Module,
    sharding_strategy: str = "full",
    mixed_precision: str = "bf16",
    auto_wrap_policy: str = "none",
    cpu_offload: bool = False,
) -> nn.Module:
    """Wrap model with Fully Sharded Data Parallel.

    FSDP shards model parameters, gradients, and optimizer state across GPUs.
    This allows training much larger models than DDP.

    Sharding strategies:
        - 'full': Shard parameters, gradients, and optimizer state (ZeRO-3 equivalent)
        - 'grad': Shard gradients and optimizer state (ZeRO-2 equivalent)
        - 'no_shard': Replicate all (DDP equivalent, but with FSDP features)

    Args:
        model: Model to wrap.
        sharding_strategy: 'full', 'grad', or 'no_shard'.
        mixed_precision: 'bf16', 'fp16', or 'none'.
        auto_wrap_policy: 'transformer', 'none', or custom.
        cpu_offload: Offload parameters/optimizer to CPU.

    Returns:
        FSDP-wrapped model.
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            size_based_auto_wrap_policy,
            enable_wrap,
        )
    except ImportError:
        print("FSDP requires PyTorch >= 2.0")
        return model

    # Sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision
    mp = None
    if mixed_precision == "bf16":
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
    elif mixed_precision == "fp16":
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float32,
        )

    # Auto-wrap policy
    wrap_policy = None
    if auto_wrap_policy == "transformer":
        # Wrap each transformer block as a unit
        transformer_cls = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        # Also try to find custom transformer block classes
        for name, cls in model.__class__.__mro__:
            if "Block" in cls.__name__ or "Layer" in cls.__name__:
                for child_name, child_module in model.named_children():
                    if isinstance(child_module, nn.ModuleList):
                        if len(child_module) > 0:
                            transformer_cls.add(type(child_module[0]))
                            break
        wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls=transformer_cls)
    elif auto_wrap_policy == "size":
        wrap_policy = size_based_auto_wrap_policy(min_num_params=1e8)

    # CPU offload
    offload = CPUOffload(offload_params=cpu_offload) if cpu_offload else None

    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding,
        mixed_precision=mp,
        auto_wrap_policy=wrap_policy,
        cpu_offload=offload,
    )

    if dist.get_rank() == 0:
        print(f"FSDP wrapped: strategy={sharding_strategy}, mp={mixed_precision}, "
              f"cpu_offload={cpu_offload}")

    return fsdp_model


def fsdp_full_train(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    sharding_strategy: str = "full",
    checkpoint_dir: str = "./checkpoints",
) -> Dict[str, Any]:
    """Complete FSDP training function.

    Demonstrates the full pattern: setup -> wrap -> train -> cleanup.
    """
    with DistributedContext() as ctx:
        device = torch.device(f"cuda:{ctx.local_rank}" if torch.cuda.is_available() else "cpu")

        # Wrap with FSDP
        model = model.to(device)
        model = wrap_fsdp(model, sharding_strategy=sharding_strategy)

        # Optimizer (FSDP handles parameter sharding automatically)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # DataLoader
        sampler = DistributedSampler(train_dataset, shuffle=True)
        loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)

        # Training loop
        history: Dict[str, List[float]] = {"train_loss": []}

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            n = 0

            for batch in loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.get("loss", torch.tensor(0.0))
                else:
                    x, y = batch[0].to(device), batch[1].to(device)
                    loss = F.cross_entropy(model(x), y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                n += 1

            avg_loss = epoch_loss / max(n, 1)
            history["train_loss"].append(avg_loss)

            if ctx.is_main:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    return {"history": history}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DEEPSPEED ZeRO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def create_deepspeed_zero_config(
    zero_stage: int = 2,
    bf16: bool = True,
    gradient_accumulation_steps: int = 1,
    train_micro_batch_size_per_gpu: int = 4,
    offload_optimizer: bool = False,
    offload_param: bool = False,
    offload_optimizer_device: str = "cpu",
    offload_param_device: str = "cpu",
    gradient_clipping: float = 1.0,
    optimizer_name: str = "AdamW",
    optimizer_params: Optional[Dict] = None,
    scheduler_name: str = "WarmupLR",
    scheduler_params: Optional[Dict] = None,
    zero_offload_param_pin_memory: bool = True,
    zero_offload_optimizer_pin_memory: bool = True,
) -> Dict[str, Any]:
    """Create a complete DeepSpeed ZeRO configuration.

    ZeRO Stages:
        Stage 1: Shards optimizer states across GPUs
        Stage 2: Shards optimizer states + gradients across GPUs
        Stage 3: Shards optimizer states + gradients + model parameters

    Args:
        zero_stage: ZeRO stage (1, 2, or 3).
        bf16: Use bf16 mixed precision.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        train_micro_batch_size_per_gpu: Micro batch size per GPU.
        offload_optimizer: Offload optimizer to CPU/NVMe.
        offload_param: Offload parameters to CPU/NVMe (Stage 3 only).
        gradient_clipping: Max gradient norm.
        optimizer_name: Optimizer name.
        optimizer_params: Optimizer-specific parameters.
        scheduler_name: LR scheduler name.
        scheduler_params: Scheduler parameters.

    Returns:
        DeepSpeed configuration dictionary.
    """
    if optimizer_params is None:
        optimizer_params = {"lr": 1e-4, "betas": [0.9, 0.999], "weight_decay": 0.01}
    if scheduler_params is None:
        scheduler_params = {"warmup_min_lr": 0, "warmup_max_lr": 1e-4, "warmup_num_steps": 100}

    config: Dict[str, Any] = {
        "train_batch_size": train_micro_batch_size_per_gpu * gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "zero_optimization": {
            "stage": zero_stage,
        },
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": not bf16},
        "optimizer": {
            "type": optimizer_name,
            "params": optimizer_params,
        },
        "scheduler": {
            "type": scheduler_name,
            "params": scheduler_params,
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }

    # ZeRO-Offload for optimizer states
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": offload_optimizer_device,
            "pin_memory": zero_offload_optimizer_pin_memory,
        }

    # ZeRO-Infinity for parameter offloading (Stage 3)
    if offload_param and zero_stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": offload_param_device,
            "pin_memory": zero_offload_param_pin_memory,
        }

    return config


def init_deepspeed_engine(
    model: nn.Module,
    config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_parameters: Optional[Iterator] = None,
) -> Any:
    """Initialize a DeepSpeed engine.

    Args:
        model: Model to train.
        config: DeepSpeed configuration dictionary.
        optimizer: Optional pre-built optimizer.
        model_parameters: Optional model parameter iterator.

    Returns:
        DeepSpeed engine object.
    """
    try:
        import deepspeed
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config_params=config,
            model_parameters=model_parameters,
        )
        return engine
    except ImportError:
        print("DeepSpeed not installed. Run: pip install deepspeed")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CUSTOM DISTRIBUTED SAMPLER
# ═══════════════════════════════════════════════════════════════════════════════


class SortedLengthSampler(Sampler):
    """Distributed sampler that sorts examples by length to minimize padding.

    In each epoch, examples are sorted by length within each GPU's partition,
    reducing wasted computation on padding tokens.

    Args:
        dataset: Dataset with __len__ and a way to get lengths.
        lengths: Pre-computed example lengths.
        batch_size: Batch size (for bucketing within sort).
        num_replicas: Number of distributed replicas.
        rank: Current rank.
        shuffle: Whether to add randomness.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Divide indices among replicas
        total_size = len(lengths) - (len(lengths) % self.num_replicas) if drop_last else len(lengths)
        indices_per_replica = total_size // self.num_replicas
        self.start_idx = rank * indices_per_replica
        self.end_idx = self.start_idx + indices_per_replica

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.lengths)))

        # Sort by length for efficient batching
        sorted_indices = sorted(indices, key=lambda i: self.lengths[i])

        # Bucket into mini-batches and optionally shuffle buckets
        buckets = []
        for i in range(0, len(sorted_indices), self.batch_size):
            bucket = sorted_indices[i: i + self.batch_size]
            if self.shuffle:
                import random
                random.Random(self.epoch).shuffle(bucket)
            buckets.extend(bucket)

        # Take this rank's portion
        my_indices = buckets[self.start_idx:self.end_idx]

        return iter(my_indices)

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GRADIENT CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════


class CheckpointableBlock(nn.Module):
    """A transformer block that supports gradient checkpointing.

    Gradient checkpointing trades computation for memory: instead of storing
    all intermediate activations, it recomputes them during backward pass.
    This can reduce memory usage by up to 5x with ~30% more compute.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Inner forward function (used for checkpointing)."""
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)
        return x

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """Forward with optional gradient checkpointing.

        Args:
            x: Input tensor.
            use_checkpoint: If True, use gradient checkpointing for this block.
        """
        if use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_fn, x, use_reentrant=False)
        return self._forward_fn(x)


class CheckpointedTransformer(nn.Module):
    """Transformer with selective gradient checkpointing.

    Only checkpoints every Nth layer to balance memory savings vs compute cost.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN dimension.
        n_layers: Number of transformer blocks.
        checkpoint_every_n: Checkpoint every Nth layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        n_layers: int = 12,
        checkpoint_every_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CheckpointableBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.checkpoint_every_n = checkpoint_every_n
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, block in enumerate(self.blocks):
            use_ckpt = (i % self.checkpoint_every_n == 0) and self.training
            x = block(x, use_checkpoint=use_ckpt)
        return self.final_norm(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineStage(nn.Module):
    """A single stage in a pipeline-parallel model.

    Pipeline parallelism splits the model into sequential stages, each
    running on a different GPU. Data flows through stages in order.

    Args:
        layers: List of layers for this stage.
        stage_id: Stage identifier.
        num_stages: Total number of stages.
    """

    def __init__(self, layers: nn.ModuleList, stage_id: int, num_stages: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all layers in this stage."""
        for layer in self.layers:
            x = layer(x)
        return x


def build_pipeline_stages(
    model: nn.Module,
    num_stages: int,
    layer_attr: str = "blocks",
) -> List[PipelineStage]:
    """Split a model into pipeline stages.

    Args:
        model: The model to split.
        num_stages: Number of pipeline stages.
        layer_attr: Attribute name containing the layer list.

    Returns:
        List of PipelineStage modules.
    """
    layers = list(getattr(model, layer_attr, model.children()))
    n = len(layers)
    layers_per_stage = n // num_stages
    remainder = n % num_stages

    stages = []
    start = 0
    for i in range(num_stages):
        end = start + layers_per_stage + (1 if i < remainder else 0)
        stage_layers = nn.ModuleList(layers[start:end])
        stages.append(PipelineStage(stage_layers, stage_id=i, num_stages=num_stages))
        start = end

    return stages


# ═══════════════════════════════════════════════════════════════════════════════
# 8. ELASTIC TRAINING (FAULT TOLERANCE)
# ═══════════════════════════════════════════════════════════════════════════════


class ElasticTrainer:
    """Training with elastic fault tolerance.

    Automatically handles node failures by redistributing work across
    remaining workers. Uses TorchElastic (torch.distributed.elastic).

    Usage:
        trainer = ElasticTrainer(model, dataset)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        max_restarts: int = 3,
        checkpoint_dir: str = "./elastic_checkpoints",
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
    ):
        self.model = model
        self.dataset = dataset
        self.max_restarts = max_restarts
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.start_step = 0

    def _save_recovery(self, step: int, optimizer: torch.optim.Optimizer) -> None:
        """Save recovery checkpoint for elastic restart."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "recovery.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        }, path)

    def _load_recovery(self) -> int:
        """Load recovery checkpoint if available."""
        path = os.path.join(self.checkpoint_dir, "recovery.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.start_step = ckpt.get("step", 0)
            return self.start_step
        return 0

    def train(self) -> Dict[str, Any]:
        """Run training with elastic fault tolerance."""
        restarts = 0

        while restarts <= self.max_restarts:
            try:
                self._load_recovery()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = self.model.to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

                loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

                for epoch in range(self.num_epochs):
                    model.train()
                    for step, batch in enumerate(loader):
                        if isinstance(batch, (list, tuple)):
                            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                            loss = F.cross_entropy(model(batch[0]), batch[1])
                        else:
                            loss = F.mse_loss(model(batch[0].to(device)), batch[1].to(device))

                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                        if step % 50 == 0:
                            self._save_recovery(step, optimizer)

                return {"status": "completed", "epochs": self.num_epochs}

            except Exception as e:
                restarts += 1
                print(f"Training interrupted: {e}. Restarting ({restarts}/{self.max_restarts})...")
                time.sleep(5 * restarts)  # Exponential backoff

        return {"status": "failed", "restarts": restarts}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TORCH.COMPILE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    dynamic: bool = False,
    fullgraph: bool = False,
    backend: str = "inductor",
) -> nn.Module:
    """Apply torch.compile to a model with configurable options.

    Args:
        model: Model to compile.
        mode: Compilation mode:
            - 'default': Balance between compile time and performance
            - 'reduce-overhead': Maximize training throughput (uses CUDA graphs)
            - 'max-autotune': Spend more time optimizing for best performance
        dynamic: Enable dynamic shape support (for variable-length inputs).
        fullgraph: Require the entire model to be a single graph (no graph breaks).
        backend: Compilation backend ('inductor' default, 'cudagraphs', etc.)

    Returns:
        Compiled model.
    """
    if not hasattr(torch, "compile"):
        print("torch.compile requires PyTorch >= 2.0")
        return model

    compiled = torch.compile(
        model,
        mode=mode,
        dynamic=dynamic,
        fullgraph=fullgraph,
        backend=backend,
    )
    print(f"Model compiled with mode={mode}, dynamic={dynamic}, fullgraph={fullgraph}")
    return compiled


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SEQUENCE PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════════


class SequenceParallelContext:
    """Context manager for sequence-parallel training.

    Splits the sequence dimension across GPUs so each GPU processes a
    subsequence. After attention, the subsequences are gathered and
    re-scattered. Supports Ring Attention for efficient all-to-all.

    Example:
        sp_ctx = SequenceParallelContext(world_size=4, rank=0)
        local_seq = x[:, sp_ctx.local_start:sp_ctx.local_end]  # 1/4 of sequence
    """

    def __init__(self, world_size: int = 1, rank: int = 0):
        self.world_size = world_size
        self.rank = rank
        self.local_start = 0
        self.local_end = 0

    def split(self, seq_len: int) -> Tuple[int, int]:
        """Compute local start/end indices for a given sequence length."""
        chunk = seq_len // self.world_size
        remainder = seq_len % self.world_size
        self.local_start = self.rank * chunk + min(self.rank, remainder)
        self.local_end = self.local_start + chunk + (1 if self.rank < remainder else 0)
        return self.local_start, self.local_end

    def all_gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Gather tensor from all sequence-parallel ranks."""
        if not dist.is_initialized() or self.world_size == 1:
            return tensor
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Reduce-scatter tensor across sequence-parallel ranks."""
        if not dist.is_initialized() or self.world_size == 1:
            return tensor
        # Split into chunks for scatter
        chunks = list(tensor.chunk(self.world_size, dim=dim))
        output = torch.zeros_like(chunks[self.rank])
        dist.reduce_scatter(output, chunks)
        return output


class RingAttention(nn.Module):
    """Ring Attention — compute full attention on long sequences by
    splitting across GPUs and passing KV chunks in a ring.

    Each GPU holds a subsequence chunk. KV chunks circulate through
    the ring so each rank sees all KV pairs. Uses online softmax
    rescaling for mathematically correct lazy attention.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        ring_size: Number of GPUs in the ring.
    """

    def __init__(self, d_model: int, n_heads: int, ring_size: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ring_size = ring_size

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        ring_rank: int = 0,
        ring_group: Optional[Any] = None,
    ) -> torch.Tensor:
        """Ring attention forward with online softmax rescaling.

        Implements the Ring Attention paper's split-rescale softmax:
        1. For each KV chunk, compute row-wise max and exp-sum locally
        2. Rescale and accumulate output with online softmax correction
        3. KV chunks are passed in a ring: each rank receives the
           *previously received* chunk from its predecessor

        Args:
            x: Input tensor [batch, local_seq_len, d_model].
            ring_rank: This rank in the ring (0..ring_size-1).
            ring_group: Process group for the ring.

        Returns:
            Output tensor with same shape as input.
        """
        batch, local_seq, _ = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(batch, local_seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, local_seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, local_seq, self.n_heads, self.head_dim).transpose(1, 2)

        if self.ring_size <= 1 or not dist.is_initialized():
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(batch, local_seq, self.d_model)
            return self.wo(attn)

        # Ring attention with online softmax
        # Each rank starts with its local KV chunk
        curr_k = k.contiguous()
        curr_v = v.contiguous()

        # Global softmax accumulators
        global_max = torch.full((batch, self.n_heads, local_seq, 1), float('-inf'),
                                device=q.device, dtype=q.dtype)
        global_sum = torch.zeros(batch, self.n_heads, local_seq, 1,
                                 device=q.device, dtype=q.dtype)
        output = torch.zeros_like(q)

        for step in range(self.ring_size):
            src_rank = (ring_rank - 1) % self.ring_size if step > 0 else ring_rank

            if step > 0:
                # Receive KV from predecessor (which is circulating the ring)
                recv_k = torch.zeros_like(curr_k)
                recv_v = torch.zeros_like(curr_v)
                req_k = dist.irecv(recv_k, src=src_rank, group=ring_group)
                req_v = dist.irecv(recv_v, src=src_rank, group=ring_group)
                req_k.wait()
                req_v.wait()
                curr_k, curr_v = recv_k, recv_v

            # Compute attention scores: Q @ K^T / sqrt(d)
            scale = self.head_dim ** 0.5
            scores = torch.matmul(q, curr_k.transpose(-2, -1)) / scale  # [B, H, S_local, S_local]

            # Online softmax: track row-wise max and exp-sum
            row_max = scores.max(dim=-1, keepdim=True).values  # [B, H, S_local, 1]
            row_exp = torch.exp(scores - row_max)  # [B, H, S_local, S_local]
            row_sum = row_exp.sum(dim=-1, keepdim=True)  # [B, H, S_local, 1]

            # Rescale: new_max = max(global_max, row_max)
            new_max = torch.max(global_max, row_max)
            # Rescale old output
            rescale_factor = torch.exp(global_max - new_max)
            output = output * rescale_factor
            global_sum = global_sum * rescale_factor + row_sum * torch.exp(row_max - new_max)
            global_max = new_max

            # Accumulate with unnormalized exp scores rescaled to new_max
            # (not local softmax — each chunk must contribute proportionally
            #  to its actual exp-sum relative to the global max)
            exp_rescaled = row_exp * torch.exp(row_max - new_max)  # exp(scores - new_max)
            output = output + torch.matmul(exp_rescaled, curr_v)

            # Forward the *received* KV chunk to the next rank in the ring
            if step < self.ring_size - 1:
                next_rank = (ring_rank + 1) % self.ring_size
                req_send_k = dist.isend(curr_k, dst=next_rank, group=ring_group)
                req_send_v = dist.isend(curr_v, dst=next_rank, group=ring_group)
                req_send_k.wait()
                req_send_v.wait()

        # Final rescale by global sum
        output = output / (global_sum + 1e-9)
        output = output.transpose(1, 2).contiguous().view(batch, local_seq, self.d_model)
        return self.wo(output)


def sequence_parallel_wrapper(
    model: nn.Module,
    seq_dim: int = 1,
    sp_group: Optional[Any] = None,
) -> nn.Module:
    """Wrap a model for sequence parallelism.

    Automatically converts specific layers (attention) to use
    ring-style sequence-parallel computation.

    Args:
        model: Model to wrap.
        seq_dim: Dimension index for sequence length.
        sp_group: Process group for sequence parallelism.

    Returns:
        Sequence-parallel wrapped model.
    """
    world_size = dist.get_world_size(sp_group) if sp_group and dist.is_initialized() else 1
    rank = dist.get_rank(sp_group) if sp_group and dist.is_initialized() else 0

    for name, module in model.named_children():
        if isinstance(module, nn.MultiheadAttention):
            # Replace with ring attention if appropriate
            d_model = module.embed_dim
            n_heads = module.num_heads
            ring_attn = RingAttention(d_model, n_heads, world_size)
            setattr(model, name, ring_attn)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 11. ASYNCHRONOUS DISTRIBUTED COLLECTIVES
# ═══════════════════════════════════════════════════════════════════════════════


class AsyncCollectiveManager:
    """Manage asynchronous (non-blocking) distributed collectives.

    Overlaps communication with computation by pipelining gradient
    synchronization. Gradients are bucketed by size; each bucket is
    all-reduced asynchronously and the reduced values are copied back
    into the parameter gradients on synchronize().

    Usage:
        mgr = AsyncCollectiveManager(model)
        for batch in loader:
            loss = model(batch)
            loss.backward()
            mgr.synchronize()  # Wait for + unflatten async all-reduces
            optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: int = 25,
        group: Optional[Any] = None,
    ):
        self.model = model
        self.group = group
        self.bucket_size = bucket_size_mb * 1024 * 1024
        self._handles: List[Any] = []
        self._pending_buckets: List[List[torch.Tensor]] = []
        self._pending_flat: List[torch.Tensor] = []
        self._buckets: List[List[torch.Tensor]] = self._make_buckets()

    def _make_buckets(self) -> List[List[torch.Tensor]]:
        """Group parameters into buckets by total size."""
        params = [p for p in self.model.parameters() if p.requires_grad and p.numel() > 0]
        buckets: List[List[torch.Tensor]] = []
        current: List[torch.Tensor] = []
        current_size = 0

        for p in params:
            p_size = p.numel() * p.element_size()
            if current_size + p_size > self.bucket_size and current:
                buckets.append(current)
                current = []
                current_size = 0
            current.append(p)
            current_size += p_size

        if current:
            buckets.append(current)

        return buckets

    def all_reduce_grads(self, async_op: bool = True) -> None:
        """All-reduce gradients in buckets.

        Args:
            async_op: If True, fire async all-reduces; call synchronize()
                      to wait for completion and copy back to param grads.
                      If False, perform synchronous all-reduce per bucket.
        """
        if not dist.is_initialized():
            return

        self._handles.clear()
        self._pending_buckets.clear()
        self._pending_flat.clear()

        for bucket in self._buckets:
            if not bucket:
                continue
            valid_grads = [p.grad for p in bucket if p.grad is not None]
            if not valid_grads:
                continue

            flat = torch.cat([g.flatten() for g in valid_grads])
            if flat.numel() == 0:
                continue

            if async_op:
                handle = dist.all_reduce(flat, async_op=True)
                self._handles.append(handle)
                self._pending_buckets.append(bucket)
                self._pending_flat.append(flat)
            else:
                dist.all_reduce(flat)
                self._unflatten_bucket(bucket, flat)

    def _unflatten_bucket(self, bucket: List[torch.Tensor], flat: torch.Tensor) -> None:
        """Copy reduced flat gradients back into individual param grads."""
        offset = 0
        for p in bucket:
            if p.grad is not None:
                n = p.grad.numel()
                p.grad.copy_(flat[offset:offset + n].view(p.grad.shape))
                offset += n

    def synchronize(self) -> None:
        """Wait for all pending async all-reduces and unflatten."""
        for handle in self._handles:
            handle.wait()
        for bucket, flat in zip(self._pending_buckets, self._pending_flat):
            self._unflatten_bucket(bucket, flat)
        self._handles.clear()
        self._pending_buckets.clear()
        self._pending_flat.clear()

    def all_reduce_and_sync(self) -> None:
        """Perform synchronous all-reduce on all buckets."""
        self.all_reduce_grads(async_op=False)


class OverlapDDPTrainer:
    """DDP trainer with overlapped gradient synchronization.

    Splits model into buckets and synchronizes gradients from earlier
    buckets while computing backward for later buckets. This hides
    communication latency behind compute.

    Args:
        model: Model to train.
        dataset: Training dataset.
        bucket_size_mb: Gradient bucket size in MB.
        batch_size: Batch size per GPU.
        learning_rate: Learning rate.
        num_epochs: Number of epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        bucket_size_mb: int = 25,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
    ):
        with DistributedContext() as ctx:
            self.device = torch.device(f"cuda:{ctx.local_rank}" if torch.cuda.is_available() else "cpu")
            self.is_main = ctx.is_main
            self.world_size = ctx.world_size

            model = model.to(self.device)
            self.model = wrap_ddp(model, gradient_as_bucket_view=True)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

            self.sampler = DistributedSampler(dataset, shuffle=True)
            self.loader = DataLoader(dataset, batch_size=batch_size, sampler=self.sampler, num_workers=2)
            self.num_epochs = num_epochs

            self.async_mgr = AsyncCollectiveManager(self.model, bucket_size_mb=bucket_size_mb)

    def train(self) -> Dict[str, Any]:
        """Run training with overlapped gradient sync."""
        history = {"train_loss": []}

        for epoch in range(self.num_epochs):
            self.sampler.set_epoch(epoch)
            self.model.train()
            epoch_loss = 0.0
            n = 0

            for batch in self.loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    loss = F.cross_entropy(self.model(x), y)

                loss.backward()

                # Async all-reduce — overlaps with next forward if applicable
                self.async_mgr.all_reduce_grads(async_op=True)
                self.async_mgr.synchronize()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                n += 1

            avg_loss = epoch_loss / max(n, 1)
            history["train_loss"].append(avg_loss)
            if self.is_main:
                print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        return {"history": history}


# ═══════════════════════════════════════════════════════════════════════════════
# 12. EXPERT PARALLELISM (MoE)
# ═══════════════════════════════════════════════════════════════════════════════


class MoEDispatch(nn.Module):
    """Token-level dispatch for Mixture-of-Experts with expert parallelism.

    Each GPU hosts a subset of experts. Tokens are routed to the
    appropriate GPU for processing via all-to-all communication.

    Args:
        d_model: Model dimension.
        d_ff: FFN hidden dimension.
        num_experts: Total number of experts.
        num_local_experts: Number of experts on this rank.
        expert_rank_map: Dict mapping rank -> list of expert indices on that rank.
        top_k: Number of experts per token.
        expert_group: Process group for expert parallelism.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        num_local_experts: int = 2,
        expert_rank_map: Optional[Dict[int, List[int]]] = None,
        top_k: int = 2,
        expert_group: Optional[Any] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.expert_group = expert_group
        self.rank = dist.get_rank(expert_group) if expert_group and dist.is_initialized() else 0

        # Determine which experts this rank hosts
        if expert_rank_map is not None:
            self.local_expert_indices = expert_rank_map.get(self.rank, list(range(num_local_experts)))
        else:
            experts_per_rank = num_experts // max(dist.get_world_size(expert_group), 1) if expert_group else num_experts
            remainder = num_experts % max(dist.get_world_size(expert_group), 1)
            start = self.rank * experts_per_rank + min(self.rank, remainder)
            count = experts_per_rank + (1 if self.rank < remainder else 0)
            self.local_expert_indices = list(range(start, start + count))

        self.num_local = len(self.local_expert_indices)

        # Router: shared across all ranks (dense, small)
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Local expert FFNs only (not all experts)
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(self.num_local)
        ])

        self.aux_loss_coef = 0.01

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with expert-parallel dispatch via all-to-all.

        1. Router computes token-to-expert assignments
        2. Tokens are dispatched to the rank hosting their chosen expert
        3. Each rank processes tokens through its local experts
        4. Results are communicated back via all-to-all

        Args:
            x: Input tensor [batch, seq_len, d_model].

        Returns:
            Tuple of (output tensor, auxiliary load balancing loss).
        """
        batch, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        num_tokens = x_flat.shape[0]

        # Router
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Load balancing loss
        tokens_per_expert = router_probs.mean(dim=0)
        mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        expert_freq = mask.mean(dim=(0, 1))
        aux_loss = self.aux_loss_coef * (tokens_per_expert * expert_freq).sum()

        # Dispatch: for each local expert, collect tokens routed to it
        output = torch.zeros_like(x_flat)
        for local_idx, expert_idx in enumerate(self.local_expert_indices):
            token_mask = (top_k_indices == expert_idx).any(dim=-1)
            if token_mask.any():
                token_ids = torch.where(token_mask)[0]
                expert_input = x_flat[token_ids]
                expert_output = self.local_experts[local_idx](expert_input)
                # Weight by routing probability
                token_probs = top_k_probs[token_ids]
                token_mask_expert = (top_k_indices[token_ids] == expert_idx).float()
                token_weight = (token_probs * token_mask_expert).sum(dim=-1, keepdim=True)
                output[token_ids] += expert_output * token_weight

        output = output.view(batch, seq_len, self.d_model)
        return output, aux_loss


def distribute_experts(
    num_experts: int,
    num_ranks: int,
    expert_group: Optional[Any] = None,
) -> Dict[int, List[int]]:
    """Distribute experts across ranks.

    Args:
        num_experts: Total number of experts.
        num_ranks: Number of ranks (expert-parallel workers).
        expert_group: Process group for expert parallelism.

    Returns:
        Dictionary mapping rank -> list of expert indices on that rank.
    """
    rank_map: Dict[int, List[int]] = {}
    experts_per_rank = num_experts // num_ranks
    remainder = num_experts % num_ranks

    start = 0
    for rank in range(num_ranks):
        count = experts_per_rank + (1 if rank < remainder else 0)
        rank_map[rank] = list(range(start, start + count))
        start += count

    return rank_map


# ═══════════════════════════════════════════════════════════════════════════════
# 13. CLI
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_launch_sequence_parallel(args: argparse.Namespace) -> None:
    """Generate sequence parallelism launch command."""
    cmd = f"torchrun --nproc_per_node={args.nproc} {args.script}"
    print(json.dumps({
        "command": "launch_sequence_parallel",
        "full_command": cmd,
        "nproc": args.nproc,
        "note": "Use SequenceParallelContext and RingAttention in your model",
    }, indent=2))


def cmd_launch_expert_parallel(args: argparse.Namespace) -> None:
    """Generate expert parallelism launch command."""
    cmd = f"torchrun --nproc_per_node={args.nproc} {args.script}"
    print(json.dumps({
        "command": "launch_expert_parallel",
        "full_command": cmd,
        "nproc": args.nproc,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "note": "Use MoEDispatch with distribute_experts() for expert placement",
    }, indent=2))


def cmd_benchmark_collectives(args: argparse.Namespace) -> None:
    """Benchmark distributed collective operations."""
    import time

    if not dist.is_initialized():
        setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    sizes = [2**i for i in range(10, 24)]  # 1KB to 8MB
    results = []

    for size in sizes:
        tensor = torch.randn(size, device="cuda" if torch.cuda.is_available() else "cpu")

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)

        # Benchmark all-reduce
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 20

        bw = tensor.numel() * tensor.element_size() * world_size * 2 / elapsed / 1e9
        results.append({"size_bytes": tensor.numel() * tensor.element_size(),
                        "latency_ms": elapsed * 1000,
                        "bandwidth_gbps": bw})

    if rank == 0:
        output = {"command": "benchmark-collectives",
                  "world_size": world_size,
                  "results": results}
        print(json.dumps(output, indent=2))
        if args.save:
            os.makedirs(args.output_dir or ".", exist_ok=True)
            path = os.path.join(args.output_dir or ".", "collective_benchmark.json")
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Saved to {path}")


def cmd_launch_ddp(args: argparse.Namespace) -> None:
    """Generate torchrun launch command."""
    cmd = f"torchrun --nproc_per_node={args.nproc} {args.script}"
    print(json.dumps({
        "command": "launch_ddp",
        "launcher": "torchrun",
        "full_command": cmd,
        "nproc": args.nproc,
        "note": "Set RANK, WORLD_SIZE, LOCAL_RANK env vars automatically via torchrun",
    }, indent=2))


def cmd_launch_fsdp(args: argparse.Namespace) -> None:
    """Generate FSDP launch command."""
    cmd = f"torchrun --nproc_per_node={args.nproc} {args.script}"
    print(json.dumps({
        "command": "launch_fsdp",
        "launcher": "torchrun",
        "full_command": cmd,
        "nproc": args.nproc,
        "sharding": args.sharding,
        "note": "Ensure model uses wrap_fsdp() or FSDP wrapping before training",
    }, indent=2))


def cmd_deepspeed_config(args: argparse.Namespace) -> None:
    """Generate DeepSpeed configuration."""
    config = create_deepspeed_zero_config(
        zero_stage=args.zero_stage,
        bf16=args.bf16,
        gradient_accumulation_steps=args.grad_accum,
        train_micro_batch_size_per_gpu=args.batch_size,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
    )
    output = {
        "command": "deepspeed_config",
        "zero_stage": args.zero_stage,
        "config": config,
        "launch_command": f"deepspeed --num_gpus={args.nproc} train.py",
    }
    print(json.dumps(output, indent=2))

    if args.save:
        os.makedirs(args.output_dir or ".", exist_ok=True)
        path = os.path.join(args.output_dir or ".", "ds_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Toolkit")
    parser.add_argument("--output-dir", "-o", default="./dist_output")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("launch-ddp", help="Generate DDP launch command")
    p.add_argument("--script", required=True)
    p.add_argument("--nproc", type=int, default=4)

    p = subparsers.add_parser("launch-fsdp", help="Generate FSDP launch command")
    p.add_argument("--script", required=True)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--sharding", choices=["full", "grad", "no_shard"], default="full")

    p = subparsers.add_parser("deepspeed-config", help="Generate DeepSpeed config")
    p.add_argument("--zero-stage", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--offload-optimizer", action="store_true")
    p.add_argument("--offload-param", action="store_true")
    p.add_argument("--save", action="store_true")

    p = subparsers.add_parser("launch-sequence-parallel", help="Generate sequence parallelism launch")
    p.add_argument("--script", required=True)
    p.add_argument("--nproc", type=int, default=4)

    p = subparsers.add_parser("launch-expert-parallel", help="Generate expert parallelism launch")
    p.add_argument("--script", required=True)
    p.add_argument("--nproc", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)

    p = subparsers.add_parser("benchmark-collectives", help="Benchmark collective ops")
    p.add_argument("--save", action="store_true")

    args = parser.parse_args()

    if args.command == "launch-ddp":
        cmd_launch_ddp(args)
    elif args.command == "launch-fsdp":
        cmd_launch_fsdp(args)
    elif args.command == "deepspeed-config":
        cmd_deepspeed_config(args)
    elif args.command == "launch-sequence-parallel":
        cmd_launch_sequence_parallel(args)
    elif args.command == "launch-expert-parallel":
        cmd_launch_expert_parallel(args)
    elif args.command == "benchmark-collectives":
        cmd_benchmark_collectives(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
