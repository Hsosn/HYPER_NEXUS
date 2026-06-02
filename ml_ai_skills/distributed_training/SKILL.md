# Distributed Training — Advanced Distributed Training Skill

**Level:** Advanced  
**Category:** AI/ML  
**Runtime:** Python 3.10+ with PyTorch >= 2.0

## Overview

Production-grade distributed training patterns for scaling model training across
multiple GPUs and nodes. Includes DDP, FSDP, DeepSpeed ZeRO, gradient checkpointing,
torch.compile, pipeline parallelism, and elastic training.

## Capabilities

### Core Patterns
| Pattern | Class | Description |
|---------|-------|-------------|
| DDP | `DDPTrainer` | Full DDP training loop with grad accumulation, AMP, logging |
| FSDP | `wrap_fsdp()` | FSDP wrapping with auto-wrap policies and CPU offload |
| DeepSpeed | `create_deepspeed_zero_config()` | ZeRO Stage 1/2/3 with offload |
| Pipeline | `build_pipeline_stages()` | Split model into sequential stages |

### Memory Optimization
- **Gradient Checkpointing**: `CheckpointedTransformer` trades compute for memory (5x reduction)
- **Mixed Precision**: bf16/fp16 with native AMP
- **CPU Offload**: ZeRO-Offload and ZeRO-Infinity patterns

### Production Features
- **Elastic Training**: `ElasticTrainer` handles node failures with auto-restart
- **Custom Samplers**: `SortedLengthSampler` minimizes padding waste
- **torch.compile**: `compile_model()` with multiple optimization modes
- **Checkpoint Management**: Save/resume across restarts

## Usage

### Launch DDP training
```bash
torchrun --nproc_per_node=4 train_script.py
```

### Programmatic DDP
```python
from distributed_training import DDPTrainer
trainer = DDPTrainer(model, dataset, batch_size=8, use_bf16=True)
result = trainer.train()
```

### FSDP
```python
from distributed_training import wrap_fsdp
model = wrap_fsdp(model, sharding_strategy="full", mixed_precision="bf16")
```

### DeepSpeed config
```bash
python distributed_training.py deepspeed-config --zero-stage 3 --bf16 --offload-param --save
```

### Gradient Checkpointing
```python
from distributed_training import CheckpointedTransformer
model = CheckpointedTransformer(d_model=768, n_layers=24, checkpoint_every_n=2)
```

## Key Classes

| Class | Description |
|-------|-------------|
| `DistributedContext` | Context manager for init/cleanup |
| `DDPTrainer` | Complete DDP training loop |
| `CheckpointedTransformer` | Selective gradient checkpointing |
| `ElasticTrainer` | Fault-tolerant training |
| `SortedLengthSampler` | Length-sorted distributed sampler |
| `PipelineStage` | Pipeline parallelism stage |

## Requirements
- Python >= 3.10
- PyTorch >= 2.0
- NCCL backend for GPU training
- (Optional) DeepSpeed for ZeRO optimization
