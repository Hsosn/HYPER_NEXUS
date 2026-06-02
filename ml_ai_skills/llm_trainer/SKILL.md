# LLM Trainer — Advanced Large Language Model Training

**Level:** Advanced  
**Category:** AI/ML  
**Runtime:** Python 3.10+ with PyTorch >= 2.0

## Overview

The LLM Trainer provides a complete, executable toolkit for training and fine-tuning
GPT-style transformer language models. Unlike config generators, this is **real executable
PyTorch code** with actual model implementations, training loops, and distributed patterns.

## Capabilities

### Core Architecture
- **GPT Transformer** — Full GPT-2 style model from scratch (`GPT` class)
  - Multi-head self-attention with causal masking
  - Sinusoidal and learned positional encodings
  - Pre-LN transformer blocks with residual connections
  - Weight tying between embedding and output projection
  - Autoregressive generation with temperature, top-k, top-p sampling

### Fine-Tuning
- **LoRA (from scratch)** — `LoRALinear` class implements Low-Rank Adaptation
  - Freezes original weights, adds trainable low-rank A and B matrices
  - Configurable rank, alpha, dropout
  - Weight merging for deployment
- **QLoRA** — `NF4QuantizedLinear` + LoRA on 4-bit quantized base
  - 4-bit NormalFloat quantization with per-row scaling
  - On-the-fly dequantization during forward pass
- **Full fine-tuning** with gradient accumulation

### RLHF Pipeline
- **Reward Model** — `RewardModel` class for preference learning
  - GPT backbone + scalar reward head
  - Bradley-Terry preference ranking loss
- **PPO Trainer** — `PPOTrainer` for alignment
  - Clipped surrogate objective
  - KL divergence penalty from reference model
  - GAE (Generalized Advantage Estimation)
  - Entropy bonus for exploration

### Advanced Training
- **Mixed precision** (bf16/fp16) with native PyTorch AMP
- **Gradient checkpointing** for memory efficiency
- **torch.compile** optimization
- **DDP and FSDP** distributed training
- **DeepSpeed ZeRO** Stage 1/2/3 configuration
- **Cosine LR schedule** with linear warmup
- **Checkpoint management** with save/resume

## Usage Examples

### Train GPT from scratch
```bash
python llm_trainer.py train-gpt --config my_config.json --data train.jsonl
```

### LoRA fine-tuning
```bash
python llm_trainer.py lora-finetune --base-model base_model.pt \
    --data finetune_data.jsonl --rank 16 --alpha 32
```

### RLHF alignment pipeline
```bash
python llm_trainer.py rlhf
```

### Export model
```bash
python llm_trainer.py export --checkpoint model.pt --format torchscript
```

### Programmatic usage
```python
from llm_trainer import GPT, apply_lora_to_model, train_language_model, TrainingConfig

model = GPT(vocab_size=50257, d_model=768, n_heads=12, n_layers=12)
model, lora_params = apply_lora_to_model(model, rank=8, alpha=16)

result = train_language_model(model, train_dataset, config=TrainingConfig())
```

## Key Classes

| Class | Description |
|-------|-------------|
| `GPT` | Full GPT-style autoregressive LM |
| `MultiHeadSelfAttention` | Scaled dot-product attention with causal mask |
| `TransformerBlock` | Pre-LN transformer block |
| `LoRALinear` | LoRA-wrapped Linear layer |
| `NF4QuantizedLinear` | 4-bit quantized linear for QLoRA |
| `RewardModel` | Preference reward model |
| `PPOTrainer` | PPO alignment trainer |
| `CosineLRScheduleWithWarmup` | LR scheduler with warmup |
| `TextDataset` | Text dataset for LM training |
| `PreferenceDataset` | Preference pair dataset |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- (Optional) deepspeed for ZeRO optimization
- (Optional) transformers for tokenizer integration

## Distributed Training

Launch with torchrun:
```bash
torchrun --nproc_per_node=4 llm_trainer.py train-gpt --config config.json
```

With DeepSpeed:
```bash
deepspeed --num_gpus=4 llm_trainer.py train-gpt --config config.json
```
