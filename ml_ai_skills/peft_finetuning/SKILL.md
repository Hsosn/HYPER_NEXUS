# PEFT Fine-Tuning — Parameter-Efficient Fine-Tuning

**Level:** Advanced  
**Category:** AI/ML  
**Runtime:** Python 3.10+ with PyTorch >= 2.0

## Overview

From-scratch implementations of all major parameter-efficient fine-tuning methods.
Each method is real executable PyTorch code — not a wrapper around a library.

## Methods

| Method | Class | Trainable % | Description |
|--------|-------|-------------|-------------|
| LoRA | `LoRALinear` | ~0.1-0.5% | Low-rank weight decomposition ΔW = B@A |
| QLoRA | `QuantizedLoRALinear` | ~0.1% | 4-bit quantized base + LoRA |
| Adapter | `AdapterLayer` | ~1-5% | Bottleneck adapter with residual |
| Prefix Tuning | `PrefixTuning` | ~0.1% | Learnable KV prefixes |
| Prompt Tuning | `PromptTuning` | ~0.01% | Soft virtual tokens |
| BitFit | `apply_bitfit()` | ~0.1% | Bias-only fine-tuning |
| IA3 | `IA3Linear` | ~0.05% | Element-wise K/V scaling |

## Usage

```bash
# Apply LoRA
python peft_finetuning.py apply --method lora --rank 16 --alpha 32

# Apply QLoRA (4-bit)
python peft_finetuning.py apply --method qlora --bits 4 --rank 8

# Apply BitFit (bias-only)
python peft_finetuning.py apply --method bitfit

# Analyze parameter efficiency
python peft_finetuning.py analyze
```

## Programmatic API

```python
from peft_finetuning import apply_peft, LoRALinear, merge_lora_weights

# Apply LoRA to specific modules
model, lora_params = apply_peft(model, method="lora", rank=16, alpha=32,
                                  target_modules=["W_q", "W_k", "W_v"])

# Train only LoRA parameters
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

# Merge for deployment
merge_lora_weights(model)
```

## Key Features
- Weight merging (LoRA → base model)
- Adapter composition (linear, tied, scaled)
- Per-method parameter analysis
- Gradient checkpointing compatible
