# Transformer Architect — Advanced Transformer Architecture Design

**Level:** Advanced  
**Category:** AI/ML  
**Runtime:** Python 3.10+ with PyTorch >= 2.0

## Overview

Complete implementations of production-grade transformer architectures with modern
attention mechanisms, position encodings, and normalization techniques. Every class
is real executable PyTorch code.

## Architectures

### GPT-2 (Decoder-Only)
- Full GPT-2 model with causal multi-head attention
- Pre-LN architecture, GELU FFN, weight tying
- Supports training (with labels) and autoregressive generation

### BERT (Encoder-Only)
- Bidirectional encoder with segment embeddings
- MLM (Masked Language Modeling) head
- NSP (Next Sentence Prediction) head
- [CLS] token pooling for classification tasks

### T5 (Encoder-Decoder)
- Encoder-decoder with cross-attention
- Shared token embeddings
- Text-to-text transfer learning
- Built-in autoregressive generation

## Modern Components

### Attention Variants
| Type | Class | Description |
|------|-------|-------------|
| Multi-Head | `FlashAttention` | Uses PyTorch native SDPA (FlashAttention-2) |
| Grouped Query | `GroupedQueryAttention` | GQA: n_heads Q, fewer KV heads |
| Multi-Query | `MultiQueryAttention` | MQA: all Q heads share 1 KV head |
| KV-Cache | `AttentionWithKVCache` | Efficient autoregressive inference |

### Position Encodings
- **RoPE** (`RotaryPositionalEmbedding`): Used in LLaMA, Mistral, PaLM
- **Learned** (`nn.Embedding`): Standard approach in GPT-2, BERT

### Normalization
- **RMSNorm**: Used in LLaMA, Mistral (faster than LayerNorm)
- **LayerNorm**: Standard (used in GPT-2, BERT)

### Activation
- **SwiGLU** (`SwiGLU`): Outperforms GELU in FFN (used in LLaMA, PaLM)
- **GELU**: Standard activation in GPT-2, BERT
- **ReLU**: Used in T5

### Modern Block
- `ModernTransformerBlock`: LLaMA-style with RMSNorm + GQA + RoPE + SwiGLU

## Parallelism Patterns
- `ColumnParallelLinear`: Splits output dimension across GPUs
- `RowParallelLinear`: Splits input dimension across GPUs

## Usage

```bash
# Build GPT-2
python transformer_architect.py build --type gpt2 --d-model 768 --n-layers 12

# Build BERT
python transformer_architect.py build --type bert --vocab-size 30522

# Build T5 encoder-decoder
python transformer_architect.py build --type t5 --n-layers 6 --d-model 512

# Build modern LLaMA-style blocks
python transformer_architect.py build --type modern --n-heads 32 --n-kv-heads 8

# Test attention variants
python transformer_architect.py attention --attn-type gqa --n-heads 32 --n-kv-heads 8
python transformer_architect.py attention --attn-type mqa --n-heads 32 --causal
```

## Requirements
- Python >= 3.10
- PyTorch >= 2.0 (for `scaled_dot_product_attention`)
