# RLHF Lab — Reinforcement Learning from Human Feedback Training Lab

**Level:** Advanced  
**Category:** AI/ML  
**Runtime:** Python 3.10+ with PyTorch >= 2.0

## Overview

Complete implementations of modern LLM alignment methods. All code is real
executable PyTorch — not configuration generators.

## Alignment Methods

| Method | Class | Needs RM? | Needs Pairs? | Description |
|--------|-------|-----------|--------------|-------------|
| SFT | `SimpleGPT` + standard training | No | No | Supervised fine-tuning |
| Reward Model | `RewardModel` | — | Yes | Bradley-Terry preference model |
| PPO | See `llm_trainer.py` | Yes | No | Classic RLHF |
| DPO | `DPOTrainer` | No | Yes | Direct preference optimization |
| KTO | `KTOTrainer` | No | No* | Unary label optimization |
| ORPO | `ORPOTrainer` | No | Yes | Combined SFT + alignment |
| SimPO | `SimPOTrainer` | No | Yes | Length-normalized DPO |
| CAI | `ConstitutionalAI` | No | Generated | Self-critique and revision |

*KTO uses unary good/bad labels instead of paired comparisons.

## Safety Evaluation

`SafetyEvaluator` provides:
- **Win rate**: Policy vs reference using reward model
- **Toxicity detection**: Keyword-based screening
- **Safety rate**: Fraction of safe responses

## Constitutional AI

`ConstitutionalAI` implements:
1. Self-critique against constitutional principles
2. Response revision based on critiques
3. Automatic preference pair generation

## Usage

```bash
# DPO alignment
python rlhf_lab.py dpo --beta 0.1 --epochs 3

# KTO (unary labels)
python rlhf_lab.py kto --beta 0.1

# ORPO (single-stage)
python rlhf_lab.py orpo --lambda 0.1

# Safety evaluation
python rlhf_lab.py safety-eval
```

## Programmatic API

```python
from rlhf_lab import SimpleGPT, DPOTrainer, DPOConfig, SafetyEvaluator

policy = SimpleGPT(vocab_size=50257, d_model=512)
reference = SimpleGPT(vocab_size=50257, d_model=512)
reference.load_state_dict(policy.state_dict())

trainer = DPOTrainer(policy, reference, DPOConfig(beta=0.1))
result = trainer.train(preference_dataset)
```

## Requirements
- Python >= 3.10
- PyTorch >= 2.0
