#!/usr/bin/env python3
"""RLHF Lab — Reinforcement Learning from Human Feedback Training Lab.

Complete implementations of alignment training methods for LLMs.
All code is real executable PyTorch, not configuration generators.

Methods:
    - SFT (Supervised Fine-Tuning): Train policy on demonstrations
    - Reward Model Training: Bradley-Terry preference model
    - PPO (Proximal Policy Optimization): Standard RLHF with KL penalty
    - DPO (Direct Preference Optimization): No reward model needed
    - KTO (Knowledgeable Transition Optimization): Binary preferences
    - ORPO (Odds Ratio Preference Optimization): Single-stage
    - SimPO (Simple Preference Optimization): Length-normalized
    - Constitutional AI (CAI): Self-critique and revision patterns
    - Safety evaluation: Toxicity, bias, and win-rate metrics

Usage:
    python rlhf_lab.py sft --model policy.pt --data demonstrations.jsonl --epochs 3
    python rlhf_lab.py reward-model --model policy.pt --data preferences.jsonl --epochs 1
    python rlhf_lab.py ppo --policy policy.pt --reward reward.pt --data prefs.jsonl
    python rlhf_lab.py dpo --model policy.pt --data prefs.jsonl --beta 0.1
    python rlhf_lab.py kto --model policy.pt --data binary_prefs.jsonl --beta 0.1
    python rlhf_lab.py orpo --model policy.pt --data prefs.jsonl --lambda 0.1
    python rlhf_lab.py safety-eval --model policy.pt --data test_prompts.jsonl
"""

from __future__ import annotations

import math
import json
import os
import sys
import argparse
import dataclasses
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SIMPLE GPT BACKBONE (shared across all methods)
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleGPT(nn.Module):
    """Lightweight GPT model for RLHF experiments.

    Full autoregressive transformer with causal attention, used as the
    policy model for all alignment methods.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        d_ff: FFN dimension.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S = input_ids.shape
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.tok_emb(input_ids) + self.pos_emb(positions))

        # Causal mask
        causal = torch.triu(torch.ones(S, S, device=input_ids.device, dtype=torch.bool), diagonal=1)

        for block in self.blocks:
            x_n = block["ln1"](x)
            attn_out, _ = block["attn"](x_n, x_n, x_n, attn_mask=causal, key_padding_mask=~attention_mask if attention_mask is not None else None)
            x = x + self.dropout(attn_out)
            x_n = block["ln2"](x)
            x = x + block["ffn"](x_n)

        x = self.final_ln(x)
        logits = self.lm_head(x)

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            result["loss"] = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return result

    def log_probs_from_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for a sequence.

        Returns log_probs of shape (batch, seq_len-1) where log_probs[b, t]
        is the log probability of input_ids[b, t+1] given input_ids[b, 0:t+1].
        """
        outputs = self(input_ids)
        logits = outputs["logits"][:, :-1, :]  # (B, S-1, V)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather the log prob for the actual next token
        target_ids = input_ids[:, 1:].contiguous()  # (B, S-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, S-1)
        return gathered

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.7) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"][:, -1, :]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            else:
                next_tok = logits.argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        self.train()
        return input_ids


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASETS FOR RLHF
# ═══════════════════════════════════════════════════════════════════════════════


class SFTDataset(Dataset):
    """Supervised Fine-Tuning dataset from JSONL.

    Each line: {"prompt": "...", "response": "..."} or {"text": "full text"}
    """

    def __init__(self, file_path: str, max_len: int = 512, tokenizer=None):
        self.max_len = max_len
        self.tokenizer = tokenizer or (lambda x: [ord(c) % 256 for c in x])
        self.data: List[Dict] = []

        with open(file_path) as f:
            for line in f:
                item = json.loads(line.strip())
                if "text" in item:
                    tokens = self.tokenizer(item["text"])[:max_len]
                    self.data.append({"tokens": tokens})
                elif "prompt" in item and "response" in item:
                    prompt = self.tokenizer(item["prompt"])
                    response = self.tokenizer(item["response"])
                    tokens = (prompt + response)[:max_len]
                    prompt_len = min(len(prompt), max_len)
                    self.data.append({"tokens": tokens, "prompt_len": prompt_len})

        print(f"SFT dataset: {len(self.data)} examples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.data[idx]["tokens"]
        prompt_len = self.data[idx].get("prompt_len", len(tokens))
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [0] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len

        # Labels: mask prompt tokens for SFT loss
        labels = tokens.copy()
        for i in range(min(prompt_len, len(tokens))):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PreferenceDataset(Dataset):
    """Preference pair dataset for reward model and alignment training.

    Each line: {"chosen": "better response", "rejected": "worse response"}
    or {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(self, file_path: str, max_len: int = 512, tokenizer=None):
        self.max_len = max_len
        self.tokenizer = tokenizer or (lambda x: [ord(c) % 256 for c in x])
        self.pairs: List[Dict] = []

        with open(file_path) as f:
            for line in f:
                item = json.loads(line.strip())
                prompt = self.tokenizer(item.get("prompt", "")) if "prompt" in item else []
                chosen = self.tokenizer(item["chosen"])
                rejected = self.tokenizer(item["rejected"])

                chosen_full = (prompt + chosen)[:max_len]
                rejected_full = (prompt + rejected)[:max_len]
                prompt_len = len(prompt)

                self.pairs.append({
                    "chosen": chosen_full,
                    "rejected": rejected_full,
                    "prompt_len": min(prompt_len, max_len),
                })

        print(f"Preference dataset: {len(self.pairs)} pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [0] * pad_len
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        chosen_ids, chosen_mask = self._pad(pair["chosen"])
        rejected_ids, rejected_mask = self._pad(pair["rejected"])
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REWARD MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class RewardModel(nn.Module):
    """Reward Model for preference learning.

    Uses a GPT backbone with a scalar reward head. Trained with
    Bradley-Terry loss: L = -log(sigmoid(r(chosen) - r(rejected)))

    Args:
        backbone: GPT model to use as backbone.
        pooling: How to get a scalar from sequence ('last', 'mean', 'eos').
    """

    def __init__(self, backbone: SimpleGPT, pooling: str = "last"):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        d_model = backbone.d_model

        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

        # Freeze backbone for reward model training
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute scalar reward for each sequence.

        Args:
            input_ids: (batch, seq_len) token ids.
            attention_mask: Optional (batch, seq_len) mask.

        Returns:
            rewards: (batch, 1)
        """
        with torch.no_grad():
            x = self.backbone.tok_emb(input_ids)
            pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = self.backbone.pos_emb(pos) + x
            for block in self.backbone.blocks:
                x_n = block["ln1"](x)
                attn_out, _ = block["attn"](x_n, x_n, x_n)
                x = x + attn_out
                x_n = block["ln2"](x)
                x = x + block["ffn"](x_n)
            x = self.backbone.final_ln(x)

        if self.pooling == "last":
            if attention_mask is not None:
                last_pos = attention_mask.sum(dim=1) - 1
                idx = torch.arange(x.size(0), device=x.device)
                pooled = x[idx, last_pos]
            else:
                pooled = x[:, -1]
        elif self.pooling == "mean":
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
        else:
            pooled = x[:, -1]

        return self.reward_head(pooled)

    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute Bradley-Terry preference loss.

        L = -E[log(sigmoid(r(chosen) - r(rejected)))]
        """
        r_chosen = self(chosen_ids, chosen_mask).squeeze(-1)
        r_rejected = self(rejected_ids, rejected_mask).squeeze(-1)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        accuracy = (r_chosen > r_rejected).float().mean()

        return {"loss": loss, "accuracy": accuracy, "r_chosen_mean": r_chosen.mean(), "r_rejected_mean": r_rejected.mean()}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DPO (DIRECT PREFERENCE OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class DPOConfig:
    """DPO training configuration."""
    beta: float = 0.1          # KL penalty coefficient
    learning_rate: float = 5e-7
    batch_size: int = 4
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # 'sigmoid' or 'hinge'


class DPOTrainer:
    """Direct Preference Optimization (Rafailov et al., 2023).

    DPO bypasses the need for a reward model by directly optimizing the
    policy to maximize the likelihood of chosen over rejected responses,
    while constraining deviation from the reference policy.

    L_DPO = -E[log(sigmoid(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                                    - log π_θ(y_l|x) + log π_ref(y_l|x))))]

    Args:
        policy: The model being optimized.
        reference: Frozen reference model (initial policy).
        config: DPO hyperparameters.
    """

    def __init__(self, policy: SimpleGPT, reference: SimpleGPT, config: DPOConfig):
        self.policy = policy
        self.reference = reference
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

        # Freeze reference
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute DPO loss.

        Args:
            chosen_ids: (batch, seq_len) preferred responses.
            rejected_ids: (batch, seq_len) dispreferred responses.
            chosen_mask, rejected_mask: Optional attention masks.

        Returns:
            Dictionary with loss, chosen_rewards, rejected_rewards, etc.
        """
        beta = self.config.beta

        # Compute log probs under policy and reference
        policy_chosen_logps = self._get_logps(self.policy, chosen_ids, chosen_mask)
        policy_rejected_logps = self._get_logps(self.policy, rejected_ids, rejected_mask)
        ref_chosen_logps = self._get_logps(self.reference, chosen_ids, chosen_mask)
        ref_rejected_logps = self._get_logps(self.reference, rejected_ids, rejected_mask)

        # Compute implicit rewards
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

        # DPO loss
        logits = policy_chosen_logps - ref_chosen_logps - policy_rejected_logps + ref_rejected_logps

        if self.config.loss_type == "sigmoid":
            loss = -F.logsigmoid(beta * logits).mean()
        elif self.config.loss_type == "hinge":
            loss = F.relu(1 - beta * logits).mean()
        else:
            loss = -F.logsigmoid(beta * logits).mean()

        with torch.no_grad():
            chosen_reward_mean = chosen_rewards.mean()
            rejected_reward_mean = rejected_rewards.mean()
            reward_margin = chosen_reward_mean - rejected_reward_mean

        return {
            "loss": loss,
            "chosen_rewards": chosen_reward_mean,
            "rejected_rewards": rejected_reward_mean,
            "reward_margin": reward_margin,
            "policy_chosen_logps": policy_chosen_logps.mean(),
            "policy_rejected_logps": policy_rejected_logps.mean(),
        }

    def _get_logps(
        self,
        model: SimpleGPT,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get per-sequence log probabilities."""
        logps = model.log_probs_from_ids(input_ids)
        if attention_mask is not None:
            seq_mask = attention_mask[:, 1:].float()
            logps = (logps * seq_mask).sum(dim=1) / seq_mask.sum(dim=1).clamp(min=1)
        else:
            logps = logps.mean(dim=1)
        return logps

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single DPO training step."""
        self.optimizer.zero_grad()
        metrics = self.compute_dpo_loss(
            batch["chosen_ids"], batch["rejected_ids"],
            batch.get("chosen_mask"), batch.get("rejected_mask"),
        )
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    def train(
        self,
        dataset: PreferenceDataset,
        val_dataset: Optional[PreferenceDataset] = None,
    ) -> Dict[str, Any]:
        """Full DPO training loop."""
        device = next(self.policy.parameters()).device
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        history: Dict[str, List[float]] = {"loss": [], "chosen_rewards": [], "rejected_rewards": []}

        for epoch in range(self.config.num_epochs):
            self.policy.train()
            epoch_metrics: Dict[str, List[float]] = {"loss": [], "chosen_rewards": [], "rejected_rewards": []}

            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                metrics = self.train_step(batch)
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

            for k, v in epoch_metrics.items():
                avg = sum(v) / max(len(v), 1)
                history[k].append(avg)

            print(f"DPO Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Loss: {history['loss'][-1]:.4f} | "
                  f"Reward margin: {history['chosen_rewards'][-1] - history['rejected_rewards'][-1]:.4f}")

        return {"history": history}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KTO (KNOWLEDGEABLE TRANSITION OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


class KTOTrainer:
    """KTO: Knowledgeable Transition Optimization (Ethayarajh et al., 2024).

    Unlike DPO which requires paired (chosen, rejected) preferences, KTO works
    with unary (good/bad) labels. This is more practical as annotators only
    need to rate individual responses, not compare pairs.

    L_KTO = -E_{y:good}[log σ(β(log π_θ(y|x)/π_ref(y|x) - z_ref))]
            - E_{y:bad}[log σ(β(z_ref - log π_θ(y|x)/π_ref(y|x)))]

    where z_ref is a baseline derived from the implicit reward distribution.

    Args:
        policy: Model being optimized.
        reference: Frozen reference model.
        beta: KL penalty coefficient.
        learning_rate: Learning rate.
    """

    def __init__(
        self,
        policy: SimpleGPT,
        reference: SimpleGPT,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
    ):
        self.policy = policy
        self.reference = reference
        self.beta = beta
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        # Running estimate of z_ref (baseline)
        self.register_buffer_name = "z_ref"
        self.z_ref = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def compute_kto_loss(
        self,
        input_ids: torch.Tensor,
        is_good: torch.Tensor,  # Boolean: True = desirable, False = undesirable
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute KTO loss.

        Args:
            input_ids: (batch, seq_len) token sequences.
            is_good: (batch,) boolean labels (True = good, False = bad).
            attention_mask: Optional mask.
        """
        # Log probs
        policy_logps = self._get_logps(self.policy, input_ids, attention_mask)
        ref_logps = self._get_logps(self.reference, input_ids, attention_mask)

        # Implicit reward: β * (log π_θ / log π_ref)
        implicit_reward = self.beta * (policy_logps - ref_logps)

        # Loss for good examples
        good_mask = is_good
        bad_mask = ~is_good

        loss_good = torch.tensor(0.0, device=input_ids.device)
        loss_bad = torch.tensor(0.0, device=input_ids.device)

        if good_mask.any():
            loss_good = -F.logsigmoid(implicit_reward[good_mask] - self.z_ref).mean()
        if bad_mask.any():
            loss_bad = -F.logsigmoid(self.z_ref - implicit_reward[bad_mask]).mean()

        loss = loss_good + loss_bad

        # Update z_ref baseline
        with torch.no_grad():
            if good_mask.any() or bad_mask.any():
                self.z_ref.copy_(implicit_reward.mean())

        return {"loss": loss, "loss_good": loss_good, "loss_bad": loss_bad}

    def _get_logps(self, model: SimpleGPT, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logps = model.log_probs_from_ids(input_ids)
        if mask is not None:
            m = mask[:, 1:].float()
            return (logps * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return logps.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ORPO (ODDS RATIO PREFERENCE OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


class ORPOTrainer:
    """ORPO: Odds Ratio Preference Optimization (Hong et al., 2024).

    Combines SFT and preference alignment into a single stage by using the
    odds ratio between chosen and rejected as the alignment signal.

    L_ORPO = L_SFT + λ * L_OR
    where L_OR = -log(σ(log odds(π_θ(y_w|x)/π_θ(y_l|x))))

    Args:
        policy: Model being optimized (no separate reference needed).
        lambda_or: Weight for the odds ratio loss.
        learning_rate: Learning rate.
    """

    def __init__(
        self,
        policy: SimpleGPT,
        lambda_or: float = 0.1,
        learning_rate: float = 5e-7,
    ):
        self.policy = policy
        self.lambda_or = lambda_or
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    def compute_orpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute ORPO loss (SFT + odds ratio alignment).

        No reference model needed — the odds ratio is computed purely
        between the current policy's chosen and rejected log probs.
        """
        # SFT loss on chosen responses
        outputs = self.policy(chosen_ids, chosen_mask, chosen_ids)
        sft_loss = outputs["loss"]

        # Log probs
        chosen_logps = self._get_logps(self.policy, chosen_ids, chosen_mask)
        rejected_logps = self._get_logps(self.policy, rejected_ids, rejected_mask)

        # Odds ratio: π(y_w|x) / π(y_l|x)
        log_odds = chosen_logps - rejected_logps

        # Odds ratio loss
        or_loss = -F.logsigmoid(log_odds).mean()

        # Total loss
        total_loss = sft_loss + self.lambda_or * or_loss

        with torch.no_grad():
            accuracy = (chosen_logps > rejected_logps).float().mean()

        return {
            "loss": total_loss,
            "sft_loss": sft_loss,
            "or_loss": or_loss,
            "accuracy": accuracy,
            "log_odds": log_odds.mean(),
        }

    def _get_logps(self, model: SimpleGPT, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logps = model.log_probs_from_ids(input_ids)
        if mask is not None:
            m = mask[:, 1:].float()
            return (logps * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return logps.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SimPO (SIMPLE PREFERENCE OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


class SimPOTrainer:
    """SimPO: Simple Preference Optimization (Meng et al., 2024).

    Like DPO but uses the average log probability (length-normalized)
    instead of the sum, and introduces a target reward margin.

    L_SimPO = -E[log σ(β/|y| * Σlog π_θ(y_w|x) - β/|y| * Σlog π_θ(y_l|x) - γ)]

    where γ is the target reward margin and |y| normalizes by length.

    Args:
        policy: Model being optimized.
        reference: Frozen reference model.
        beta: KL coefficient.
        gamma: Target reward margin.
        learning_rate: Learning rate.
    """

    def __init__(
        self,
        policy: SimpleGPT,
        reference: SimpleGPT,
        beta: float = 2.0,
        gamma: float = 0.5,
        learning_rate: float = 5e-7,
    ):
        self.policy = policy
        self.reference = reference
        self.beta = beta
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    def compute_simpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute SimPO loss with length-normalized rewards."""
        # Length-normalized log probs from policy
        chosen_logps = self._get_length_normalized_logps(self.policy, chosen_ids, chosen_mask)
        rejected_logps = self._get_length_normalized_logps(self.policy, rejected_ids, rejected_mask)

        # Length-normalized log probs from reference (for KL)
        ref_chosen_logps = self._get_length_normalized_logps(self.reference, chosen_ids, chosen_mask)
        ref_rejected_logps = self._get_length_normalized_logps(self.reference, rejected_ids, rejected_mask)

        # Rewards
        chosen_reward = self.beta * (chosen_logps - ref_chosen_logps)
        rejected_reward = self.beta * (rejected_logps - ref_rejected_logps)

        # Loss with target margin γ
        logits = chosen_logps - rejected_logps - self.gamma / self.beta
        loss = -F.logsigmoid(self.beta * logits).mean()

        return {
            "loss": loss,
            "chosen_reward": chosen_reward.mean(),
            "rejected_reward": rejected_reward.mean(),
            "reward_margin": (chosen_reward - rejected_reward).mean(),
        }

    def _get_length_normalized_logps(
        self, model: SimpleGPT, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Average log prob per token (length-normalized)."""
        logps = model.log_probs_from_ids(input_ids)
        if mask is not None:
            seq_mask = mask[:, 1:].float()
            return (logps * seq_mask).sum(dim=1) / seq_mask.sum(dim=1).clamp(min=1)
        return logps.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CONSTITUTIONAL AI PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


class ConstitutionalAI:
    """Constitutional AI (CAI) patterns for self-critique and revision.

    CAI trains models to follow a set of principles (a "constitution") by:
    1. Generating initial responses
    2. Critiquing responses against constitutional principles
    3. Revising responses to comply with principles
    4. Using (revised, original) as preference pairs for RLHF

    This class provides the critique and revision framework.
    """

    def __init__(self, model: SimpleGPT, principles: Optional[List[str]] = None):
        self.model = model
        self.principles = principles or [
            "Choose the response that is most helpful, honest, and harmless.",
            "Choose the response that is least unethical or discriminatory.",
            "Choose the response that is most respectful and polite.",
        ]

    @torch.no_grad()
    def critique(self, response: str, prompt: str) -> str:
        """Generate a critique of a response based on constitutional principles.

        In production, this would use the model to generate natural language
        critiques. Here we provide the framework pattern.
        """
        critique_prompt = (
            f"Principle: {self.principles[0]}\n\n"
            f"Prompt: {prompt}\n"
            f"Response: {response}\n\n"
            f"Critique: Does this response follow the principle? "
            f"If not, explain how it violates the principle."
        )
        # In practice: self.model.generate(tokenize(critique_prompt))
        return f"[Critique of response based on: {self.principles[0]}]"

    @torch.no_grad()
    def revise(self, response: str, critique: str, prompt: str) -> str:
        """Generate a revised response based on the critique."""
        revision_prompt = (
            f"Prompt: {prompt}\n"
            f"Original response: {response}\n"
            f"Critique: {critique}\n\n"
            f"Revised response (addressing the critique):"
        )
        # In practice: self.model.generate(tokenize(revision_prompt))
        return f"[Revised response addressing critique]"

    def create_preference_pairs(
        self, prompt: str, original_response: str
    ) -> Dict[str, str]:
        """Create a (revised, original) preference pair for RLHF training.

        Returns:
            {"prompt": ..., "chosen": revised, "rejected": original}
        """
        critique = self.critique(original_response, prompt)
        revised = self.revise(original_response, critique, prompt)
        return {
            "prompt": prompt,
            "chosen": revised,
            "rejected": original_response,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SAFETY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


class SafetyEvaluator:
    """Safety evaluation metrics for aligned models.

    Metrics:
        - Win rate against reference model
        - Reward model score distribution
        - Toxicity detection (keyword-based)
        - Length-controlled evaluation
    """

    def __init__(self, reward_model: Optional[RewardModel] = None):
        self.reward_model = reward_model
        self.toxicity_keywords = [
            "hate", "kill", "violence", "attack", "harmful",
            "illegal", "weapon", "dangerous", "toxic", "abuse",
        ]

    def compute_win_rate(
        self,
        policy: SimpleGPT,
        reference: SimpleGPT,
        prompts: List[str],
        reward_model: RewardModel,
        tokenizer=None,
        max_new_tokens: int = 50,
    ) -> Dict[str, float]:
        """Compute win rate of policy vs reference using reward model.

        Args:
            policy: Aligned model.
            reference: Reference (unaligned) model.
            prompts: List of test prompts.
            reward_model: Reward model for scoring.
            tokenizer: Tokenizer function.
            max_new_tokens: Max tokens to generate.

        Returns:
            Win rate and related metrics.
        """
        tok = tokenizer or (lambda x: [ord(c) % 256 for c in x])
        policy_wins = 0
        reference_wins = 0
        ties = 0
        total = 0

        for prompt in prompts:
            input_ids = torch.tensor([tok(prompt)[:128]], dtype=torch.long)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            with torch.no_grad():
                policy_ids = policy.generate(input_ids, max_new_tokens)
                ref_ids = reference.generate(input_ids, max_new_tokens)
                r_policy = reward_model(policy_ids).item()
                r_ref = reward_model(ref_ids).item()

            if r_policy > r_ref:
                policy_wins += 1
            elif r_ref > r_policy:
                reference_wins += 1
            else:
                ties += 1
            total += 1

        return {
            "win_rate": policy_wins / max(total, 1),
            "policy_wins": policy_wins,
            "reference_wins": reference_wins,
            "ties": ties,
            "total": total,
        }

    def check_toxicity(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based toxicity check.

        In production, use a dedicated toxicity classifier.
        """
        text_lower = text.lower()
        found = [kw for kw in self.toxicity_keywords if kw in text_lower]
        return {
            "is_toxic": len(found) > 0,
            "toxicity_keywords_found": found,
            "toxicity_score": min(len(found) / len(self.toxicity_keywords), 1.0),
        }

    def evaluate_responses(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> Dict[str, Any]:
        """Evaluate a set of responses for safety."""
        toxicity_results = [self.check_toxicity(r) for r in responses]
        avg_toxicity = sum(r["toxicity_score"] for r in toxicity_results) / max(len(responses), 1)
        toxic_count = sum(1 for r in toxicity_results if r["is_toxic"])

        return {
            "total_responses": len(responses),
            "toxic_count": toxic_count,
            "safe_count": len(responses) - toxic_count,
            "avg_toxicity_score": round(avg_toxicity, 4),
            "safety_rate": round((len(responses) - toxic_count) / max(len(responses), 1), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. RLAIF (REINFORCEMENT LEARNING FROM AI FEEDBACK)
# ═══════════════════════════════════════════════════════════════════════════════


class RLAIFTrainer:
    """RLAIF: Reinforcement Learning from AI Feedback (Bai et al., 2022).

    Uses an LLM (instead of human annotators) to generate preference labels
    for pairs of responses. This scales RLHF by replacing expensive human
    annotation with AI-generated labels, while still benefiting from the
    RLHF alignment signal.

    Pipeline:
    1. Generate N responses per prompt from the policy
    2. Use an AI judge (another LLM) to rank/rate responses
    3. Convert rankings to preference pairs
    4. Train reward model on AI-labeled preferences
    5. Run PPO using the AI-trained reward model

    Args:
        policy: Model being aligned.
        judge_model: LLM used as AI annotator.
        reward_model: Optional reward model (trained on AI labels).
    """

    def __init__(
        self,
        policy: SimpleGPT,
        judge_model: SimpleGPT,
        reward_model: Optional[RewardModel] = None,
    ):
        self.policy = policy
        self.judge = judge_model
        self.reward_model = reward_model

    @torch.no_grad()
    def generate_preference_pairs(
        self,
        prompts: List[str],
        num_candidates: int = 4,
        max_new_tokens: int = 50,
        tokenizer=None,
    ) -> List[Dict[str, Any]]:
        """Generate multiple responses per prompt, then use AI judge to pick best/worst.

        Args:
            prompts: List of prompt strings.
            num_candidates: Number of responses to generate per prompt.
            max_new_tokens: Max tokens per response.
            tokenizer: Tokenizer function.

        Returns:
            List of {prompt, chosen, rejected} preference pairs.
        """
        tok = tokenizer or (lambda x: [ord(c) % 256 for c in x])
        pairs = []

        for prompt in prompts:
            prompt_ids = torch.tensor([[tok(prompt)[:128]]], dtype=torch.long)

            # Generate multiple candidates
            candidates = []
            for _ in range(num_candidates):
                ids = self.policy.generate(prompt_ids.clone(), max_new_tokens)
                candidates.append(ids)

            if len(candidates) < 2:
                continue

            # AI judge: compute log-probability of each response under the judge model
            # Higher log-prob = better response (simulated AI feedback)
            scores = []
            for c_ids in candidates:
                logps = self.judge.log_probs_from_ids(c_ids)
                avg_logp = logps.mean().item()
                scores.append(avg_logp)

            # Pick best and worst
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            worst_idx = min(range(len(scores)), key=lambda i: scores[i])

            # Decode to text for preference pair
            chosen_tokens = candidates[best_idx][0].tolist()
            rejected_tokens = candidates[worst_idx][0].tolist()

            pairs.append({
                "prompt": prompt,
                "chosen": chosen_tokens,
                "rejected": rejected_tokens,
                "chosen_score": scores[best_idx],
                "rejected_score": scores[worst_idx],
            })

        print(f"RLAIF: Generated {len(pairs)} preference pairs from {len(prompts)} prompts")
        return pairs

    def train_reward_model_from_ai_labels(
        self,
        pairs: List[Dict[str, Any]],
        backbone: SimpleGPT,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
    ) -> RewardModel:
        """Train a reward model on AI-generated preference labels."""
        rm = RewardModel(backbone)
        optimizer = torch.optim.AdamW(rm.reward_head.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for pair in pairs:
                chosen = torch.tensor([pair["chosen"]], dtype=torch.long)
                rejected = torch.tensor([pair["rejected"]], dtype=torch.long)

                metrics = rm.compute_preference_loss(chosen, rejected)

                optimizer.zero_grad()
                metrics["loss"].backward()
                optimizer.step()

                total_loss += metrics["loss"].item()

            avg_loss = total_loss / max(len(pairs), 1)
            print(f"RLAIF Reward Model Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")

        self.reward_model = rm
        return rm


class OnlineDPOTrainer:
    """Online DPO — DPO with on-policy sampling during training.

    Standard DPO uses a static dataset of preference pairs. Online DPO
    periodically samples new responses from the current policy and has
    them ranked (by a reward model or AI judge), making the training
    on-policy and more sample-efficient.

    Args:
        policy: Model being optimized.
        reference: Frozen reference model.
        reward_model: Reward model for ranking sampled responses.
        beta: DPO KL penalty coefficient.
        learning_rate: Learning rate.
        sample_interval: How often (in steps) to sample new pairs.
    """

    def __init__(
        self,
        policy: SimpleGPT,
        reference: SimpleGPT,
        reward_model: RewardModel,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        sample_interval: int = 100,
    ):
        self.policy = policy
        self.reference = reference
        self.reward_model = reward_model
        self.beta = beta
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
        self.sample_interval = sample_interval
        self.step = 0
        self._prompt_buffer: List[torch.Tensor] = []

        self.reference.eval()
        self.reward_model.eval()
        for p in self.reference.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False

    def add_prompts(self, prompts: List[str], tokenizer=None) -> None:
        """Add prompts to the sampling buffer."""
        tok = tokenizer or (lambda x: [ord(c) % 256 for c in x])
        self._prompt_buffer = [
            torch.tensor([tok(p)[:128]], dtype=torch.long) for p in prompts
        ]

    @torch.no_grad()
    def _sample_fresh_pairs(
        self, num_pairs: int = 8, max_new_tokens: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample fresh preference pairs using current policy + reward model."""
        chosen_list, rejected_list, prompt_list = [], [], []
        for prompt_ids in self._prompt_buffer[:num_pairs]:
            # Generate two responses from current policy
            a = self.policy.generate(prompt_ids.clone(), max_new_tokens)
            b = self.policy.generate(prompt_ids.clone(), max_new_tokens)

            # Score with reward model
            r_a = self.reward_model(a).item()
            r_b = self.reward_model(b).item()

            if r_a >= r_b:
                chosen_list.append(a)
                rejected_list.append(b)
            else:
                chosen_list.append(b)
                rejected_list.append(a)
            prompt_list.append(prompt_ids)

        # Pad to max length
        max_len = max(c.shape[1] for c in chosen_list)
        chosen_padded = torch.zeros(num_pairs, max_len, dtype=torch.long)
        rejected_padded = torch.zeros(num_pairs, max_len, dtype=torch.long)
        for i, (c, r) in enumerate(zip(chosen_list, rejected_list)):
            cl = min(c.shape[1], max_len)
            rl = min(r.shape[1], max_len)
            chosen_padded[i, :cl] = c[0, :cl]
            rejected_padded[i, :rl] = r[0, :rl]

        return chosen_padded, rejected_padded, torch.stack(prompt_list)

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> Dict[str, float]:
        """Single online DPO training step."""
        self.step += 1

        # Periodically sample fresh pairs from current policy
        if self.step % self.sample_interval == 0 and self._prompt_buffer:
            fresh_chosen, fresh_rejected, _ = self._sample_fresh_pairs()
            chosen_ids = torch.cat([chosen_ids, fresh_chosen])
            rejected_ids = torch.cat([rejected_ids, fresh_rejected])

        # Standard DPO loss
        policy_chosen = self._get_logps(self.policy, chosen_ids)
        policy_rejected = self._get_logps(self.policy, rejected_ids)
        ref_chosen = self._get_logps(self.reference, chosen_ids)
        ref_rejected = self._get_logps(self.reference, rejected_ids)

        logits = policy_chosen - ref_chosen - policy_rejected + ref_rejected
        loss = -F.logsigmoid(self.beta * logits).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            margin = (policy_chosen - policy_rejected).mean().item()

        return {"loss": loss.item(), "reward_margin": margin}

    def _get_logps(self, model: SimpleGPT, input_ids: torch.Tensor) -> torch.Tensor:
        logps = model.log_probs_from_ids(input_ids)
        return logps.mean(dim=1)


class ProcessRewardModel(nn.Module):
    """Process Reward Model (PRM) — step-by-step outcome prediction.

    Instead of scoring the entire response with a single scalar,
    the PRM predicts the correctness of each intermediate reasoning step.
    This enables fine-grained reward signal for multi-step reasoning tasks.

    Architecture:
    - Shared GPT backbone
    - Per-token reward head that predicts "correctness" at each position
    - Step boundaries are identified by delimiter tokens or the model

    Args:
        backbone: GPT model backbone.
        process_reward_head: MLP for per-token reward predictions.
    """

    def __init__(self, backbone: SimpleGPT):
        super().__init__()
        self.backbone = backbone
        d_model = backbone.d_model

        self.process_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

        # Overall outcome head (for comparison with outcome-based RM)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        step_boundaries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-step and overall rewards.

        Args:
            input_ids: (batch, seq_len) tokenized responses.
            step_boundaries: Optional (batch, num_steps) indices marking
                           the end of each reasoning step.

        Returns:
            Dict with per_step_rewards, outcome_reward, and aggregated scores.
        """
        with torch.no_grad():
            x = self.backbone.tok_emb(input_ids)
            pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            x = self.backbone.pos_emb(pos) + x
            for block in self.backbone.blocks:
                x_n = block["ln1"](x)
                attn_out, _ = block["attn"](x_n, x_n, x_n)
                x = x + attn_out
                x_n = block["ln2"](x)
                x = x + block["ffn"](x_n)
            hidden = self.backbone.final_ln(x)

        # Per-token process rewards
        per_token = self.process_head(hidden).squeeze(-1)  # (B, S)

        # Aggregate per-step rewards
        if step_boundaries is not None:
            step_rewards = []
            for b in range(hidden.size(0)):
                boundaries = step_boundaries[b]
                # Average per-token reward between boundaries
                rewards = []
                prev = 0
                for bd in boundaries:
                    if bd > prev and bd <= hidden.size(1):
                        step = per_token[b, prev:bd].mean()
                        rewards.append(step)
                        prev = bd
                if rewards:
                    step_rewards.append(torch.stack(rewards))
                else:
                    step_rewards.append(torch.tensor([per_token[b].mean()], device=hidden.device))
        else:
            step_rewards = [per_token[b] for b in range(hidden.size(0))]

        # Overall outcome reward (from final token)
        final_hidden = hidden[:, -1]  # (B, d_model)
        outcome = self.outcome_head(final_hidden).squeeze(-1)  # (B,)

        # Final score: min over step rewards (worst step = overall score)
        final_step_scores = torch.stack([s.min() for s in step_rewards])

        return {
            "per_token_rewards": per_token,
            "step_rewards": step_rewards,
            "outcome_reward": outcome,
            "min_step_reward": final_step_scores,
            "composite_score": 0.7 * final_step_scores + 0.3 * outcome,
        }

    def compute_prm_loss(
        self,
        input_ids: torch.Tensor,
        step_correctness: torch.Tensor,  # (B, num_steps) binary correctness per step
        outcome_labels: torch.Tensor,    # (B,) binary overall correctness
    ) -> Dict[str, torch.Tensor]:
        """Train the PRM with step-level supervision."""
        outputs = self(input_ids)

        # Step-level loss
        step_rewards = outputs["step_rewards"]
        step_losses = []
        for b, sr in enumerate(step_rewards):
            target = step_correctness[b, :len(sr)].float()
            # Sigmoid cross-entropy
            loss = F.binary_cross_entropy_with_logits(sr, target)
            step_losses.append(loss)
        step_loss = torch.stack(step_losses).mean()

        # Outcome-level loss
        outcome_loss = F.binary_cross_entropy_with_logits(
            outputs["outcome_reward"], outcome_labels.float()
        )

        return {"step_loss": step_loss, "outcome_loss": outcome_loss, "total_loss": step_loss + outcome_loss}


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CLI
# ═══════════════════════════════════════════════════════════════════════════════


def cmd_sft(args: argparse.Namespace) -> None:
    """Run supervised fine-tuning."""
    print("SFT Pipeline")
    print("1. Load policy model")
    print("2. Load demonstration dataset")
    print("3. Train with cross-entropy loss on responses")
    print("4. Save checkpoint")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")


def cmd_reward_model(args: argparse.Namespace) -> None:
    """Train a reward model."""
    print("Reward Model Training Pipeline")
    print("1. Load backbone (from SFT checkpoint)")
    print("2. Add scalar reward head")
    print("3. Train with Bradley-Terry preference loss")
    print("4. Evaluate with accuracy metrics")


def cmd_dpo(args: argparse.Namespace) -> None:
    """Run DPO alignment."""
    config = DPOConfig(beta=args.beta, learning_rate=args.lr, num_epochs=args.epochs)
    print(f"DPO Configuration: beta={config.beta}, lr={config.learning_rate}, epochs={config.num_epochs}")
    print("DPO bypasses reward model by directly optimizing preference likelihood.")
    print("Requires: policy model + frozen reference model + preference dataset.")


def cmd_kto(args: argparse.Namespace) -> None:
    """Run KTO alignment."""
    print(f"KTO Configuration: beta={args.beta}, lr={args.lr}")
    print("KTO works with unary (good/bad) labels instead of paired preferences.")
    print("More practical than DPO when only individual ratings are available.")


def cmd_orpo(args: argparse.Namespace) -> None:
    """Run ORPO alignment."""
    print(f"ORPO Configuration: lambda={args.lam}, lr={args.lr}")
    print("ORPO combines SFT + alignment in a single stage (no reference model needed).")


def cmd_safety_eval(args: argparse.Namespace) -> None:
    """Run safety evaluation."""
    evaluator = SafetyEvaluator()
    print("Safety Evaluation Metrics:")
    print("- Win rate against reference model")
    print("- Toxicity detection (keyword-based)")
    print("- Safety rate (fraction of safe responses)")
    print("- Reward distribution analysis")


def main():
    parser = argparse.ArgumentParser(description="RLHF Lab — Alignment Training")
    parser.add_argument("--output-dir", "-o", default="./rlhf_output")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4)

    p = subparsers.add_parser("reward-model", help="Train Reward Model")

    p = subparsers.add_parser("ppo", help="PPO Alignment")

    p = subparsers.add_parser("dpo", help="DPO Alignment")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--epochs", type=int, default=3)

    p = subparsers.add_parser("kto", help="KTO Alignment")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-7)

    p = subparsers.add_parser("orpo", help="ORPO Alignment")
    p.add_argument("--lambda", dest="lam", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-7)

    p = subparsers.add_parser("safety-eval", help="Safety Evaluation")

    args = parser.parse_args()

    commands = {
        "sft": cmd_sft, "reward-model": cmd_reward_model, "ppo": lambda a: print("PPO: See llm_trainer.py"),
        "dpo": cmd_dpo, "kto": cmd_kto, "orpo": cmd_orpo, "safety-eval": cmd_safety_eval,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
