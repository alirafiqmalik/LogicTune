"""Pipeline configuration (importable without PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    rounds: int = 3
    # data generation
    train_samples_per_prompt: int = 8
    train_temperature: float = 1.0
    train_top_p: float = 0.95
    min_gap: int = 1
    max_pairs_per_prompt: int = 4
    min_pairs: int = 32
    # DPO
    epochs: float = 2.0
    learning_rate: float = 5e-5
    beta: float = 0.1
    batch_size: int = 8
    grad_accum: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    rpo_alpha: Optional[float] = None
    # evaluation
    eval_samples: int = 10
    eval_temperature: float = 0.7
    eval_top_p: float = 0.9
    max_new_tokens: int = 384
    gen_batch_size: int = 32
    engine: str = "both"
    seed: int = 0
