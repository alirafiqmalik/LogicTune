"""
DPO training on verifier-ranked preference pairs.

One call trains one round: LoRA adapters on top of the current policy, with
the current policy (adapters disabled) as the DPO reference model. After
training the adapters are merged, so the next round starts from - and uses as
its reference - the policy that produced its data (on-policy iterative DPO).

Pinned stack: transformers 4.46.3, trl 0.12.2, peft 0.13.2.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from .data_generator import read_jsonl
from .generation import load_tokenizer
from .prompts import chat_prompt

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_dpo(
    policy_path: str,
    pairs_path: str,
    output_dir: str,
    epochs: float = 2.0,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    batch_size: int = 8,
    grad_accum: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    rpo_alpha: Optional[float] = None,
    seed: int = 0,
) -> Dict:
    """Train one DPO round and save the merged model to ``output_dir/model``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(policy_path)
    pairs = read_jsonl(Path(pairs_path))
    rows = [{"prompt": chat_prompt(tok, p["prompt"]), "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs]
    dataset = Dataset.from_list(rows).shuffle(seed=seed)
    print(f"DPO round: {len(rows)} pairs from {pairs_path}", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(policy_path, torch_dtype=dtype)
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    args = DPOConfig(
        output_dir=str(out / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        bf16=dtype == torch.bfloat16,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        rpo_alpha=rpo_alpha,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args, train_dataset=dataset,
        processing_class=tok, peft_config=peft_config,
    )
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out / "model", safe_serialization=True)
    tok.save_pretrained(out / "model")

    log = {
        "policy_in": policy_path,
        "pairs": len(rows),
        "optimizer_steps": trainer.state.global_step,
        "train_seconds": round(train_seconds, 1),
        "final_train_loss": result.training_loss,
        "hyperparameters": {
            "epochs": epochs, "learning_rate": learning_rate, "beta": beta,
            "batch_size": batch_size, "grad_accum": grad_accum, "lora_r": lora_r,
            "lora_alpha": lora_alpha, "lora_dropout": lora_dropout, "max_length": max_length,
            "max_prompt_length": max_prompt_length, "rpo_alpha": rpo_alpha, "seed": seed,
        },
        "log_history": trainer.state.log_history,
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    del trainer, model, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log
