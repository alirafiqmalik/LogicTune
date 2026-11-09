"""
Batched, seeded sampling from a causal language model.

Every generation call fixes the torch seed per batch, and batches are built
in a fixed order, so a (model, prompts, seed) triple gives the same samples
on the same hardware and software stack.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import chat_prompt


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(path: str, dtype: Optional[torch.dtype] = None):
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
    model.eval()
    return model


@torch.no_grad()
def sample_responses(
    model,
    tokenizer,
    user_messages: Sequence[str],
    n_per_prompt: int,
    temperature: float,
    top_p: float = 0.95,
    max_new_tokens: int = 384,
    seed: int = 0,
    batch_size: int = 32,
    log_every: int = 1,
) -> List[List[str]]:
    """Return ``n_per_prompt`` sampled responses for every user message."""
    chats = [chat_prompt(tokenizer, m) for m in user_messages]
    out: List[List[str]] = []
    t0 = time.time()
    for b in range(0, len(chats), batch_size):
        batch = chats[b : b + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seed + b)
        gen = model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_per_prompt,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for i in range(len(batch)):
            out.append([t.strip() for t in texts[i * n_per_prompt : (i + 1) * n_per_prompt]])
        if log_every and (b // batch_size) % log_every == 0:
            done = min(b + batch_size, len(chats))
            print(f"  generated {done}/{len(chats)} prompts x {n_per_prompt} ({time.time() - t0:.0f}s)", flush=True)
    return out
