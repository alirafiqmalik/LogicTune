"""
Evaluation: specification satisfaction on held-out prompts.

Each model answers every evaluation prompt (17 scenarios x 2 unseen
instruction templates) ``n_samples`` times with a fixed seed. Every response
is parsed and model checked. The satisfaction rate is the mean fraction of
the 15 specifications satisfied, with a prompt-level bootstrap interval.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .data_generator import read_jsonl, score_samples, write_jsonl
from .generation import load_model, load_tokenizer, sample_responses
from .prompts import eval_prompts
from .stats import compare, summarize_eval  # noqa: F401  (compare re-exported)


def evaluate_model(
    model_path: str,
    out_dir: str,
    tag: str,
    n_samples: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 384,
    seed: int = 1234,
    batch_size: int = 34,
    engine: str = "both",
    reuse: bool = True,
) -> Tuple[List[Dict], Dict]:
    out = Path(out_dir)
    samples_path = out / f"eval_{tag}.jsonl"
    if reuse and samples_path.exists():
        records = read_jsonl(samples_path)
    else:
        items = eval_prompts()
        tok = load_tokenizer(model_path)
        model = load_model(model_path)
        print(f"Evaluating {tag} ({model_path}) on {len(items)} prompts x {n_samples}", flush=True)
        responses = sample_responses(
            model, tok, [it.text for it in items], n_per_prompt=n_samples, temperature=temperature,
            top_p=top_p, max_new_tokens=max_new_tokens, seed=seed, batch_size=batch_size,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        records = score_samples(items, responses, engine=engine, tag=tag)
        write_jsonl(samples_path, records)
    summary = summarize_eval(records)
    summary.update({"tag": tag, "model_path": model_path, "decoding": {
        "temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens,
        "n_samples": n_samples, "seed": seed}})
    (out / f"eval_{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  {tag}: satisfaction {summary['satisfaction_rate']:.3f} "
          f"[{summary['satisfaction_ci95'][0]:.3f}, {summary['satisfaction_ci95'][1]:.3f}], "
          f"full pass {summary['full_pass_rate']:.3f}, parse {summary['parse_rate']:.3f}", flush=True)
    return records, summary
