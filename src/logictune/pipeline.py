"""
Iterative DPO with formal-methods feedback (end-to-end pipeline).

    round 0: evaluate the base model
    round r: sample controllers from policy_{r-1} on training prompts
             -> parse + model check -> preference pairs
             -> DPO (LoRA, merged) -> policy_r
             -> evaluate policy_r on held-out prompts

Every stage writes its outputs under ``run_dir`` and is skipped when its
outputs already exist, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from .data_generator import build_preference_pairs, read_jsonl, score_samples, summarize_records, write_jsonl
from .evaluator import compare, evaluate_model
from .generation import load_model, load_tokenizer, sample_responses
from .config import PipelineConfig
from .prompts import train_prompts


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(cfg: PipelineConfig, run_dir: str) -> Dict:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    t_start = time.time()
    history: List[Dict] = []

    def ev(model_path: str, tag: str):
        return evaluate_model(
            model_path, str(run / "eval"), tag, n_samples=cfg.eval_samples,
            temperature=cfg.eval_temperature, top_p=cfg.eval_top_p, max_new_tokens=cfg.max_new_tokens,
            seed=1234, engine=cfg.engine,
        )

    base_records, base_summary = ev(cfg.base_model, "round0")
    history.append({"round": 0, "model": cfg.base_model, "eval": base_summary})
    policy = cfg.base_model
    items = train_prompts()

    for r in range(1, cfg.rounds + 1):
        rdir = run / f"round{r}"
        rdir.mkdir(exist_ok=True)
        samples_path, pairs_path = rdir / "train_samples.jsonl", rdir / "pairs.jsonl"
        print(f"\n=== Round {r}: policy {policy}", flush=True)

        if not samples_path.exists():
            tok, model = load_tokenizer(policy), load_model(policy)
            responses = sample_responses(
                model, tok, [it.text for it in items], n_per_prompt=cfg.train_samples_per_prompt,
                temperature=cfg.train_temperature, top_p=cfg.train_top_p,
                max_new_tokens=cfg.max_new_tokens, seed=cfg.seed + 1000 * r, batch_size=cfg.gen_batch_size,
            )
            del model
            _free()
            records = score_samples(items, responses, engine=cfg.engine, tag=f"round{r}_train")
            write_jsonl(samples_path, records)
        records = read_jsonl(samples_path)
        data_summary = summarize_records(records)

        if not pairs_path.exists():
            pairs = build_preference_pairs(records, cfg.min_gap, cfg.max_pairs_per_prompt, seed=cfg.seed + r)
            write_jsonl(pairs_path, pairs)
        pairs = read_jsonl(pairs_path)
        data_summary["pairs"] = len(pairs)
        data_summary["prompts_with_pairs"] = len({p["prompt_id"] for p in pairs})
        data_summary["mean_pair_gap"] = sum(p["score_diff"] for p in pairs) / max(1, len(pairs))
        (rdir / "data_summary.json").write_text(json.dumps(data_summary, indent=2))
        print(f"  train samples: rate {data_summary['satisfaction_rate']:.3f}, "
              f"pairs {len(pairs)} from {data_summary['prompts_with_pairs']}/{len(items)} prompts", flush=True)
        if len(pairs) < cfg.min_pairs:
            print(f"  only {len(pairs)} pairs (< {cfg.min_pairs}); stopping", flush=True)
            break

        model_dir = rdir / "model"
        if not (model_dir / "config.json").exists():
            from .trainer import train_dpo

            train_dpo(
                policy, str(pairs_path), str(rdir), epochs=cfg.epochs, learning_rate=cfg.learning_rate,
                beta=cfg.beta, batch_size=cfg.batch_size, grad_accum=cfg.grad_accum, lora_r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha, rpo_alpha=cfg.rpo_alpha, seed=cfg.seed + r,
            )
            _free()
        train_log = json.loads((rdir / "train_log.json").read_text())

        rec, summ = ev(str(model_dir), f"round{r}")
        cmp = compare(base_records, rec)
        history.append({
            "round": r, "model": str(model_dir), "data": data_summary, "eval": summ,
            "vs_base": cmp,
            "train": {k: train_log[k] for k in ("pairs", "optimizer_steps", "train_seconds", "final_train_loss")},
        })
        print(f"  round {r} vs base: {cmp['satisfaction_rate']['diff']:+.3f} "
              f"CI {cmp['satisfaction_rate']['ci95']} p={cmp['satisfaction_rate']['p_value']:.4f}", flush=True)
        policy = str(model_dir)
        (run / "history.json").write_text(json.dumps(history, indent=2))

    summary = {"config": asdict(cfg), "history": history, "wall_seconds": round(time.time() - t_start, 1)}
    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
