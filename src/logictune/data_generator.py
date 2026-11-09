"""
Preference-data generation with formal-verification feedback.

For every training prompt the current policy samples several controllers.
Each one is parsed (GLM2FSA-lite) and model checked against the 15 LTL
specifications. Responses with more satisfied specifications are preferred,
and the resulting (chosen, rejected) pairs train the next policy with DPO.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from .prompts import PromptItem
from .specs import NUM_SPECS
from .verifier import verify_response


def score_samples(
    items: Sequence[PromptItem],
    responses: Sequence[Sequence[str]],
    engine: str = "both",
    tag: str = "",
) -> List[Dict]:
    """Verify every sampled response; return one record per response."""
    records = []
    for item, resp_list in zip(items, responses):
        for k, text in enumerate(resp_list):
            res = verify_response(text, engine=engine)
            records.append(
                {
                    "tag": tag,
                    "prompt_id": item.prompt_id,
                    "scenario_id": item.scenario_id,
                    "split": item.split,
                    "sample_idx": k,
                    "prompt": item.text,
                    "response": text,
                    "score": res.score,
                    "specs": res.specs,
                    "n_rules": len(res.controller.rules),
                    "n_free_atoms": len(res.controller.unknown_atoms),
                    "smv_hash": res.smv_hash,
                    "controller": res.controller.to_json(),
                }
            )
    return records


def build_preference_pairs(
    records: Sequence[Dict],
    min_gap: int = 1,
    max_pairs_per_prompt: int = 4,
    seed: int = 0,
) -> List[Dict]:
    """
    Turn scored samples into DPO pairs.

    Per prompt, every (higher score, lower score) combination with a gap of
    at least ``min_gap`` is a candidate. Candidates are taken in order of
    decreasing gap, using each rejected response at most once and each chosen
    response at most twice, up to ``max_pairs_per_prompt`` pairs.
    """
    rng = random.Random(seed)
    by_prompt: Dict[str, List[Dict]] = {}
    for r in records:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    pairs = []
    for pid, recs in by_prompt.items():
        uniq, seen = [], set()
        for r in recs:
            key = r["response"].strip()
            if key and key not in seen:
                seen.add(key)
                uniq.append(r)
        cands = []
        for a in uniq:
            for b in uniq:
                gap = a["score"] - b["score"]
                if gap >= min_gap:
                    cands.append((gap, rng.random(), a, b))
        cands.sort(key=lambda c: (-c[0], c[1]))
        used_chosen: Counter = Counter()
        used_rejected = set()
        n = 0
        for gap, _, a, b in cands:
            if n >= max_pairs_per_prompt:
                break
            if id(b) in used_rejected or used_chosen[id(a)] >= 2:
                continue
            used_rejected.add(id(b))
            used_chosen[id(a)] += 1
            pairs.append(
                {
                    "prompt_id": pid,
                    "scenario_id": a["scenario_id"],
                    "prompt": a["prompt"],
                    "chosen": a["response"],
                    "rejected": b["response"],
                    "score_chosen": a["score"],
                    "score_rejected": b["score"],
                    "score_diff": gap,
                }
            )
            n += 1
    return pairs


def summarize_records(records: Sequence[Dict]) -> Dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    scores = [r["score"] for r in records]
    return {
        "n": n,
        "mean_score": sum(scores) / n,
        "satisfaction_rate": sum(scores) / (n * NUM_SPECS),
        "full_pass_rate": sum(s == NUM_SPECS for s in scores) / n,
        "parse_rate": sum(r["n_rules"] > 0 for r in records) / n,
        "score_histogram": dict(sorted(Counter(scores).items())),
    }


def write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
