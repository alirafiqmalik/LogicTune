"""
Statistics for satisfaction rates.

Samples that share a prompt are correlated, so every interval resamples
whole prompts (cluster bootstrap). Comparisons between two models use the
same evaluation prompts, so the difference is bootstrapped per prompt, and a
sign-flip permutation test over prompts gives a p-value.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .specs import NUM_SPECS, SPEC_IDS


def _per_prompt(records: Sequence[Dict], value) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for r in records:
        out.setdefault(r["prompt_id"], []).append(value(r))
    return out


def cluster_bootstrap_ci(records: Sequence[Dict], value, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    groups = _per_prompt(records, value)
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    point = sums.sum() / counts.sum()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_difference(base: Sequence[Dict], new: Sequence[Dict], value, n_boot: int = 5000, seed: int = 0) -> Dict:
    """Mean(new) - mean(base) with a prompt-level bootstrap CI and permutation p-value."""
    gb, gn = _per_prompt(base, value), _per_prompt(new, value)
    keys = sorted(set(gb) & set(gn))
    d = np.array([np.mean(gn[k]) - np.mean(gb[k]) for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    signs = rng.choice([-1.0, 1.0], size=(20000, len(d)))
    null = (signs * d).mean(1)
    p = float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_value": p, "n_prompts": len(keys)}


def rate(r: Dict) -> float:
    return r["score"] / NUM_SPECS


def full_pass(r: Dict) -> float:
    return float(r["score"] == NUM_SPECS)


def summarize_eval(records: Sequence[Dict]) -> Dict:
    """Headline numbers, per-spec, per-scenario, and seen/held-out splits."""
    from .prompts import HELDOUT_SCENARIOS

    if not records:
        return {"n": 0}
    sat, sat_lo, sat_hi = cluster_bootstrap_ci(records, rate)
    fp, fp_lo, fp_hi = cluster_bootstrap_ci(records, full_pass)
    per_spec = {s: float(np.mean([r["specs"][s] for r in records])) for s in SPEC_IDS}
    per_scen: Dict[int, Dict] = {}
    for r in records:
        per_scen.setdefault(r["scenario_id"], []).append(r)
    per_scen_sum = {
        sid: {"n": len(rs), "satisfaction_rate": float(np.mean([rate(x) for x in rs])),
              "full_pass_rate": float(np.mean([full_pass(x) for x in rs]))}
        for sid, rs in sorted(per_scen.items())
    }
    seen = [r for r in records if r["scenario_id"] not in HELDOUT_SCENARIOS]
    held = [r for r in records if r["scenario_id"] in HELDOUT_SCENARIOS]
    hist = np.bincount([r["score"] for r in records], minlength=NUM_SPECS + 1)
    return {
        "n": len(records),
        "satisfaction_rate": sat,
        "satisfaction_ci95": [sat_lo, sat_hi],
        "full_pass_rate": fp,
        "full_pass_ci95": [fp_lo, fp_hi],
        "parse_rate": float(np.mean([r["n_rules"] > 0 for r in records])),
        "mean_score": float(np.mean([r["score"] for r in records])),
        "score_histogram": [int(x) for x in hist],
        "per_spec": per_spec,
        "per_scenario": per_scen_sum,
        "seen_scenarios": {"n": len(seen), "satisfaction_rate": float(np.mean([rate(x) for x in seen])) if seen else None},
        "heldout_scenarios": {"n": len(held), "satisfaction_rate": float(np.mean([rate(x) for x in held])) if held else None},
    }
