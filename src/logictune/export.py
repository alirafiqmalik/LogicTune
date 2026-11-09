"""
Export a pipeline run to plain, versioned files (``results/latest``).

Files written:

* ``summary.json``          headline numbers, per-round metrics, config, provenance
* ``rounds.csv``            one row per round (satisfaction, CI, full pass, pairs, steps)
* ``per_spec.csv``          satisfaction of each LTL spec for each round
* ``per_scenario.csv``      satisfaction of each scenario for each round
* ``eval_samples.csv``      one row per evaluated response, with all 15 verdicts
* ``training_curves.csv``   DPO log history (loss, reward margins, accuracies)
* ``replay.json``           everything the simulator needs to replay recorded controllers

Only light dependencies (no PyTorch), so it runs on a laptop or in CI.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .data_generator import read_jsonl
from .environment import describe_environment
from .prompts import EVAL_TEMPLATES, HELDOUT_SCENARIOS, SYSTEM_PROMPT, scenarios_as_dicts
from .specs import NUM_SPECS, SPEC_IDS, specs_as_dicts

SCHEMA_VERSION = 1


def _round_of(tag: str) -> int:
    m = re.match(r"round(\d+)", tag)
    return int(m.group(1)) if m else -1


def _label(round_idx: int, final_round: int) -> str:
    if round_idx == 0:
        return "Base model"
    if round_idx == final_round:
        return f"Fine-tuned (round {round_idx}, final)"
    return f"Fine-tuned (round {round_idx})"


def export_run(run_dir: str, out_dir: str = "results/latest", label: Optional[str] = None) -> Dict:
    run, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    final_round = max(h["round"] for h in history)

    provenance = {"run_name": run.name, "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    commit_file = run / "source_commit.txt"
    if commit_file.exists():
        provenance["source_commit"] = commit_file.read_text().strip()
    gpus = sorted(run.glob("gpu-*.csv"))
    if gpus:
        lines = [l for l in gpus[-1].read_text().splitlines() if l and not l.startswith("name")]
        provenance["gpu"] = lines[0].split(",")[0].strip() if lines else None
        provenance["slurm_jobs"] = [g.stem.split("-", 1)[1] for g in gpus]

    # ---- per-round tables -------------------------------------------------
    rounds = []
    for h in history:
        e = h["eval"]
        row = {
            "round": h["round"],
            "label": _label(h["round"], final_round),
            "satisfaction_rate": e["satisfaction_rate"],
            "satisfaction_ci95": e["satisfaction_ci95"],
            "full_pass_rate": e["full_pass_rate"],
            "full_pass_ci95": e["full_pass_ci95"],
            "parse_rate": e["parse_rate"],
            "mean_score": e["mean_score"],
            "n_eval": e["n"],
            "seen_satisfaction": e["seen_scenarios"]["satisfaction_rate"],
            "heldout_satisfaction": e["heldout_scenarios"]["satisfaction_rate"],
            "score_histogram": e["score_histogram"],
            "per_spec": e["per_spec"],
            "per_scenario": e["per_scenario"],
        }
        if "data" in h:
            d = h["data"]
            row["train_data"] = {k: d.get(k) for k in ("n", "satisfaction_rate", "parse_rate", "pairs", "prompts_with_pairs", "mean_pair_gap")}
            row["train"] = h.get("train")
            row["vs_base"] = h.get("vs_base")
        rounds.append(row)

    base, final = rounds[0], rounds[-1]
    headline = {
        "base_model": config["base_model"],
        "base_satisfaction": base["satisfaction_rate"],
        "final_satisfaction": final["satisfaction_rate"],
        "final_round": final_round,
        "change_pp": 100 * (final["satisfaction_rate"] - base["satisfaction_rate"]),
        "base_full_pass": base["full_pass_rate"],
        "final_full_pass": final["full_pass_rate"],
    }
    if final.get("vs_base"):
        headline["change_ci95_pp"] = [100 * x for x in final["vs_base"]["satisfaction_rate"]["ci95"]]
        headline["p_value"] = final["vs_base"]["satisfaction_rate"]["p_value"]
        headline["heldout_change_pp"] = 100 * (final["heldout_satisfaction"] - base["heldout_satisfaction"])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "label": label or f"{config['base_model']} - iterative DPO with formal-methods feedback",
        "provenance": provenance,
        "headline": headline,
        "rounds": rounds,
        "config": config,
        "evaluation": {
            "templates": EVAL_TEMPLATES,
            "heldout_scenarios": list(HELDOUT_SCENARIOS),
            "num_specs": NUM_SPECS,
            "metric": "mean fraction of the 15 LTL specifications satisfied per response",
            "ci": "95% cluster bootstrap over evaluation prompts",
            "test": "sign-flip permutation test over prompts (paired)",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(out / "rounds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "label", "satisfaction_rate", "ci95_low", "ci95_high", "full_pass_rate", "parse_rate",
                    "seen_satisfaction", "heldout_satisfaction", "train_pairs", "optimizer_steps", "final_train_loss",
                    "diff_vs_base", "diff_ci95_low", "diff_ci95_high", "p_value"])
        for r in rounds:
            vs = (r.get("vs_base") or {}).get("satisfaction_rate") or {}
            tr = r.get("train") or {}
            w.writerow([r["round"], r["label"], f"{r['satisfaction_rate']:.4f}", f"{r['satisfaction_ci95'][0]:.4f}",
                        f"{r['satisfaction_ci95'][1]:.4f}", f"{r['full_pass_rate']:.4f}", f"{r['parse_rate']:.4f}",
                        f"{r['seen_satisfaction']:.4f}", f"{r['heldout_satisfaction']:.4f}", tr.get("pairs", ""),
                        tr.get("optimizer_steps", ""),
                        f"{tr['final_train_loss']:.4f}" if tr.get("final_train_loss") is not None else "",
                        f"{vs['diff']:.4f}" if vs else "", f"{vs['ci95'][0]:.4f}" if vs else "",
                        f"{vs['ci95'][1]:.4f}" if vs else "", f"{vs['p_value']:.5f}" if vs else ""])

    with open(out / "per_spec.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round"] + SPEC_IDS)
        for r in rounds:
            w.writerow([r["round"]] + [f"{r['per_spec'][s]:.4f}" for s in SPEC_IDS])

    with open(out / "per_scenario.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "scenario_id", "heldout", "n", "satisfaction_rate", "full_pass_rate"])
        for r in rounds:
            for sid, v in r["per_scenario"].items():
                w.writerow([r["round"], sid, int(int(sid) in HELDOUT_SCENARIOS), v["n"],
                            f"{v['satisfaction_rate']:.4f}", f"{v['full_pass_rate']:.4f}"])

    # ---- per-sample records ------------------------------------------------
    models = []
    samples_by_model: Dict[str, List[Dict]] = {}
    for r in rounds:
        tag = f"round{r['round']}"
        path = run / "eval" / f"eval_{tag}.jsonl"
        if not path.exists():
            continue
        samples_by_model[tag] = read_jsonl(path)
        models.append({"id": tag, "round": r["round"], "label": r["label"],
                       "satisfaction_rate": r["satisfaction_rate"], "full_pass_rate": r["full_pass_rate"]})

    with open(out / "eval_samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "round", "prompt_id", "scenario_id", "heldout", "template", "sample_idx", "score",
                    *SPEC_IDS, "n_rules", "n_free_atoms", "response"])
        for tag, recs in samples_by_model.items():
            for x in recs:
                w.writerow([tag, _round_of(tag), x["prompt_id"], x["scenario_id"], int(x["scenario_id"] in HELDOUT_SCENARIOS),
                            int(x["prompt_id"].rsplit("_", 1)[1]), x["sample_idx"], x["score"],
                            *[int(x["specs"][s]) for s in SPEC_IDS], x["n_rules"], x["n_free_atoms"], x["response"]])

    with open(out / "training_curves.csv", "w", newline="") as f:
        keys = ["loss", "rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins",
                "logps/chosen", "logps/rejected", "learning_rate", "grad_norm"]
        w = csv.writer(f)
        w.writerow(["round", "step", "epoch"] + keys)
        for r in rounds[1:]:
            log_path = run / f"round{r['round']}" / "train_log.json"
            if not log_path.exists():
                continue
            for e in json.loads(log_path.read_text())["log_history"]:
                if "loss" not in e:
                    continue
                w.writerow([r["round"], e.get("step"), e.get("epoch")] + [e.get(k, "") for k in keys])

    # ---- simulator replay ----------------------------------------------------
    replay = {
        "schema_version": SCHEMA_VERSION,
        "label": summary["label"],
        "provenance": provenance,
        "headline": headline,
        "environment": describe_environment(),
        "system_prompt": SYSTEM_PROMPT,
        "specs": specs_as_dicts(),
        "scenarios": scenarios_as_dicts(),
        "models": models,
        "rounds": [{k: r[k] for k in ("round", "label", "satisfaction_rate", "satisfaction_ci95", "full_pass_rate",
                                       "parse_rate", "seen_satisfaction", "heldout_satisfaction", "per_spec")}
                   for r in rounds],
        "samples": {
            tag: [
                {
                    "prompt_id": x["prompt_id"],
                    "scenario_id": x["scenario_id"],
                    "template": int(x["prompt_id"].rsplit("_", 1)[1]),
                    "sample_idx": x["sample_idx"],
                    "prompt": x["prompt"],
                    "response": x["response"],
                    "score": x["score"],
                    "specs": [int(x["specs"][s]) for s in SPEC_IDS],
                    "rules": [{"text": rr["text"], "guard": rr["guard"], "guard_text": rr["guard_text"], "action": rr["action"]}
                              for rr in x["controller"]["rules"]],
                    "free_atoms": x["controller"]["unknown_atoms"],
                }
                for x in recs
            ]
            for tag, recs in samples_by_model.items()
        },
    }
    (out / "replay.json").write_text(json.dumps(replay, separators=(",", ":")))
    print(f"exported {run} -> {out} ({len(models)} models, "
          f"{sum(len(v) for v in samples_by_model.values())} evaluated responses)")
    return summary
