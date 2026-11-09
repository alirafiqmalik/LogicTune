"""
Reproduce the paper's experiment: iterative DPO with formal-methods feedback.

    python reproduce_paper.py --run-dir runs/tinyllama            # 3 rounds (GPU)
    python reproduce_paper.py --run-dir runs/quick --quick        # 1 round, fewer samples

The paper (Yang et al., 2023) reports that specification satisfaction rose
from about 60% to about 90%. That figure is the paper's; the numbers this
repository measured are in results/latest/summary.json.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from logictune.config import PipelineConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default=PipelineConfig.base_model)
    p.add_argument("--quick", action="store_true", help="1 round, 4 samples per prompt, 4 eval samples")
    args = p.parse_args()

    cfg = PipelineConfig(base_model=args.model)
    if args.quick:
        cfg.rounds, cfg.train_samples_per_prompt, cfg.eval_samples = 1, 4, 4

    from logictune.pipeline import run_pipeline

    summary = run_pipeline(cfg, args.run_dir)
    base = summary["history"][0]["eval"]["satisfaction_rate"]
    last = summary["history"][-1]
    print("\nSpecification satisfaction (held-out prompts):")
    for h in summary["history"]:
        lo, hi = h["eval"]["satisfaction_ci95"]
        print(f"  round {h['round']}: {h['eval']['satisfaction_rate']:.1%}  (95% CI {lo:.1%} - {hi:.1%})")
    if "vs_base" in last:
        d = last["vs_base"]["satisfaction_rate"]
        print(f"Change vs base: {d['diff']:+.1%} (95% CI {d['ci95'][0]:+.1%} .. {d['ci95'][1]:+.1%}, p = {d['p_value']:.4f})")
    print(f"Paper (for reference, not measured here): ~60% -> ~90%. This run: {base:.1%} -> {last['eval']['satisfaction_rate']:.1%}")
    print(json.dumps({"run_dir": args.run_dir}, indent=2))


if __name__ == "__main__":
    main()
