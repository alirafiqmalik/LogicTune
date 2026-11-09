"""
LogicTune command-line interface.

    python main.py demo                         # verify example controllers (no GPU)
    python main.py verify --file response.txt   # parse + model check one response
    python main.py pipeline --run-dir runs/x    # base eval -> iterative DPO -> eval (GPU)
    python main.py evaluate --model PATH --tag T --out DIR
    python main.py export --run-dir runs/x      # write results/latest/* for the README and UI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

DEMO_RESPONSES = {
    "Safe, complete controller": """1. If pedestrian_in_front, stop.
2. If stop_sign, stop.
3. If not green_traffic_light, stop.
4. If opposite_car or car_from_left or car_from_right or pedestrian_at_left or pedestrian_at_right, stop.
5. Otherwise, go_straight.""",
    "Natural-language controller": """1. If there is a pedestrian in front of the car, stop.
2. If the light is red or yellow, come to a complete stop.
3. If the light is green and the intersection is clear, proceed straight.
4. Otherwise, stop.""",
    "Unsafe rule order": """1. If the light is green, go straight.
2. If a pedestrian is in front, stop.""",
    "Prose without rules": "The car slows down near the intersection and is prepared to stop if needed.",
}


def cmd_demo(args):
    from logictune import find_nuxmv, format_result, verify_response

    engine = "both" if find_nuxmv() else "explicit"
    print(f"Verification engine: {engine}\n")
    for name, text in DEMO_RESPONSES.items():
        print("=" * 72)
        print(name)
        print("-" * 72)
        print(text.strip(), "\n")
        print(format_result(verify_response(text, engine=engine)), "\n")


def cmd_verify(args):
    from logictune import format_result, verify_response

    text = open(args.file).read() if args.file else args.text
    res = verify_response(text, engine=args.engine)
    print(format_result(res))
    if args.smv:
        from logictune import controller_to_smv

        print("\n" + controller_to_smv(res.controller))


def cmd_pipeline(args):
    from logictune.config import PipelineConfig
    from logictune.pipeline import run_pipeline

    cfg = PipelineConfig()
    for f in fields(PipelineConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    summary = run_pipeline(cfg, args.run_dir)
    print(json.dumps([{"round": h["round"], "satisfaction": h["eval"]["satisfaction_rate"]} for h in summary["history"]], indent=2))


def cmd_evaluate(args):
    from logictune.evaluator import evaluate_model

    evaluate_model(args.model, args.out, args.tag, n_samples=args.n_samples, engine=args.engine)


def cmd_export(args):
    from logictune.export import export_run

    export_run(args.run_dir, args.out, label=args.label)


def main():
    p = argparse.ArgumentParser(description="LogicTune: fine-tuning LMs with formal-methods feedback")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="verify example controllers")

    v = sub.add_parser("verify", help="parse and model check one response")
    v.add_argument("--file")
    v.add_argument("--text")
    v.add_argument("--engine", default="explicit", choices=["explicit", "nuxmv", "both"])
    v.add_argument("--smv", action="store_true", help="also print the generated SMV model")

    pl = sub.add_parser("pipeline", help="run base eval + iterative DPO + eval")
    pl.add_argument("--run-dir", required=True)
    from logictune.config import PipelineConfig

    for f in fields(PipelineConfig):
        flag = "--" + f.name.replace("_", "-")
        typ = {int: int, float: float, str: str}.get(type(f.default), str)
        if f.name == "rpo_alpha":
            typ = float
        pl.add_argument(flag, dest=f.name, type=typ, default=None)

    e = sub.add_parser("evaluate", help="evaluate one model")
    e.add_argument("--model", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--n-samples", type=int, default=10)
    e.add_argument("--engine", default="both")

    x = sub.add_parser("export", help="export a run to results/latest")
    x.add_argument("--run-dir", required=True)
    x.add_argument("--out", default="results/latest")
    x.add_argument("--label", default=None)

    args = p.parse_args()
    {"demo": cmd_demo, "verify": cmd_verify, "pipeline": cmd_pipeline, "evaluate": cmd_evaluate, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
