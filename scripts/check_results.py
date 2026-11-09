"""
Re-verify the committed results.

Every response in results/latest/eval_samples.csv is parsed and model
checked again, and the per-model satisfaction rates are recomputed. The
script fails if any recorded score, verdict, or summary number differs, so
the simulator and README can only show numbers that the verifier reproduces.

    python scripts/check_results.py [--results results/latest] [--engine explicit|both]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
csv.field_size_limit(10**8)

from logictune.specs import NUM_SPECS, SPEC_IDS  # noqa: E402
from logictune.verifier import verify_response  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/latest")
    ap.add_argument("--engine", default="explicit", choices=["explicit", "nuxmv", "both"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.results, "eval_samples.csv"), newline="")))
    summary = json.load(open(os.path.join(args.results, "summary.json")))
    replay = json.load(open(os.path.join(args.results, "replay.json")))

    mismatches, per_model = 0, defaultdict(list)
    for r in rows:
        res = verify_response(r["response"], engine=args.engine)
        recorded = {s: r[s] == "1" for s in SPEC_IDS}
        if res.score != int(r["score"]) or res.specs != recorded:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH {r['model']} {r['prompt_id']}#{r['sample_idx']}: recorded {r['score']}, now {res.score}")
        per_model[int(r["round"])].append(res.score / NUM_SPECS)

    for rnd in summary["rounds"]:
        got = sum(per_model[rnd["round"]]) / len(per_model[rnd["round"]])
        if abs(got - rnd["satisfaction_rate"]) > 1e-9:
            mismatches += 1
            print(f"MISMATCH round {rnd['round']}: summary {rnd['satisfaction_rate']:.6f}, recomputed {got:.6f}")
        print(f"round {rnd['round']}: {len(per_model[rnd['round']])} responses, satisfaction {got:.4f}")

    n_replay = sum(len(v) for v in replay["samples"].values())
    if n_replay != len(rows):
        mismatches += 1
        print(f"MISMATCH replay.json has {n_replay} samples, eval_samples.csv has {len(rows)}")

    print(f"checked {len(rows)} recorded responses with engine={args.engine}: {mismatches} mismatches")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
