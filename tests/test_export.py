"""Export round trip: a small synthetic run -> results files -> re-verification."""

import json
import os
import sys

from logictune.data_generator import score_samples, write_jsonl
from logictune.export import export_run, update_index
from logictune.prompts import eval_prompts
from logictune.stats import compare, summarize_eval

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import check_results  # noqa: E402

BASE = ["The car checks the light and is ready to stop.", "1. If the light is green, go straight."]
TUNED = [
    "1. If pedestrian_in_front, stop.\n2. If stop_sign, stop.\n3. If not green_traffic_light, stop.\n"
    "4. If opposite_car or car_from_left or car_from_right or pedestrian_at_left or pedestrian_at_right, stop.\n"
    "5. Otherwise, go_straight.",
    "1. If not green_traffic_light, stop.\n2. Otherwise, go_straight.",
]


def _fake_run(tmp_path):
    run = tmp_path / "runs" / "fake"
    (run / "eval").mkdir(parents=True)
    items = eval_prompts()
    history = []
    records = {}
    for rnd, texts in ((0, BASE), (1, TUNED)):
        recs = score_samples(items, [texts for _ in items], engine="explicit", tag=f"round{rnd}")
        write_jsonl(run / "eval" / f"eval_round{rnd}.jsonl", recs)
        records[rnd] = recs
        entry = {"round": rnd, "model": f"m{rnd}", "eval": summarize_eval(recs)}
        if rnd:
            entry["vs_base"] = compare(records[0], recs)
            entry["data"] = {"n": 10, "satisfaction_rate": 0.5, "parse_rate": 1.0, "pairs": 5, "prompts_with_pairs": 3, "mean_pair_gap": 4.0}
            entry["train"] = {"pairs": 5, "optimizer_steps": 2, "train_seconds": 1.0, "final_train_loss": 0.5}
        history.append(entry)
    (run / "history.json").write_text(json.dumps(history))
    (run / "config.json").write_text(json.dumps({"base_model": "fake/model"}))
    (run / "source_commit.txt").write_text("0" * 40)
    return run


def test_export_and_recheck(tmp_path):
    run = _fake_run(tmp_path)
    results = tmp_path / "results"
    summary = export_run(str(run), str(results / "fake"))
    update_index(str(results), "fake", summary, make_default=True)

    index = json.loads((results / "index.json").read_text())
    assert index["default"] == "fake" and index["runs"][0]["path"] == "fake/"
    for name in ("summary.json", "rounds.csv", "per_spec.csv", "per_scenario.csv", "eval_samples.csv", "training_curves.csv", "replay.json"):
        assert (results / "fake" / name).exists(), name

    replay = json.loads((results / "fake" / "replay.json").read_text())
    assert [m["id"] for m in replay["models"]] == ["round0", "round1"]
    assert len(replay["samples"]["round1"]) == 2 * len(eval_prompts())
    assert summary["headline"]["final_satisfaction"] > summary["headline"]["base_satisfaction"]
    assert check_results.check_run(str(results / "fake"), "explicit") == 0
