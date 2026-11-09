# Results

`results/latest` holds the exported evaluation of every published run (one folder per run, listed in `index.json`). `results/archive` keeps the 2025-12-02 Colab run, which used the old 3-specification keyword scorer and is kept only as history.

```text
results/
├── latest/
│   ├── index.json                 # runs, default run, headline numbers
│   └── <run>/
│       ├── summary.json           # headline, per-round metrics, config, provenance
│       ├── rounds.csv             # one row per DPO round
│       ├── per_spec.csv           # satisfaction of each LTL spec per round
│       ├── per_scenario.csv       # satisfaction of each scenario per round
│       ├── eval_samples.csv       # one row per evaluated response (all 15 verdicts + text)
│       ├── training_curves.csv    # DPO log history per round
│       ├── replay.json            # everything the simulator replays
│       └── figures/*.png          # rendered by scripts/plot_results.py
└── archive/colab_run_2025-12-02/  # superseded run (3-spec keyword scorer)
```

## `eval_samples.csv`

| Column | Meaning |
|---|---|
| `model`, `round` | `round0` is the base model; `roundN` is the policy after N DPO rounds |
| `prompt_id`, `scenario_id`, `heldout`, `template` | Evaluation prompt (`eNN_T`), its scenario, whether the scenario was held out of training, and the instruction template (0 or 1) |
| `sample_idx` | Sample number (0-9) for this prompt; decoding is seeded |
| `score` | Number of satisfied specifications (0-15) |
| `phi_1` … `phi_15` | 1 = satisfied, 0 = violated (nuXmv verdict) |
| `n_rules`, `n_free_atoms` | Size of the parsed controller and number of unknown phrases |
| `response` | The full recorded model output |

## `replay.json` (abridged)

```json
{
  "schema_version": 1,
  "headline": {"base_model": "...", "base_satisfaction": 0.0, "final_satisfaction": 0.0, "p_value": 0.0},
  "specs": [{"id": "phi_1", "formula": "□(pedestrian_in_front → ◇ stop)", "smv": "...", "kind": "liveness"}],
  "scenarios": [{"id": 1, "text": "...", "intent": "straight", "init": {"green_traffic_light": true}, "heldout": false}],
  "models": [{"id": "round0", "label": "Base model", "satisfaction_rate": 0.0}],
  "rounds": [{"round": 0, "satisfaction_rate": 0.0, "satisfaction_ci95": [0.0, 0.0], "per_spec": {}}],
  "samples": {
    "round0": [{
      "prompt_id": "e01_0", "scenario_id": 1, "sample_idx": 0, "prompt": "...", "response": "...",
      "score": 8, "specs": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
      "rules": [{"text": "...", "guard": {"op": "var", "name": "pedestrian_in_front"}, "guard_text": "pedestrian_in_front", "action": "stop"}],
      "free_atoms": {}
    }]
  }
}
```

## Reproduce and check

```bash
python main.py export --run-dir runs/<run> --run-id <run> [--default]   # write results/latest/<run>/
python scripts/check_results.py          # re-parse + re-verify every recorded response
python scripts/plot_results.py           # render results/latest/<run>/figures/*.png
```

`scripts/check_results.py` also runs in CI and before every Pages deploy, so the published page and the README can only show numbers that the current verifier reproduces from the recorded text. Results are never edited by hand.
