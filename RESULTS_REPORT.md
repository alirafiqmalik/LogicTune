# LogicTune — Results Report

**Scope.** This report documents what the LogicTune pipeline actually measured, and answers three outstanding review questions about the claims in the README.

**Source of truth.** One Colab session, **2025-12-02**, on an **NVIDIA L4**, whose cell outputs are saved in [`Colab_Experiments.ipynb`](Colab_Experiments.ipynb). The notebook has not been re-run. Its numbers are transcribed into [`results/colab_run_2025-12-02/run_results.json`](results/colab_run_2025-12-02/run_results.json), and every figure in this repository is rendered from that JSON by [`results/plot_colab_results.py`](results/plot_colab_results.py).

**Version caveat that applies to everything below.** The notebook cloned the repository at commit `69c17e3`. At that commit the verifier checked **3** safety specifications per response, not the 15 (Φ1–Φ15) documented in the README. That is why every evaluation reports a score out of 60 for 20 responses. The 15-specification suite was introduced afterwards in `112185a` and **has never been run on a GPU.**

---

## Answers to the three questions

### 1. Is 60→90% your measured result, or the paper's?

**It is the paper's. Drop it as a result claim, or attribute it explicitly.**

The string exists in this codebase only as a hardcoded comparison target printed by `reproduce_paper.py`:

```python
print(f"Paper reports: 60% → 90% (30% improvement)")
```

It is not computed from anything. The README previously presented it as "60% → 90% as reported in paper" attached to the `reproduce_paper.py` instructions, which reads as a prediction of what the script will score. It is not.

**What our run actually scored:**

| Experiment | Base model | After DPO | Change |
|---|---|---|---|
| `reproduce_paper.py` (2 iterations) | 96.7% | 93.3% | **−3.3 pp** |
| `main.py --mode full` (1 iteration) | 95.0% | 93.3% | **−1.7 pp** |
| `main.py --mode evaluate` (re-score of the same two checkpoints) | 100.0% | 91.7% | **−8.3 pp** |

The closest analogue to the paper's headline number is the first row: **96.7% → 93.3%**, a change of **−3.3 pp**. Fine-tuning made the model slightly worse on the verifier, in all three experiments.

Two things to know before swapping that number in:

- **It is not comparable to the paper's 60→90%.** Ours is over 3 specifications; the paper's is over 15. Different denominators, different tasks.
- **It is not distinguishable from noise** (see question 3). The honest version of this bullet carries no improvement number at all.

**Suggested wording:** "The pipeline runs end to end. In our one recorded run, DPO fine-tuning did not improve specification satisfaction (96.7% → 93.3% on a 3-specification verifier, within measurement noise). The paper reports 60% → 90%; we have not reproduced that."

### 2. Did you ever use RLAF here, or does it belong to different work?

**Not in this repository. Removing it was correct.**

`RLAF` — and `RLAIF`, `RLHF`, and `reinforcement` — appear **zero times** anywhere in the repo: no source file, no notebook, no config, no docs, and no commit message in the full 29-commit history. There is no PPO/GRPO/reward-model code path and no reward model in `requirements.txt`.

The only preference-learning method implemented is **DPO**, via `trl`'s `DPOTrainer` with a `DPOConfig` (`beta=0.1`) and LoRA adapters through `peft`. DPO is specifically the *reward-model-free* alternative to RL-based preference tuning, so "RLAF" would be a mischaracterization of the method here even informally.

If RLAF belongs to something, it is not this codebase. Keep it out unless a separate artifact is produced.

### 3. How many controller generations were evaluated?

**20 controller generations per model evaluation; 140 evaluation generations in total across the session.**

The evaluation harness (`ModelEvaluator.evaluate_model`) uses a fixed list of **10** test prompts. `main.py` passes `n_samples_per_prompt = max(1, n_samples // 10)`, which is **2** at the default `--n-eval-samples 20`. So:

> 10 prompts × 2 samples = **20 generations**, scored against 3 specifications = **60 points maximum**

which matches the `Total Score: 58/60` style lines throughout the notebook. Seven model evaluations were performed, giving **140** evaluation generations.

Full accounting of every generation in the session:

| Stage | Generations |
|---|---|
| `reproduce_paper.py` — training-data generation (2 rounds × 6 prompts × 4 samples) | 48 |
| `reproduce_paper.py` — evaluation (3 × 20) | 60 |
| `main.py --mode full` — training-data generation (6 prompts × 4 samples) | 24 |
| `main.py --mode full` — single ad-hoc test generation | 1 |
| `main.py --mode full` — evaluation (2 × 20) | 40 |
| `main.py --mode evaluate` — evaluation (2 × 20) | 40 |
| **Total model generations** | **213** |
| of which **evaluation** generations | **140** |
| of which **training-data** generations | **72** |

Other dataset sizes worth stating alongside it, because they are smaller than the generation count suggests:

| Quantity | Value |
|---|---|
| Distinct evaluation prompts | **10** (2 samples each) |
| Distinct training prompts | **6** (hardcoded at the run commit) |
| Usable preference pairs trained on | **18** total (3 + 9 + 6) |
| Optimizer steps taken | **3, 6, 3** per round |
| Hand-written demo controllers verified | 2 |

The 10-prompt evaluation set is the number that matters for the reliability of the headline percentages, and it is the one the README omitted.

---

## Report summary

**The pipeline works; the result does not reproduce.** All five stages — environment model, response generation, formal scoring, preference-pair construction, DPO fine-tuning — execute end to end on a GPU without intervention. But across three experiments in the recorded session, fine-tuning changed specification satisfaction by −3.3, −1.7, and −8.3 pp. Every measured effect is negative. There is no evidence of the paper's improvement in this repository.

**The measured effects are inside the noise floor.** Nothing in the pipeline sets a seed, and evaluation decodes with `do_sample=True` at temperature 0.7. The unmodified base model was scored three times during the session and returned **96.7%, 95.0%, and 100.0%** — a 5.0 pp spread on identical weights, wider than two of the three effects attributed to fine-tuning. With 60 points per evaluation, one specification check is worth 1.7 pp, so the three reported deltas are a net movement of 2, 1, and 5 checks. These numbers cannot support a claim in either direction.

**The mechanism failed for a diagnosable reason: preference-pair starvation.** DPO needs a score gap between two responses to a shared prompt. On a 3-specification verifier the base model already scores 95–100%, so most candidates tie at 3/3 and are discarded — in the first training round, 5 of 6 prompts produced **zero** pairs. Across the session, 72 candidate generations yielded **18** usable pairs, and training ran for 3, 6, and 3 optimizer steps. DPO loss finished at **0.688, 0.700, 0.694**, all within 0.008 of `ln 2 ≈ 0.693`, the value at initialization when the policy is identical to the reference. **The model barely moved.** The negative deltas are best read as sampling noise around an unchanged policy, not as evidence that the method harms safety.

**A saturated verifier is the root cause, and it is also a validity problem.** The paper's setup starts near 60%, leaving 40 points of headroom. Ours starts at 95–100% with almost none. The verifier is also not a model checker despite the framing: `score_response()` at `HEAD` runs `parse_controller_rules()` and keyword heuristics in `check_critical_violations()` over the raw response text, and the automaton machinery (`get_traffic_safety_specs()`, `build_product_automaton()`, `verify_safety_specs()`) is retained only as no-op stubs. A 3/3 score reflects phrasing as much as policy safety, and is not a proof of anything.

**What would make the next run informative**, roughly in order of expected value:

1. **Fix the measurement before touching the method.** Set a seed, raise evaluation to a few hundred generations, and evaluate each checkpoint several times to report a confidence interval. Until the noise floor is below the effect being claimed, no result from this pipeline is interpretable.
2. **Restore headroom.** Run the 15-specification verifier (already at `HEAD`, never GPU-tested) and confirm the base model scores well below saturation. If it still saturates, the specifications are too easy to satisfy by phrasing.
3. **Fix pair yield.** Use the larger prompt sets now in the repo (`prompts.json`, 85; `prompts_200.json`, 200) and report the pair yield per round as a first-class metric. A round that produces 3 pairs should be treated as a failed round, not trained on.
4. **Train long enough to move the loss.** 3–6 optimizer steps cannot fine-tune a 1.1B model. DPO loss departing from `ln 2` is the minimum bar for claiming that training happened at all.
5. **Replace text heuristics with real model checking.** `src/modelchecker/traffic_specs.smv` and a 52 MB `nuXmv` binary are already vendored, but no Python file references either — there is no `subprocess` call to the model checker anywhere in the repo. Wiring them up is the difference between measuring phrasing and measuring safety; until then, "specification satisfaction rate" overstates what is being computed.

**Recommended disposition for the README bullet:** ship it without a number, describing the pipeline as implemented and the reproduction as outstanding. The infrastructure is a real contribution; the 60→90% figure belongs to the paper and should be attributed to it explicitly wherever it appears.
