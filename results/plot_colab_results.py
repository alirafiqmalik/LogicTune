"""
Render the figures for the 2025-12-02 Colab run.

This script only plots numbers transcribed into
results/colab_run_2025-12-02/run_results.json from the saved cell outputs of
Colab_Experiments.ipynb. It does not load a model, generate text, or verify
anything, so it cannot change the results it draws.

Usage:
    python results/plot_colab_results.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = Path(__file__).parent / "colab_run_2025-12-02" / "run_results.json"
FIGURE_DIR = Path(__file__).parent / "figures"

BASE_COLOR = "#4C72B0"
TUNED_COLOR = "#C44E52"
NOISE_COLOR = "#8C8C8C"
PAPER_COLOR = "#55A868"


def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)


def experiment(results, name):
    for exp in results["experiments"]:
        if exp["name"] == name:
            return exp
    raise KeyError(name)


def annotate_bars(ax, bars, values, fmt="{:.1f}%", offset=0.8):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_satisfaction_by_iteration(results, out_path):
    """Iterative run, with the base model's measurement spread drawn behind it."""
    exp = experiment(results, "reproduce_paper.py")
    labels = [c["label"] for c in exp["checkpoints"]]
    rates = [c["satisfaction_rate_pct"] for c in exp["checkpoints"]]
    x = np.arange(len(labels))

    base_measurements = results["repeated_measurements"][
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]["measurements_pct"]
    noise_low, noise_high = min(base_measurements), max(base_measurements)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axhspan(
        noise_low,
        noise_high,
        color=NOISE_COLOR,
        alpha=0.22,
        zorder=0,
        label=(
            f"Untrained base model, {len(base_measurements)} independent evaluations "
            f"({noise_low:.1f}–{noise_high:.1f}%)"
        ),
    )

    ax.plot(x, rates, "-o", color=TUNED_COLOR, linewidth=2.4, markersize=10, zorder=3)
    for xi, rate in zip(x, rates):
        ax.annotate(
            f"{rate:.1f}%",
            (xi, rate),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    pairs = [c.get("preference_pairs_used") for c in exp["checkpoints"]]
    sublabels = [
        label if p is None else f"{label}\n({p} preference pairs)"
        for label, p in zip(labels, pairs)
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(sublabels, fontsize=10)
    ax.set_ylabel("Specification satisfaction rate (%)")
    ax.set_ylim(85, 102)
    ax.set_title(
        "Specification satisfaction across DPO iterations\n"
        "TinyLlama-1.1B-Chat, 3 safety specs, 20 generations per point (Colab L4, 2025-12-02)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_measurement_noise(results, out_path):
    """Repeated scoring of unchanged checkpoints, next to the observed effect sizes."""
    repeated = results["repeated_measurements"]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    base = repeated["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    tuned = repeated["dpo_model"]
    groups = [
        ("Base model\n(unchanged weights)", base, BASE_COLOR),
        ("Fine-tuned dpo_model\n(unchanged weights)", tuned, TUNED_COLOR),
    ]

    for idx, (label, group, color) in enumerate(groups):
        values = group["measurements_pct"]
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax_left.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=130,
            color=color,
            zorder=3,
            edgecolor="white",
            linewidth=1.4,
        )
        ax_left.vlines(idx, min(values), max(values), color=color, linewidth=2.2, alpha=0.55)
        for xi, value in zip(np.full(len(values), idx) + jitter, values):
            ax_left.annotate(
                f"{value:.1f}",
                (xi, value),
                textcoords="offset points",
                xytext=(11, -4),
                fontsize=9,
            )
        ax_left.annotate(
            f"spread {group['spread_pct']:.1f} pp",
            (idx, max(values)),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax_left.set_xticks(range(len(groups)))
    ax_left.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax_left.set_ylabel("Specification satisfaction rate (%)")
    ax_left.set_ylim(88, 104)
    ax_left.set_title(
        "Same checkpoint, scored more than once\n(sampled decoding, no fixed seed)", fontsize=12
    )
    ax_left.grid(axis="y", alpha=0.3)

    effects = [
        (exp["name"].replace(" --mode", "\n--mode"), exp["headline"]["improvement_pct"])
        for exp in results["experiments"]
    ]
    names = [e[0] for e in effects]
    deltas = [e[1] for e in effects]
    x = np.arange(len(names))

    bars = ax_right.bar(x, deltas, color=TUNED_COLOR, alpha=0.85, width=0.55)
    ax_right.axhline(0, color="black", linewidth=1)
    ax_right.axhspan(
        -base["spread_pct"],
        base["spread_pct"],
        color=NOISE_COLOR,
        alpha=0.22,
        label=f"+/- base-model spread ({base['spread_pct']:.1f} pp)",
    )
    for bar, delta in zip(bars, deltas):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            delta - 0.55,
            f"{delta:+.1f} pp",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    within_noise = sum(1 for d in deltas if abs(d) <= base["spread_pct"])
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(names, fontsize=9)
    ax_right.set_ylabel("Change after fine-tuning (percentage points)")
    ax_right.set_ylim(-11, 4)
    ax_right.set_title(
        "Measured effect of fine-tuning\n"
        f"All {len(deltas)} are negative, {within_noise} of {len(deltas)} "
        "smaller than the noise",
        fontsize=12,
    )
    ax_right.grid(axis="y", alpha=0.3)
    ax_right.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_base_vs_finetuned(results, out_path):
    """Base and final satisfaction for each of the three experiments."""
    exps = results["experiments"]
    names = [exp["name"].replace(" --mode", "\n--mode") for exp in exps]
    base_rates = [exp["headline"]["base_pct"] for exp in exps]
    final_rates = [exp["headline"]["final_pct"] for exp in exps]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_base = ax.bar(x - width / 2, base_rates, width, label="Base model", color=BASE_COLOR)
    bars_final = ax.bar(
        x + width / 2, final_rates, width, label="After DPO fine-tuning", color=TUNED_COLOR
    )
    annotate_bars(ax, bars_base, base_rates)
    annotate_bars(ax, bars_final, final_rates)

    deltas = [exp["headline"]["improvement_pct"] for exp in exps]
    for xi, (base_rate, final_rate, delta) in enumerate(zip(base_rates, final_rates, deltas)):
        ax.annotate(
            f"{delta:+.1f} pp",
            (xi, max(base_rate, final_rate) + 6),
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=TUNED_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Specification satisfaction rate (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title(
        "Measured satisfaction rate, three experiments from one session\n"
        "20 generations per bar, 3 safety specs (Colab L4, 2025-12-02)",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_measured_vs_paper_claim(results, out_path):
    """The paper's claimed 60->90% next to what this repository actually measured."""
    exp = experiment(results, "reproduce_paper.py")
    headline = exp["headline"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2)
    width = 0.36

    before = [60.0, headline["base_pct"]]
    after = [90.0, headline["final_pct"]]
    deltas = [30.0, headline["improvement_pct"]]

    bars_before = ax.bar(x - width / 2, before, width, label="Before fine-tuning", color=BASE_COLOR)
    bars_after = ax.bar(x + width / 2, after, width, label="After fine-tuning", color=TUNED_COLOR)
    annotate_bars(ax, bars_before, before)
    annotate_bars(ax, bars_after, after)

    for xi, delta in zip(x, deltas):
        ax.annotate(
            f"{delta:+.1f} pp",
            (xi, 108),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=PAPER_COLOR if delta > 0 else TUNED_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Paper's reported figure\n(not reproduced here)",
            "This repository, measured\n(reproduce_paper.py)",
        ],
        fontsize=10,
    )

    ax.set_ylabel("Specification satisfaction rate (%)")
    ax.set_ylim(0, 132)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title(
        "The 60% -> 90% figure is the paper's claim, not a measurement from this run\n"
        "The two groups are also not comparable: 15 specs in the paper vs 3 specs here",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", ncol=2, fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_preference_pair_yield(results, out_path):
    """Why so little was learned: nearly all candidate generations tied, yielding no pairs."""
    rounds = [
        ("reproduce_paper.py\niteration 1", 24, 3),
        ("reproduce_paper.py\niteration 2", 24, 9),
        ("main.py --mode full", 24, 6),
    ]
    names = [r[0] for r in rounds]
    candidates = [r[1] for r in rounds]
    pairs = [r[2] for r in rounds]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_c = ax.bar(
        x - width / 2, candidates, width, label="Candidate generations scored", color=BASE_COLOR
    )
    bars_p = ax.bar(
        x + width / 2, pairs, width, label="Usable preference pairs produced", color=TUNED_COLOR
    )
    annotate_bars(ax, bars_c, candidates, fmt="{:.0f}", offset=0.35)
    annotate_bars(ax, bars_p, pairs, fmt="{:.0f}", offset=0.35)

    for xi, (cand, pair) in enumerate(zip(candidates, pairs)):
        ax.annotate(
            f"{pair / cand:.0%} yield",
            (xi, max(cand, pair) + 2.6),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=TUNED_COLOR,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_ylim(0, 31)
    ax.set_title(
        "Preference-pair yield per training round\n"
        "Most candidates tie at 3/3 under a 3-spec verifier, so they cannot form a pair",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    results = load_results()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figures = [
        ("specification_satisfaction_by_iteration.png", plot_satisfaction_by_iteration),
        ("measurement_noise_vs_effect.png", plot_measurement_noise),
        ("base_vs_finetuned_by_experiment.png", plot_base_vs_finetuned),
        ("measured_vs_paper_claim.png", plot_measured_vs_paper_claim),
        ("preference_pair_yield.png", plot_preference_pair_yield),
    ]

    for filename, plot_fn in figures:
        out_path = FIGURE_DIR / filename
        plot_fn(results, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
