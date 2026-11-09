"""
Render the README figures from the exported results (no GPU, no model).

    python scripts/plot_results.py [--results results/latest]

For every run in <results>/index.json, reads summary.json,
training_curves.csv and eval_samples.csv, and writes PNG figures to
<results>/<run>/figures/.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

csv.field_size_limit(10**8)

# Validated two-series palette (base = slot 1 blue, fine-tuned = slot 2 orange).
BASE, FT, MID = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#75746f", "#e6e5e1", "#fcfcfb"
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK3, "ytick.color": INK3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def pct_axis(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def load(results):
    s = json.load(open(os.path.join(results, "summary.json")))
    rows = list(csv.DictReader(open(os.path.join(results, "eval_samples.csv"), newline="")))
    curves = list(csv.DictReader(open(os.path.join(results, "training_curves.csv"), newline="")))
    return s, rows, curves


def fig_rounds(s, out):
    rounds = s["rounds"]
    x = [r["round"] for r in rounds]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, ci, title in [
        (axes[0], "satisfaction_rate", "satisfaction_ci95", "Specification satisfaction"),
        (axes[1], "full_pass_rate", "full_pass_ci95", "Responses satisfying all 15 specs"),
    ]:
        y = [r[key] for r in rounds]
        lo = [r[ci][0] for r in rounds]
        hi = [r[ci][1] for r in rounds]
        ax.fill_between(x, lo, hi, color=BASE, alpha=0.10, linewidth=0)
        ax.plot(x, y, color=INK2, linewidth=2, solid_capstyle="round", zorder=2)
        colors = [BASE] + [MID] * (len(x) - 2) + [FT] if len(x) > 1 else [BASE]
        ax.scatter(x, y, s=60, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{y[0]:.1%}", (x[0], y[0]), textcoords="offset points", xytext=(10, -16) if y[-1] > y[0] + 0.05 else (10, 8), color=INK, fontsize=10)
        ax.annotate(f"{y[-1]:.1%}", (x[-1], y[-1]), textcoords="offset points", xytext=(-8, 8), ha="right", color=INK, fontsize=10, fontweight="bold")
        ax.set_xticks(x, ["Base"] + [f"Round {i}" for i in x[1:]])
        ax.set_ylim(0, 1)
        pct_axis(ax)
        ax.set_title(title, loc="left")
    fig.text(0.01, 0.01, "Band: 95% cluster bootstrap over evaluation prompts. 340 recorded responses per model.", color=INK3, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "satisfaction_by_round.png"), dpi=160)
    plt.close(fig)


def fig_specs(s, out):
    base, fin = s["rounds"][0], s["rounds"][-1]
    ids = list(base["per_spec"].keys())
    y = np.arange(len(ids))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(y + 0.2, [base["per_spec"][i] for i in ids], height=0.36, color=BASE, label="Base model")
    ax.barh(y - 0.2, [fin["per_spec"][i] for i in ids], height=0.36, color=FT, label=f"Fine-tuned (round {fin['round']})")
    for yy, i in zip(y, ids):
        d = 100 * (fin["per_spec"][i] - base["per_spec"][i])
        ax.text(max(base["per_spec"][i], fin["per_spec"][i]) + 0.015, yy - 0.05, f"{d:+.0f}", color=INK2, fontsize=9, va="center")
    ax.set_yticks(y, ["φ" + i.split("_")[1] for i in ids])
    ax.set_xlim(0, 1.08)
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_title("Satisfaction of each LTL specification", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_spec.png"), dpi=160)
    plt.close(fig)


def fig_scenarios(s, rows, out):
    base_r, fin_r = s["rounds"][0]["round"], s["rounds"][-1]["round"]
    held = set(s["evaluation"]["heldout_scenarios"])
    acc = defaultdict(list)
    for r in rows:
        acc[(int(r["round"]), int(r["scenario_id"]))].append(int(r["score"]) / 15)
    sids = sorted({k[1] for k in acc})
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for k, sid in enumerate(sids[::-1]):
        b, f = np.mean(acc[(base_r, sid)]), np.mean(acc[(fin_r, sid)])
        ax.plot([b, f], [k, k], color=INK3, linewidth=2, zorder=1)
        ax.scatter([b], [k], s=60, color=BASE, edgecolors=SURFACE, linewidths=2, zorder=2)
        ax.scatter([f], [k], s=60, color=FT, edgecolors=SURFACE, linewidths=2, zorder=2)
    ax.set_yticks(range(len(sids)), [f"S{sid} ◆" if sid in held else f"S{sid}" for sid in sids[::-1]])
    for t in ax.get_yticklabels():
        if "◆" in t.get_text():
            t.set_color(INK)
            t.set_fontweight("bold")
    ax.set_xlim(0, 1)
    pct_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], color=BASE, s=40, label="Base model")
    ax.scatter([], [], color=FT, s=40, label=f"Fine-tuned (round {fin_r})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=9)
    ax.set_title("Satisfaction per scenario (◆ = held out of training)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_scenario.png"), dpi=160)
    plt.close(fig)


def fig_scores(s, rows, out):
    base_r, fin_r = s["rounds"][0]["round"], s["rounds"][-1]["round"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, rnd, color, title in [(axes[0], base_r, BASE, "Base model"), (axes[1], fin_r, FT, f"Fine-tuned (round {fin_r})")]:
        sc = [int(r["score"]) for r in rows if int(r["round"]) == rnd]
        counts = np.bincount(sc, minlength=16) / max(1, len(sc))
        ax.bar(range(16), counts, width=0.8, color=color)
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel("Specifications satisfied (of 15)")
        pct_axis(ax)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of responses")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "score_distribution.png"), dpi=160)
    plt.close(fig)


def fig_training(curves, out):
    if not curves:
        return
    rounds = sorted({int(c["round"]) for c in curves})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    greys = plt.cm.Greys(np.linspace(0.45, 0.85, len(rounds)))
    for k, rnd in enumerate(rounds):
        cs = [c for c in curves if int(c["round"]) == rnd]
        ep = [float(c["epoch"]) for c in cs]
        color = FT if rnd == rounds[-1] else greys[k]
        axes[0].plot(ep, [float(c["loss"]) for c in cs], color=color, linewidth=2, label=f"Round {rnd}")
        axes[1].plot(ep, [float(c["rewards/accuracies"]) for c in cs], color=color, linewidth=2, label=f"Round {rnd}")
    axes[0].axhline(np.log(2), color=INK3, linewidth=1)
    axes[0].text(0.02, np.log(2) + 0.01, "ln 2 (no learning)", color=INK3, fontsize=8.5, transform=axes[0].get_yaxis_transform())
    axes[0].set_title("DPO loss", loc="left")
    axes[1].set_title("Preference accuracy on training batches", loc="left")
    pct_axis(axes[1])
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/latest")
    args = ap.parse_args()
    index = json.load(open(os.path.join(args.results, "index.json")))
    for run in index["runs"]:
        run_dir = os.path.join(args.results, run["path"])
        out = os.path.join(run_dir, "figures")
        os.makedirs(out, exist_ok=True)
        s, rows, curves = load(run_dir)
        fig_rounds(s, out)
        fig_specs(s, out)
        fig_scenarios(s, rows, out)
        fig_scores(s, rows, out)
        fig_training(curves, out)
        print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
