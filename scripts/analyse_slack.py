#!/usr/bin/env python3
"""Figures for the slack investigation.

    results/slack_vs_budget.png   control and gap vs training steps, per depth.
                                  The crossing point is the budget at which the
                                  experiment becomes measurable at all.
    results/slack_by_init.png     slack vs budget for each q(X) initialisation,
                                  answering whether the slack is an init artefact.

Usage:  python scripts/analyse_slack.py
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE, AMORT, GREY = "#4A5A8C", "#0F7A93", "#98A2AC"


def mean_sd(xs):
    if not xs:
        return float("nan"), 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def budget_figure(path, outdir):
    rows = json.load(open(path))["rows"]
    depths = sorted({r["depth"] for r in rows})
    steps = sorted({r["steps"] for r in rows})
    fig, axes = plt.subplots(1, len(depths), figsize=(4.8 * len(depths), 3.8),
                             squeeze=False, sharey=True)
    for i, d in enumerate(depths):
        ax = axes[0][i]
        for scheme, c, lab in [("free_form", FREE, "free-form control (slack)"),
                               ("amortised", AMORT, "amortised gap")]:
            ms, ss = [], []
            for s in steps:
                m, sd = mean_sd([r["gap"] for r in rows if r["depth"] == d
                                 and r["steps"] == s and r["scheme"] == scheme])
                ms.append(m); ss.append(sd)
            ax.errorbar(steps, ms, yerr=ss, marker="o", ms=4, lw=1.9, capsize=3,
                        color=c, ls="--" if scheme == "free_form" else "-", label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training steps"); ax.set_title(f"depth {d}")
        if i == 0:
            ax.set_ylabel("nats / point (log scale)")
            ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The gap is only measurable where the control has decayed below it",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "slack_vs_budget.png"), dpi=200)
    plt.close(fig); print("wrote slack_vs_budget.png")


def init_figure(path, outdir):
    rows = json.load(open(path))["rows"]
    cfgs, steps = [], sorted({r["steps"] for r in rows})
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.get_cmap("viridis")
    for j, c in enumerate(cfgs):
        ms = [mean_sd([r["slack"] for r in rows if r["config"] == c
                       and r["steps"] == s])[0] for s in steps]
        kl0 = next(r["kl_init"] for r in rows if r["config"] == c)
        ax.plot(steps, ms, marker="o", ms=4, lw=1.9,
                color=cmap(j / max(1, len(cfgs) - 1)),
                ls="-" if "amortised" in c else "--",
                label=f"{c.split('  ')[0]}  (KL@init {kl0:.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training steps"); ax.set_ylabel("slack (nats / point)")
    ax.set_title("Slack decays with training budget, not with initialisation")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "slack_by_init.png"), dpi=200)
    plt.close(fig); print("wrote slack_by_init.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/slack_study.json")
    ap.add_argument("--init", default="results/why_slack.json")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if os.path.exists(a.sweep):
        budget_figure(a.sweep, a.outdir)
    else:
        print(f"skipped slack_vs_budget.png ({a.sweep} not found)")
    if os.path.exists(a.init):
        init_figure(a.init, a.outdir)
    else:
        print(f"skipped slack_by_init.png ({a.init} not found)")


if __name__ == "__main__":
    main()
