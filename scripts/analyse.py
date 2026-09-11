#!/usr/bin/env python3
"""Turn results/results.json into the table and plots for the deck.

Usage:
    python scripts/analyse.py [--results results/results.json] [--outdir results]

Produces:
    results/table.md            comparison table (paste into the slide)
    results/gap_vs_depth.png    the main figure, WITH the MC noise floor shaded
    results/gap_convergence.png gap vs refinement steps -- is the gap converged?
    results/elbo_curves.png     convergence evidence for both schemes
    results/ard_weights.png     what each layer kept
    results/latent_space.png    q(X) means, coloured by the leading PC of Y

Nothing here invents a number. If a field is missing from results.json the
corresponding panel is skipped and says so, rather than being filled in.
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FREE, AMORT = "#4A5A8C", "#0F7A93"
GREY = "#98A2AC"


def mean_sd(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def input_relevance(row):
    """Per-latent-dimension ARD relevance at the first GP layer (1/lengthscale)."""
    ls = row.get("ard", {}).get("layer_0")
    if not ls:
        return None
    rows_ = ls if isinstance(ls[0], list) else [ls]
    n_in = len(rows_[0])
    return [sum(1.0 / (r[j] + 1e-8) for r in rows_) / len(rows_) for j in range(n_in)]


# ---------------------------------------------------------------- table -----
def write_table(rows, depths, outdir):
    has_test = any("heldout_elbo" in r for r in rows)
    head = ["Depth", "Scheme", "ELBO/pt", "Held-out/pt", "Gap (nats/pt)",
            "MC floor", "q params", "Embed cost", "Wall-clock (s)"]
    if not has_test:
        head = [h for h in head if h not in ("Held-out/pt", "Embed cost")]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for d in depths:
        for scheme in ["free_form", "amortised"]:
            sel = [r for r in rows if r["depth"] == d and r["scheme"] == scheme]
            if not sel:
                continue
            e, es = mean_sd([r["elbo"] for r in sel])
            g, gs = mean_sd([r["gap"] for r in sel if "gap" in r])
            floor, _ = mean_sd([r.get("gap_noise_sd", r.get("elbo_noise_sd", 0.0)) for r in sel])
            w, _ = mean_sd([r["wall_clock_s"] for r in sel])
            if scheme == "free_form":
                cvals = [r["control_gap"] for r in sel if "control_gap" in r]
                c, cs = mean_sd(cvals)
                gtxt = (f"{c:+.3f} ± {cs:.3f} (control)" if cvals
                        else "0 (by construction)")
            else:
                ctrl = [r["control_gap"] for r in rows
                        if r["depth"] == d and "control_gap" in r]
                gtxt = f"{g:+.3f} ± {gs:.3f}"
                if ctrl:
                    gtxt += f" (net {g - sum(ctrl) / len(ctrl):+.3f})"
            if scheme == "amortised" and sel and not all(
                    r.get("refine_converged", True) for r in sel):
                gtxt += " (lower bound)"
            cells = [str(d), scheme, f"{e:+.3f} ± {es:.3f}"]
            if has_test:
                t, ts = mean_sd([r["heldout_elbo"] for r in sel if "heldout_elbo" in r])
                steps = sel[0].get("heldout_embed_steps", 0)
                cells.append(f"{t:+.3f} ± {ts:.3f}")
            cells += [gtxt, f"{floor:.3f}", f"{sel[0]['n_params_variational_q_X']:,}"]
            if has_test:
                cells.append("1 fwd pass" if steps == 0 else f"{steps} steps")
            cells.append(f"{w:.0f}")
            lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    open(os.path.join(outdir, "table.md"), "w").write(table + "\n")
    print(table + "\n")


# ------------------------------------------------------ main gap figure -----
def plot_gap_vs_depth(rows, depths, outdir):
    gm, gs, floors, cm, cs = [], [], [], [], []
    for d in depths:
        sel = [r for r in rows if r["depth"] == d and "gap" in r]
        m, s = mean_sd([r["gap"] for r in sel])
        gm.append(m); gs.append(s)
        floors.append(max([r.get("gap_noise_sd", 0.0) for r in sel], default=0.0))
        cvals = [r["control_gap"] for r in rows if r["depth"] == d and "control_gap" in r]
        c, csd = mean_sd(cvals) if cvals else (None, None)
        cm.append(c); cs.append(csd)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    # The honest part of this plot: anything inside the band is indistinguishable
    # from Monte Carlo noise in the sampled ELBO (docs/decisions.md D-01).
    if any(f > 0 for f in floors):
        ax.fill_between(depths, [-f for f in floors], floors, color=GREY, alpha=0.28,
                        lw=0, label="MC noise floor (paired, ±1 sd)")
    # The free-form control: the same refinement run on a q that CANNOT have an
    # amortisation gap. Everything below this line is optimisation slack, not
    # amortisation, so it is the floor that actually matters (D-08).
    if all(c is not None for c in cm):
        ax.errorbar(depths, cm, yerr=cs, marker="s", capsize=4, lw=1.8, ls="--",
                    color=FREE, label="free-form control (optimisation slack)")
        ax.fill_between(depths, 0, cm, color=FREE, alpha=0.12, lw=0)
    ax.errorbar(depths, gm, yerr=gs, marker="o", capsize=4, lw=2, color=AMORT,
                label="amortisation gap (±1 sd over seeds)")
    ax.axhline(0, color=GREY, lw=1, ls="--")
    unconv = [r for r in rows if "gap" in r and not r.get("refine_converged", True)]
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Amortisation gap (nats / point)")
    ax.set_title("Does the shortcut cost more as depth grows?")
    if unconv:
        ax.set_title(ax.get_title() + "\n(open refinements: gaps are lower bounds)",
                     fontsize=9)
    ax.set_xticks(depths)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, "gap_vs_depth.png")


def plot_gap_convergence(rows, outdir):
    """Gap vs refinement steps. The gap only means something where this is flat."""
    sel = [r for r in rows if r.get("gap_trace")] + \
          [dict(r, gap_trace=r["control_gap_trace"],
                gap_trace_steps=r["control_gap_trace_steps"],
                refine_converged=r.get("control_refine_converged", True))
           for r in rows if r.get("control_gap_trace")]
    if not sel:
        print("skipped gap_convergence.png (no gap_trace in results)")
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for r in sel:
        ok = r.get("refine_converged", True)
        ax.plot(r["gap_trace_steps"], r["gap_trace"], marker="o", ms=3,
                lw=1.8, ls="-" if ok else "--", alpha=0.85,
                label=f"depth {r['depth']} seed {r['seed']} {r['scheme']}"
                      + ("" if ok else " (open)"))
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.set_xlabel("refinement steps (generative model frozen)")
    ax.set_ylabel("measured gap (nats / point)")
    ax.set_title("Is the gap converged, or still climbing?")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, "gap_convergence.png")


# ------------------------------------------------------------- curves -------
def plot_elbo_curves(rows, depths, outdir):
    fig, axes = plt.subplots(1, len(depths), figsize=(4.4 * len(depths), 3.4), squeeze=False)
    seed0 = min(r["seed"] for r in rows)
    for i, d in enumerate(depths):
        ax = axes[0][i]
        for scheme, c in [("free_form", FREE), ("amortised", AMORT)]:
            for r in [r for r in rows if r["depth"] == d and r["scheme"] == scheme]:
                ax.plot(r["history"]["step"], r["history"]["elbo"], color=c, alpha=0.75,
                        label=scheme if r["seed"] == seed0 else None)
        ax.set_title(f"depth {d}"); ax.set_xlabel("step"); ax.set_ylabel("ELBO / point")
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, fontsize=8)
    save(fig, outdir, "elbo_curves.png")


def plot_ard(rows, outdir):
    seed0 = min(r["seed"] for r in rows)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    plotted = False
    for r in rows:
        if r["seed"] != seed0 or r["scheme"] != "free_form":
            continue
        rel = input_relevance(r)
        if rel:
            ax.plot(sorted(rel, reverse=True), marker="o", ms=4,
                    label=f"depth {r['depth']}, layer 0 inputs")
            plotted = True
    if not plotted:
        print("skipped ard_weights.png (no ARD data)")
        plt.close(fig); return
    ax.set_xlabel("latent dimension (sorted by relevance)")
    ax.set_ylabel("ARD relevance (1 / lengthscale)")
    ax.set_title("Which latent dimensions survived")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, outdir, "ard_weights.png")


# ------------------------------------------------------- latent scatter -----
def plot_latent_space(rows, depths, outdir):
    sel = [r for r in rows if r.get("latent_mu")]
    if not sel:
        print("skipped latent_space.png (no latent_mu in results)")
        return
    schemes = ["free_form", "amortised"]
    fig, axes = plt.subplots(len(schemes), len(depths),
                             figsize=(3.5 * len(depths), 3.3 * len(schemes)), squeeze=False)
    for i, scheme in enumerate(schemes):
        for j, d in enumerate(depths):
            ax = axes[i][j]
            match = [r for r in sel if r["scheme"] == scheme and r["depth"] == d]
            if not match:
                ax.axis("off"); continue
            r = match[0]
            mu = r["latent_mu"]
            # plot the two dimensions the model actually kept, by ARD relevance,
            # falling back to the two with the widest posterior means
            rel = input_relevance(r) or r.get("mu_std_per_dim")
            a, b = sorted(range(len(rel)), key=lambda k: -rel[k])[:2]
            c = r.get("latent_colour")
            sc = ax.scatter([m[a] for m in mu], [m[b] for m in mu], c=c, s=9,
                            cmap="viridis", alpha=0.85, linewidths=0)
            ax.set_title(f"{scheme}, depth {d}\n(sigma_mean={r['q_sigma_mean']:.3f})",
                         fontsize=9)
            ax.set_xlabel(f"latent dim {a}"); ax.set_ylabel(f"latent dim {b}")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("q(X) means, coloured by the leading principal component of Y",
                 fontsize=10)
    save(fig, outdir, "latent_space.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    rows = json.load(open(a.results))["rows"]
    os.makedirs(a.outdir, exist_ok=True)
    depths = sorted({r["depth"] for r in rows})

    write_table(rows, depths, a.outdir)
    plot_gap_vs_depth(rows, depths, a.outdir)
    plot_gap_convergence(rows, a.outdir)
    plot_elbo_curves(rows, depths, a.outdir)
    plot_ard(rows, a.outdir)
    plot_latent_space(rows, depths, a.outdir)

    # collapse check, printed rather than plotted -- it is a sentence on a slide
    print("\n=== q(X) posterior spread (collapse check) ===")
    for r in sorted(rows, key=lambda r: (r["depth"], r["scheme"], r["seed"])):
        s = r.get("q_sigma_mean")
        if s is None:
            continue
        flag = ""
        if s < 0.05:
            flag = "  <- COLLAPSED to a point estimate"
        elif s > 0.95:
            flag = "  <- COLLAPSED to the prior (latent carries nothing)"
        print(f"  depth {r['depth']} {r['scheme']:10s} seed {r['seed']}: "
              f"mean sigma = {s:.4f}{flag}")


if __name__ == "__main__":
    main()
