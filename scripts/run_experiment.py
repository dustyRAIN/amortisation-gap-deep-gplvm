#!/usr/bin/env python3
"""Run the depth x inference-scheme grid and write results/results.json.

Usage:
    python scripts/run_experiment.py --dataset synthetic --depths 1 2 --seeds 0 1 2
    python scripts/run_experiment.py --dataset frey --depths 1 2 3 --seeds 0 1 2 --steps 4000

Each cell trains both schemes on identical data with the same seed, then measures
the amortisation gap for the amortised one. The comparison is PAIRED: the same
seed drives initialisation and the ELBO evaluation draws for both schemes, so
Monte Carlo noise largely cancels in the difference.

Every cell also records, because a gap number alone cannot be defended:
  * the Monte Carlo noise floor of the ELBO estimator (elbo_noise_sd) and of the
    paired gap difference (gap_noise_sd) -- the band the main figure shades;
  * whether the gap refinement plateaued (refine_converged, gap_trace);
  * a held-out predictive bound, scored the way each scheme would actually be
    deployed -- encoder forward pass vs. re-optimising q for the test points;
  * q(X) means and spreads, for the latent scatter and the collapse check.

Start with --dataset synthetic --steps 800 to check the whole thing runs in a
couple of minutes before committing to a long Frey Faces run.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import frey_faces, pca_init, synthetic_deep_gp, train_test_split
from src.latent import (AmortisedLatent, DenseAmortisedLatent,
                        DenseFreeFormLatent, FreeFormLatent, is_amortised)
from src.model import DeepGPLVM
from src.train import (elbo_noise_floor, eval_elbo, heldout_elbo, latent_summary,
                       make_optimizer, measure_amortisation_gap, train)

HIDDEN_FOR_DEPTH = {1: (), 2: (3,), 3: (3, 3)}


# -- resume ------------------------------------------------------------------
# A long run must survive losing power. Two levels:
#   cell level   completed cells are already in results.json; on restart we skip
#                any (depth, seed, scheme) that is present.
#   within cell  training checkpoints model + optimiser + RNG state every
#                --ckpt-every steps, so a cell resumes mid-training rather than
#                restarting. RNG state is saved too: without it a resumed run
#                draws a different sample stream, which would quietly break the
#                paired common-random-numbers discipline this project relies on.
# Reported numbers are unaffected either way -- eval_elbo fixes its own seed.

def ckpt_path(outdir, dataset, depth, seed, scheme):
    return os.path.join(outdir, f"ckpt_{dataset}_d{depth}_s{seed}_{scheme}.pt")


def save_ckpt(path, model, step, opt, hist):
    tmp = path + ".tmp"          # write-then-rename: a power cut cannot leave a
    torch.save({                 # half-written checkpoint behind
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "hist": {"step": hist["step"], "elbo": hist["elbo"],
                 "wall_clock_s": hist.get("wall_clock_s", 0.0)},
        "rng": torch.get_rng_state(),
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(path, model, lr, lr_encoder=None):
    """Restore a cell mid-training. Returns (start_step, optimizer, hist).

    The optimiser is rebuilt with make_optimizer so the parameter-group structure
    matches what was saved -- a plain Adam here would fail to load the state of a
    two-group optimiser.
    """
    if not os.path.exists(path):
        return 0, None, None
    try:
        c = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"  [resume] checkpoint {path} unreadable ({e}); starting cell over")
        return 0, None, None
    model.load_state_dict(c["model"])
    opt = make_optimizer(model, lr, lr_encoder)
    opt.load_state_dict(c["opt"])
    torch.set_rng_state(c["rng"])
    print(f"  [resume] restored from step {c['step']}")
    return c["step"], opt, c["hist"]

EVAL_SEED = 1234          # the reported-number seed; identical for both schemes
NOISE_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)   # re-evaluations used for the noise floor


SCHEMES = {
    # name              class                   per-point?  family
    "free_form":        (FreeFormLatent,        True,  "diagonal"),
    "amortised":        (AmortisedLatent,       False, "diagonal"),
    "free_form_dense":  (DenseFreeFormLatent,   True,  "dense"),
    "amortised_dense":  (DenseAmortisedLatent,  False, "dense"),
}


def build(scheme, N, D, Q, depth, X_init, num_inducing):
    """Run all four to decompose the inference gap (see src/latent.py):

        amortised       -> free_form         amortisation gap, diagonal family
        free_form       -> free_form_dense   cost of the diagonal assumption
        amortised_dense -> free_form_dense   amortisation gap, dense family

    Lalchand et al. compare amortised_dense against free_form -- across BOTH axes
    at once -- which is why their encoder appears to beat the per-point scheme.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"{scheme!r}; choose from {list(SCHEMES)}")
    cls, per_point, _ = SCHEMES[scheme]
    latent = cls(N, Q, X_init=X_init) if per_point else cls(N, Q, D)
    return DeepGPLVM(latent, D, hidden_dims=HIDDEN_FOR_DEPTH[depth], num_inducing=num_inducing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "frey"], default="synthetic")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--schemes", nargs="+", default=["free_form", "amortised"],
                    choices=list(SCHEMES),
                    help="add free_form_dense and amortised_dense to decompose the "
                         "gap into amortisation vs the diagonal assumption")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--latent-dim", type=int, default=5)
    ap.add_argument("--num-inducing", type=int, default=50)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--refine-steps", type=int, default=2000)
    ap.add_argument("--test-steps", type=int, default=500,
                    help="free-form test-point embedding steps (encoder needs none)")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-encoder", type=float, default=0.001,
                    help="separate rate for the amortised encoder (D-12). The "
                         "shared 0.01 destabilises it and fails the depth-1 gate. "
                         "Ignored by the free-form arm. Tune on ELBO, not on gap")
    ap.add_argument("--n-subset", type=int, default=None,
                    help="frey: keep this many frames (cost scales with N)")
    ap.add_argument("--reduce-dim", type=int, default=None,
                    help="frey: PCA-project pixels to this many dims before "
                         "modelling. Biggest cost lever; declare it on the slide")
    ap.add_argument("--no-heldout", action="store_true", help="skip the test-set arm")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the free-form negative control (not recommended)")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="save a resumable training checkpoint this often (0 = off)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing results and checkpoints, start clean")
    args = ap.parse_args()

    if args.dataset == "synthetic":
        Y_all, _ = synthetic_deep_gp(n=400, data_dim=20, latent_dim=2,
                                     true_depth=2, seed=0)
    else:
        Y_all = frey_faces(n_subset=args.n_subset, reduce_dim=args.reduce_dim)
    Y, Y_test = train_test_split(Y_all, test_frac=0.2, seed=0)
    N, D = Y.shape
    print(f"dataset={args.dataset}  train={tuple(Y.shape)}  test={tuple(Y_test.shape)}")

    # One PCA basis, computed once on the training data, used to initialise the
    # free-form latents AND to place test points on the same axes.
    X_init, pca_basis = pca_init(Y, args.latent_dim, return_basis=True)
    X_test_init = pca_init(Y_test, args.latent_dim, basis=pca_basis)
    # Colour for the latent scatter: leading principal component of Y. Always
    # available (Frey has no labels) and meaningful -- it is the dominant mode of
    # variation, so a latent space that has learned anything should track it.
    colour = X_init[:, 0].tolist()

    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    rows = []
    if not args.no_resume and os.path.exists(args.out):
        try:
            rows = json.load(open(args.out))["rows"]
            done = sorted({(r["depth"], r["seed"], r["scheme"]) for r in rows})
            print(f"[resume] {len(rows)} rows already in {args.out}; "
                  f"skipping {len(done)} completed cells")
        except Exception as e:
            print(f"[resume] could not read {args.out} ({e}); starting clean")
            rows = []
    completed = {(r["depth"], r["seed"], r["scheme"]) for r in rows}

    for depth in args.depths:
        for seed in args.seeds:
            for scheme in args.schemes:
                if (depth, seed, scheme) in completed:
                    print(f"\n=== depth {depth} | seed {seed} | {scheme} === SKIPPED (done)")
                    continue
                print(f"\n=== depth {depth} | seed {seed} | {scheme} ===")
                torch.manual_seed(seed)
                model = build(scheme, N, D, args.latent_dim, depth, X_init, args.num_inducing)

                cpath = ckpt_path(outdir, args.dataset, depth, seed, scheme)
                if args.no_resume and os.path.exists(cpath):
                    os.remove(cpath)
                start, opt, prev_hist = load_ckpt(cpath, model, args.lr, args.lr_encoder)
                if opt is None:
                    opt = make_optimizer(model, args.lr, args.lr_encoder)
                if start >= args.steps:
                    print(f"  [resume] training already complete at {start} steps")
                hist = train(model, Y, steps=args.steps, lr=args.lr,
                             num_samples=args.num_samples, log_every=max(1, args.steps // 5),
                             optimizer=opt, start_step=start, hist=prev_hist,
                             ckpt_every=(args.ckpt_every or None),
                             on_checkpoint=(
                                 (lambda st, o, h: save_ckpt(cpath, model, st, o, h))
                                 if args.ckpt_every else None))

                noise = elbo_noise_floor(model, Y, num_samples=args.eval_samples,
                                         seeds=NOISE_SEEDS)
                print(f"  ELBO MC noise floor: sd = {noise['sd']:.4f} nats/point")

                row = {
                    "depth": depth, "seed": seed, "scheme": scheme,
                    "family": SCHEMES[scheme][2], "per_point": SCHEMES[scheme][1],
                    "elbo": eval_elbo(model, Y, num_samples=args.eval_samples, seed=EVAL_SEED),
                    "elbo_noise_sd": noise["sd"],
                    "wall_clock_s": hist["wall_clock_s"],
                    **{f"n_params_{k}": v for k, v in model.parameter_counts().items()},
                    "ard": {k: v.tolist() for k, v in model.ard_report().items()},
                    "history": {"step": hist["step"], "elbo": hist["elbo"]},
                }

                # q(X) diagnostics. The full mu table is only kept for the first
                # seed -- it is for one scatter plot, not for every cell.
                lat = latent_summary(model, Y, colour=colour)
                if seed != args.seeds[0]:
                    lat.pop("latent_mu", None)
                    lat.pop("latent_colour", None)
                row.update(lat)

                if not args.no_heldout:
                    row.update(heldout_elbo(
                        model, Y_test, steps=args.test_steps, lr=args.lr,
                        num_samples=args.num_samples, eval_samples=args.eval_samples,
                        seed=EVAL_SEED, X_init=X_test_init))

                # The SAME refinement is run on the free-form arm as a negative
                # control. A free-form q has no amortisation gap by construction,
                # so whatever the refinement recovers there is pure optimisation
                # slack -- the real floor the amortised gap has to clear. Without
                # it, an under-trained model reports slack as if it were a
                # finding. See docs/decisions.md D-08.
                if SCHEMES[scheme][1] and not args.no_control:
                    c = measure_amortisation_gap(
                        model, Y, refine_steps=args.refine_steps, lr=args.lr,
                        num_samples=args.num_samples, eval_samples=args.eval_samples,
                        seed=EVAL_SEED)
                    row.update({f"control_{k}": v for k, v in c.items()
                                if k != "refine_history"})

                if not SCHEMES[scheme][1]:
                    g = measure_amortisation_gap(
                        model, Y, refine_steps=args.refine_steps, lr=args.lr,
                        num_samples=args.num_samples, eval_samples=args.eval_samples,
                        seed=EVAL_SEED)
                    row.update({k: v for k, v in g.items() if k != "refine_history"})
                    row["refine_history"] = {"step": g["refine_history"]["step"],
                                             "elbo": g["refine_history"]["elbo"]}
                rows.append(row)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                with open(args.out + ".tmp", "w") as f:
                    json.dump({"args": vars(args), "rows": rows}, f, indent=2)
                os.replace(args.out + ".tmp", args.out)
                print(f"  -> wrote {len(rows)} rows to {args.out}")
                if os.path.exists(cpath):
                    os.remove(cpath)   # cell is durable in results.json now

    print("\n=== SUMMARY: amortisation gap (nats/point) ===")
    for depth in args.depths:
        sel = [r for r in rows if r["depth"] == depth and "gap" in r]
        if not sel:
            continue
        gaps = [r["gap"] for r in sel]
        m = sum(gaps) / len(gaps)
        sd = (sum((g - m) ** 2 for g in gaps) / max(1, len(gaps) - 1)) ** 0.5
        floor = max(r["gap_noise_sd"] for r in sel)
        unconverged = [r["seed"] for r in sel if not r["refine_converged"]]
        ctrl = [r["control_gap"] for r in rows
                if r["depth"] == depth and "control_gap" in r]
        cm = sum(ctrl) / len(ctrl) if ctrl else 0.0
        floor = max(floor, cm)
        verdict = "ABOVE the floor" if abs(m) > floor else "INSIDE the floor (null)"
        print(f"  depth {depth}: raw {m:+.4f} +/- {sd:.4f}  (n={len(gaps)})"
              f"  | control {cm:+.4f} | MC {max(r['gap_noise_sd'] for r in sel):.4f}"
              f"  -> corrected {m - cm:+.4f}, {verdict}")
        if unconverged:
            print(f"            refinement had not plateaued for seed(s) {unconverged}"
                  f" -- those gaps are LOWER BOUNDS")


if __name__ == "__main__":
    main()
