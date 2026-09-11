#!/usr/bin/env python3
"""Where does the free-form control (optimisation slack) drop away?

The negative control of docs/decisions.md D-08 was +0.062 nats/point at 1,200
steps and +2.95 at 300 -- large enough to swamp the amortisation gap we are
trying to measure. This script finds the training budget at which it stops
mattering, and records enough to say WHY it is large in the first place.

Method: ONE training trajectory per (depth, scheme, seed), observed at
checkpoints. Adam's state carries across checkpoints, so this is a single run
watched over time, not a set of independent restarts -- restarts would confound
budget with optimiser re-initialisation.

At every checkpoint, with the generative model frozen and a fresh free-form q
attached at the current q's own values:
  * gap / control      what the refinement recovers, in nats/point
  * term decomposition whether it recovers FIT (q's means were wrong) or
                       KL (q's spread was wrong) -- these have different causes
  * decoder drift      how far the generative parameters moved since the last
                       checkpoint. If q is "chasing a moving target", slack
                       should track this.

Usage:
    python scripts/slack_study.py --checkpoints 500 1000 2000 4000 8000 --seeds 0 1 2
"""
import argparse, json, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import pca_init, synthetic_deep_gp, train_test_split
from src.latent import AmortisedLatent, FreeFormLatent
from src.model import DeepGPLVM
from src.train import (_free_form_at, _freeze_generative_copy, eval_elbo,
                       make_optimizer, measure_amortisation_gap, train)

HIDDEN_FOR_DEPTH = {1: (), 2: (3,), 3: (3, 3)}
EVAL_SEED = 1234


def gen_params(model):
    """Generative parameters only -- everything that is frozen during refinement."""
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if not n.startswith("latent.")}


def drift(prev, now):
    """Relative L2 movement of the generative parameters between checkpoints."""
    num = math.sqrt(sum(float(((now[k] - prev[k]) ** 2).sum()) for k in prev))
    den = math.sqrt(sum(float((prev[k] ** 2).sum()) for k in prev)) + 1e-12
    return num / den


@torch.no_grad()
def terms(model, Y, num_samples, seed=EVAL_SEED):
    torch.manual_seed(seed)
    was = model.training
    model.eval()
    t = model.elbo_terms(Y, num_data=Y.shape[0], num_samples=num_samples)
    model.train(was)
    return t


def refined_copy(model, Y, refine_steps, lr, num_samples):
    """The refinement of the gap protocol, returned so its terms can be read."""
    r = _freeze_generative_copy(model)
    r.latent = _free_form_at(model.latent, Y, Y.shape[0], model.latent.latent_dim)
    train(r, Y, steps=refine_steps, lr=lr, num_samples=num_samples,
          params=r.latent.parameters(), verbose=False, log_every=refine_steps)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[500, 1000, 2000, 4000, 8000])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--latent-dim", type=int, default=5)
    ap.add_argument("--num-inducing", type=int, default=50)
    ap.add_argument("--refine-steps", type=int, default=600)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-encoder", type=float, default=0.001,
                    help="separate rate for the amortised encoder (D-12)")
    ap.add_argument("--out", default="results/slack_study.json")
    a = ap.parse_args()

    Y_all, _ = synthetic_deep_gp(n=400, data_dim=20, latent_dim=2, true_depth=2, seed=0)
    Y, _ = train_test_split(Y_all, test_frac=0.2, seed=0)
    N, D = Y.shape
    X_init = pca_init(Y, a.latent_dim)
    print(f"train={tuple(Y.shape)}  checkpoints={a.checkpoints}")

    # Resume at cell granularity. Pointing --out at an existing file EXTENDS it:
    # completed (depth, seed, scheme) cells are skipped, so adding seeds is just a
    # re-run with a longer --seeds list. Guarded on the settings that would make
    # old and new rows incomparable.
    rows = []
    if os.path.exists(a.out):
        prev = json.load(open(a.out))
        rows = prev["rows"]
        pa = prev.get("args", {})
        clash = [k for k in ("lr", "lr_encoder", "latent_dim", "num_inducing",
                             "refine_steps", "num_samples", "eval_samples")
                 if k in pa and pa[k] != getattr(a, k)]
        if clash:
            raise SystemExit(
                f"REFUSING to extend {a.out}: it was produced with different "
                f"{clash}. Those rows are not comparable with new ones. Write to a "
                f"new --out, or pass the original settings.")
        print(f"[resume] {len(rows)} rows present")

    # A cell is one row PER CHECKPOINT, so "has some rows" is not "is done". A cell
    # interrupted mid-trajectory must be redone from scratch: this script keeps no
    # model checkpoint, and the budget sweep is meaningless unless every point comes
    # from one continuous Adam trajectory. So partial cells are dropped and re-run.
    want = set(a.checkpoints)
    have = {}
    for r in rows:
        have.setdefault((r["depth"], r["seed"], r["scheme"]), set()).add(r["steps"])
    completed = {k for k, v in have.items() if want <= v}
    partial = {k for k in have if k not in completed}
    if partial:
        print(f"[resume] discarding {len(partial)} partial cell(s), will re-run: "
              + ", ".join(f"d{d}s{s}/{sc}" for d, s, sc in sorted(partial)))
        rows = [r for r in rows
                if (r["depth"], r["seed"], r["scheme"]) not in partial]
    print(f"[resume] {len(completed)} cells complete, {len(rows)} rows kept")

    for depth in a.depths:
        for seed in a.seeds:
            for scheme in ["free_form", "amortised"]:
                if (depth, seed, scheme) in completed:
                    print(f"=== depth {depth} | seed {seed} | {scheme} === SKIPPED (done)")
                    continue
                print(f"\n=== depth {depth} | seed {seed} | {scheme} ===")
                torch.manual_seed(seed)
                latent = (FreeFormLatent(N, a.latent_dim, X_init=X_init)
                          if scheme == "free_form" else AmortisedLatent(N, a.latent_dim, D))
                model = DeepGPLVM(latent, D, hidden_dims=HIDDEN_FOR_DEPTH[depth],
                                  num_inducing=a.num_inducing)
                model.forward(Y, num_samples=2)      # lazy q(U) init before snapshots
                opt = make_optimizer(model, a.lr, a.lr_encoder)
                done, prev = 0, gen_params(model)

                for ckpt in a.checkpoints:
                    h = train(model, Y, steps=ckpt - done, lr=a.lr,
                              num_samples=a.num_samples, verbose=False,
                              log_every=10 ** 9, optimizer=opt)
                    opt = h["optimizer"]
                    now = gen_params(model)
                    d = drift(prev, now)
                    prev, done = now, ckpt

                    t_before = terms(model, Y, a.eval_samples)
                    r = refined_copy(model, Y, a.refine_steps, a.lr, a.num_samples)
                    t_after = terms(r, Y, a.eval_samples)
                    e_before = eval_elbo(model, Y, num_samples=a.eval_samples, seed=EVAL_SEED)
                    e_after = eval_elbo(r, Y, num_samples=a.eval_samples, seed=EVAL_SEED)
                    gap = e_after - e_before

                    rows.append({
                        "depth": depth, "seed": seed, "scheme": scheme, "steps": ckpt,
                        "elbo": e_before, "gap": gap,
                        "d_ell": t_after["ell"] - t_before["ell"],
                        "d_kl_x": t_after["kl_x"] - t_before["kl_x"],
                        "d_kl_u": t_after["kl_u"] - t_before["kl_u"],
                        "decoder_drift": d, "drift_per_step": d / max(1, ckpt - (
                            a.checkpoints[a.checkpoints.index(ckpt) - 1]
                            if a.checkpoints.index(ckpt) else 0)),
                        "terms": t_before,
                    })
                    label = "control" if scheme == "free_form" else "gap    "
                    print(f"  {ckpt:6d} steps: ELBO {e_before:+9.4f} | {label} "
                          f"{gap:+.4f} | dELL {rows[-1]['d_ell']:+.4f} "
                          f"dKLx {rows[-1]['d_kl_x']:+.4f} | drift {d:.4f}")
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    with open(a.out + ".tmp", "w") as f:   # write-then-rename
                        json.dump({"args": vars(a), "rows": rows}, f, indent=2)
                    os.replace(a.out + ".tmp", a.out)

    print("\n=== control vs gap by budget (mean over seeds) ===")
    for depth in a.depths:
        print(f"  depth {depth}:")
        for ckpt in a.checkpoints:
            c = [r["gap"] for r in rows if r["depth"] == depth and r["steps"] == ckpt
                 and r["scheme"] == "free_form"]
            g = [r["gap"] for r in rows if r["depth"] == depth and r["steps"] == ckpt
                 and r["scheme"] == "amortised"]
            if c and g:
                cm, gm = sum(c) / len(c), sum(g) / len(g)
                print(f"    {ckpt:6d}: control {cm:+.4f}  gap {gm:+.4f}  "
                      f"net {gm - cm:+.4f}")


if __name__ == "__main__":
    main()
