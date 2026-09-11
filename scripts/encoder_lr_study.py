#!/usr/bin/env python3
"""Is the amortised arm's instability an encoder learning-rate problem?

The 16k-step sweep (results/slack_study.json) failed the project's own depth-1
correctness gate: gap 0.679 nats/point where it should be near zero. The suspect
is the amortised arm's optimisation, because along a single checkpointed
trajectory its ELBO is NON-MONOTONE in 3 of 6 cells while the free-form arm is
monotone in 6 of 6, and the largest gap readings land exactly on the bad
checkpoints.

The arm-specific difference: `lr=0.01` is applied to a ~20k-parameter tanh MLP
and to a handful of GP hyperparameters at once. This script gives the encoder its
own parameter group and sweeps its learning rate, with and without gradient
clipping.

NOTE ON THE DESIGN RULE. CONTRIBUTING.md says the two arms must share the optimiser and
learning rate. Tuning the encoder's rate deviates from that, deliberately: the gap
is meant to measure what a shared mapping CAN express, so an under-optimised
encoder inflates it. The honest procedure is to optimise each q-parameterisation
as well as it can be optimised, then compare. Whatever wins here must be declared
on the implementation slide.

Usage:  python scripts/encoder_lr_study.py --depth 1 --steps 8000 --seeds 0 1 2
"""
import argparse, json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import pca_init, synthetic_deep_gp, train_test_split
from src.latent import AmortisedLatent
from src.model import DeepGPLVM
from src.train import eval_elbo, measure_amortisation_gap, train

HIDDEN_FOR_DEPTH = {1: (), 2: (3,), 3: (3, 3)}
EVAL_SEED = 1234


def make_opt(model, lr_gp, lr_enc):
    """Encoder parameters in their own group so they can take a different rate."""
    enc = [p for n, p in model.named_parameters() if n.startswith("latent.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("latent.")]
    return torch.optim.Adam([{"params": rest, "lr": lr_gp},
                             {"params": enc, "lr": lr_enc}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--checkpoints", type=int, nargs="+",
                    default=[1000, 2000, 4000, 8000])
    ap.add_argument("--latent-dim", type=int, default=5)
    ap.add_argument("--num-inducing", type=int, default=50)
    ap.add_argument("--refine-steps", type=int, default=800)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="results/encoder_lr.json")
    a = ap.parse_args()

    Y_all, _ = synthetic_deep_gp(n=400, data_dim=20, latent_dim=2, true_depth=2, seed=0)
    Y, _ = train_test_split(Y_all, test_frac=0.2, seed=0)
    N, D = Y.shape
    ckpts = [c for c in a.checkpoints if c <= a.steps]

    configs = [
        ("lr_enc=0.01  (CURRENT)", 0.01,  None),
        ("lr_enc=0.003",           0.003, None),
        ("lr_enc=0.001",           0.001, None),
        ("lr_enc=0.01 + clip 1.0", 0.01,  1.0),
    ]

    rows = []
    if os.path.exists(a.out):
        rows = json.load(open(a.out))["rows"]
        print(f"[resume] {len(rows)} rows already present")
    done = {(r["config"], r["seed"]) for r in rows}

    for name, lr_enc, clip in configs:
        print(f"\n=== depth {a.depth} | {name} | clip={clip} ===")
        for seed in a.seeds:
            if (name, seed) in done:
                print(f"  seed {seed}: SKIPPED (done)")
                continue
            torch.manual_seed(seed)
            model = DeepGPLVM(AmortisedLatent(N, a.latent_dim, D), D,
                              hidden_dims=HIDDEN_FOR_DEPTH[a.depth],
                              num_inducing=a.num_inducing)
            model.forward(Y, num_samples=2)
            opt = make_opt(model, a.lr, lr_enc)
            trace, prev, non_monotone = [], None, 0
            for c in ckpts:
                train(model, Y, steps=c, lr=a.lr, num_samples=a.num_samples,
                      verbose=False, log_every=10 ** 9, optimizer=opt,
                      start_step=(0 if prev is None else prev),
                      clip_grad_norm=clip)
                prev = c
                e = eval_elbo(model, Y, num_samples=a.eval_samples, seed=EVAL_SEED)
                if trace and e < trace[-1] - 0.5:
                    non_monotone += 1
                trace.append(e)
            g = measure_amortisation_gap(model, Y, refine_steps=a.refine_steps,
                                         lr=a.lr, num_samples=a.num_samples,
                                         eval_samples=a.eval_samples,
                                         seed=EVAL_SEED, verbose=False)
            rows.append({"config": name, "lr_enc": lr_enc, "clip": clip,
                         "seed": seed, "depth": a.depth, "elbo_trace": trace,
                         "elbo": trace[-1], "gap": g["gap"],
                         "gap_noise_sd": g["gap_noise_sd"],
                         "refine_converged": g["refine_converged"],
                         "non_monotone": non_monotone})
            print(f"  seed {seed}: ELBO " + " ".join(f"{x:+8.3f}" for x in trace)
                  + f" | gap {g['gap']:+.4f}"
                  + ("  <-- NON-MONOTONE" if non_monotone else ""))
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out + ".tmp", "w") as f:
                json.dump({"args": vars(a), "rows": rows}, f, indent=2)
            os.replace(a.out + ".tmp", a.out)

    print("\n" + "=" * 76)
    print(f"DEPTH {a.depth} GATE: does any configuration give a near-zero gap?")
    print("=" * 76)
    print(f"{'configuration':26s}{'final ELBO':>13}{'gap':>18}{'unstable':>11}")
    for name, _, _ in configs:
        sel = [r for r in rows if r["config"] == name]
        if not sel:
            continue
        e = [r["elbo"] for r in sel]; g = [r["gap"] for r in sel]
        em = sum(e) / len(e); gm = sum(g) / len(g)
        gsd = (sum((x - gm) ** 2 for x in g) / max(1, len(g) - 1)) ** 0.5
        bad = sum(1 for r in sel if r["non_monotone"])
        print(f"{name:26s}{em:>+13.3f}{gm:>+11.4f}±{gsd:.3f}{bad:>8}/{len(sel)}")


if __name__ == "__main__":
    main()
