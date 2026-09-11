#!/usr/bin/env python3
"""Why is the free-form optimisation slack so large?

Finding, from slack_study.py's term decomposition: the refinement on the
free-form arm does NOT improve the fit -- it makes it slightly worse -- while
cutting KL(q(X)||p(X)) by several nats. So the slack is a KL effect, i.e. q(X)
starts far from the prior and joint training pays that debt down slowly.

Two sources of initial KL debt, measured on the synthetic set (N=320, Q=5):

  sigma_0 = softplus(-2) = 0.127 against a N(0,I) prior
      -log(0.127) + 0.5*(0.127^2 - 1) = 1.572 nats per dimension
      = 7.86 nats/point, paid by BOTH arms.

  raw PCA scores have per-dimension std 2.17 ... 1.12, not 1.0
      0.5 * sum(mu^2) adds a further 6.7 nats/point, paid by the FREE-FORM arm
      ONLY -- the encoder's randomly-initialised head emits near-zero means.

The second one is an experimental-validity problem, not just a slow start: the
two arms are supposed to differ ONLY in how q(X) is parameterised, and they are
starting 6.7 nats/point apart. (D-05 checked this and found them matched, but
that check used FreeFormLatent's DEFAULT randn*0.1 init -- run_experiment.py
passes pca_init, which is a different, much larger initialisation.)

This script tests whether removing each source removes the slack.

Usage:  python scripts/why_slack.py --checkpoints 1000 2000 --depth 2
"""
import argparse, json, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import pca_init, synthetic_deep_gp, train_test_split
from src.latent import AmortisedLatent, FreeFormLatent
from src.model import DeepGPLVM
from src.train import _free_form_at, _freeze_generative_copy, eval_elbo, train

HIDDEN_FOR_DEPTH = {1: (), 2: (3,), 3: (3, 3)}
EVAL_SEED = 1234


def inv_softplus(y):
    return math.log(math.expm1(y))


@torch.no_grad()
def terms(model, Y, num_samples, seed=EVAL_SEED):
    torch.manual_seed(seed)
    was = model.training; model.eval()
    t = model.elbo_terms(Y, num_data=Y.shape[0], num_samples=num_samples)
    model.train(was)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 2000, 4000])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--latent-dim", type=int, default=5)
    ap.add_argument("--num-inducing", type=int, default=50)
    ap.add_argument("--refine-steps", type=int, default=600)
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--eval-samples", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only configs whose label starts with these letters")
    ap.add_argument("--out", default="results/why_slack.json")
    a = ap.parse_args()

    Y_all, _ = synthetic_deep_gp(n=400, data_dim=20, latent_dim=2, true_depth=2, seed=0)
    Y, _ = train_test_split(Y_all, test_frac=0.2, seed=0)
    N, D, Q = *Y.shape, a.latent_dim
    X_raw = pca_init(Y, Q)
    X_unit = X_raw / X_raw.std(0)           # scores standardised to match the prior

    S_LO = -2.0                              # sigma ~ 0.127, the current default
    S_HI = inv_softplus(1.0)                 # sigma ~ 1.0, at the prior

    configs = {
        "A free-form, raw PCA, s0=0.13  (CURRENT)":
            lambda: FreeFormLatent(N, Q, X_init=X_raw, init_log_sigma=S_LO),
        "B free-form, unit-var PCA, s0=0.13":
            lambda: FreeFormLatent(N, Q, X_init=X_unit, init_log_sigma=S_LO),
        "C free-form, unit-var PCA, s0=1.0":
            lambda: FreeFormLatent(N, Q, X_init=X_unit, init_log_sigma=S_HI),
        "D amortised, s0=0.13  (REFERENCE)":
            lambda: AmortisedLatent(N, Q, D, init_log_sigma=S_LO),
        "E amortised, s0=1.0":
            lambda: AmortisedLatent(N, Q, D, init_log_sigma=S_HI),
    }

    if a.only:
        configs = {k: v for k, v in configs.items() if k[0] in a.only}
    rows = []
    for name, make in configs.items():
        for seed in a.seeds:
            torch.manual_seed(seed)
            latent = make()
            model = DeepGPLVM(latent, D, hidden_dims=HIDDEN_FOR_DEPTH[a.depth],
                              num_inducing=a.num_inducing)
            model.forward(Y, num_samples=2)
            kl0 = float(latent.kl(Y=Y)) / N
            if seed == a.seeds[0]:
                print(f"\n=== {name} ===   KL at init = {kl0:.3f} nats/point")
            opt, done = None, 0
            for ckpt in a.checkpoints:
                h = train(model, Y, steps=ckpt - done, lr=a.lr,
                          num_samples=a.num_samples, verbose=False,
                          log_every=10 ** 9, optimizer=opt)
                opt, done = h["optimizer"], ckpt

                t_before = terms(model, Y, a.eval_samples)
                r = _freeze_generative_copy(model)
                r.latent = _free_form_at(model.latent, Y, N, Q)
                train(r, Y, steps=a.refine_steps, lr=a.lr, num_samples=a.num_samples,
                      params=r.latent.parameters(), verbose=False,
                      log_every=10 ** 9)
                t_after = terms(r, Y, a.eval_samples)
                e0 = eval_elbo(model, Y, num_samples=a.eval_samples, seed=EVAL_SEED)
                e1 = eval_elbo(r, Y, num_samples=a.eval_samples, seed=EVAL_SEED)
                rows.append({"config": name, "seed": seed, "steps": ckpt,
                             "kl_init": kl0, "elbo": e0, "slack": e1 - e0,
                             "kl_x": t_before["kl_x"],
                             "d_ell": t_after["ell"] - t_before["ell"],
                             "d_kl_x": t_after["kl_x"] - t_before["kl_x"]})
                if seed == a.seeds[0]:
                    print(f"  {ckpt:5d} steps: ELBO {e0:+9.3f} | slack {e1 - e0:+.4f} "
                          f"| KL(q(X)) now {t_before['kl_x']:6.3f} "
                          f"| refinement: dELL {rows[-1]['d_ell']:+.3f} "
                          f"dKL {rows[-1]['d_kl_x']:+.3f}")
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump({"args": vars(a), "rows": rows}, open(a.out, "w"), indent=2)

    print("\n" + "=" * 78)
    print("SLACK BY CONFIGURATION (mean over seeds)")
    print("=" * 78)
    hdr = "  " + " " * 40 + "".join(f"{c:>10d}" for c in a.checkpoints)
    print(hdr.replace(str(a.checkpoints[0]).rjust(10), f"{'@' + str(a.checkpoints[0]):>10}", 1))
    for name in configs:
        cells = []
        for ckpt in a.checkpoints:
            v = [r["slack"] for r in rows if r["config"] == name and r["steps"] == ckpt]
            cells.append(f"{sum(v) / len(v):+10.4f}" if v else f"{'-':>10}")
        kl0 = next(r["kl_init"] for r in rows if r["config"] == name)
        print(f"  {name:38s}" + "".join(cells) + f"   (KL@init {kl0:.1f})")


if __name__ == "__main__":
    main()
