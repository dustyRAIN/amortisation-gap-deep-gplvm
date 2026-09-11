#!/usr/bin/env python3
"""Measure the Monte Carlo noise floor of the gap estimator, and store it.

The report argues that the MC noise floor -- the diagnostic one would normally use
to judge whether a gap is real -- would have endorsed the under-training artifact.
That argument needs a sourced number, so this measures it in the same regime as the
artifact (short budgets, synthetic data) and writes it to results/.

The relevant quantity is `paired_gap_noise_floor`: both models evaluated under the
SAME seed, the difference taken, and the spread of those differences across seeds.
That is what survives the common-random-numbers cancellation.
"""
import json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import pca_init, synthetic_deep_gp, train_test_split
from src.latent import AmortisedLatent, FreeFormLatent
from src.model import DeepGPLVM
from src.train import (_free_form_at, _freeze_generative_copy, elbo_noise_floor,
                       eval_elbo, make_optimizer, paired_gap_noise_floor, train)

HIDDEN = {1: (), 2: (3,)}
Y_all, _ = synthetic_deep_gp(n=400, data_dim=20, latent_dim=2, true_depth=2, seed=0)
Y, _ = train_test_split(Y_all, test_frac=0.2, seed=0)
N, D, Q = *Y.shape, 5
X_init = pca_init(Y, Q)

out = []
for depth in (1, 2):
    for steps in (300, 1000, 16000):
        torch.manual_seed(0)
        m = DeepGPLVM(AmortisedLatent(N, Q, D), D, hidden_dims=HIDDEN[depth], num_inducing=50)
        m.forward(Y, num_samples=2)
        train(m, Y, steps=steps, lr=0.01, num_samples=5, verbose=False, log_every=10**9,
              optimizer=make_optimizer(m, 0.01, 0.001))
        # An eval pass before copying: training leaves non-leaf tensors attached to the
        # module, which deepcopy refuses. The gap protocol never hits this because it
        # always evaluates the bound before building the refined copy.
        m.zero_grad(set_to_none=True)
        eval_elbo(m, Y, num_samples=8, seed=0)
        # the refined counterpart, exactly as the gap protocol builds it
        r = _freeze_generative_copy(m)
        r.latent = _free_form_at(m.latent, Y, N, Q)
        train(r, Y, steps=600, lr=0.01, num_samples=5, params=r.latent.parameters(),
              verbose=False, log_every=10**9)
        paired = paired_gap_noise_floor(m, r, Y, num_samples=64)
        single = elbo_noise_floor(m, Y, num_samples=64)
        row = {"depth": depth, "steps": steps,
               "paired_gap_noise_sd": paired["sd"], "single_elbo_noise_sd": single["sd"],
               "gap_at_this_budget": paired["mean"]}
        out.append(row)
        print(f"  depth {depth} @{steps:5d} steps: paired gap noise sd {paired['sd']:.5f} | "
              f"single-ELBO noise sd {single['sd']:.5f} | gap {paired['mean']:+.4f}")

os.makedirs("results", exist_ok=True)
json.dump({"rows": out}, open("results/mc_noise_floor.json", "w"), indent=2)
sds = [r["paired_gap_noise_sd"] for r in out]
print(f"\npaired gap noise floor across all conditions: {min(sds):.5f} - {max(sds):.5f}")
print("wrote results/mc_noise_floor.json")
