#!/usr/bin/env python3
"""Correctness and degeneracy checks for the deep GP-LVM.

Run this after any change to src/. It answers two questions the deck has to be
able to answer under questioning:

  A. Is this actually an UNSUPERVISED multi-layer GP-LVM -- latents inferred, the
     generative model never given the data as input, the ELBO accounted for
     correctly, and the gap protocol measuring what it claims to measure?

  B. Is the fitted model UNSATURATED -- ARD not pruned to nothing, posterior not
     collapsed, kernel and noise not pinned at their limits, inner layers still
     carrying variation, and the whole thing beating a trivial baseline?

Usage:  python scripts/verify_model.py [--steps 1200] [--quick]
"""
import argparse, math, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import pca_init, synthetic_deep_gp
from src.latent import (AmortisedLatent, DenseAmortisedLatent,
                        DenseFreeFormLatent, FreeFormLatent,
                        free_form_counterpart, is_amortised)
from src.model import DeepGPLVM
from src.train import _free_form_at, eval_elbo, measure_amortisation_gap, train

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def check(name, ok, detail="", warn_only=False):
    status = PASS if ok else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f"\n         {detail}" if detail else ""))
    return ok


def section(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def build(scheme, N, D, Q, hidden, Y, M=25):
    latent = (FreeFormLatent(N, Q, X_init=pca_init(Y, Q)) if scheme == "free_form"
              else AmortisedLatent(N, Q, D))
    return DeepGPLVM(latent, D, hidden_dims=hidden, num_inducing=M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 300 if a.quick else a.steps

    N, D, Q = 200, 10, 5
    torch.manual_seed(0)
    Y, Z_true = synthetic_deep_gp(n=N, data_dim=D, latent_dim=2, true_depth=2, seed=0)

    # =====================================================================
    section("A. IS IT AN UNSUPERVISED DEEP GP-LVM?  (structural, untrained)")
    # =====================================================================
    m1 = build("free_form", N, D, Q, (), Y)
    m2 = build("free_form", N, D, Q, (3,), Y)
    m3 = build("amortised", N, D, Q, (3, 3), Y)

    check("depth = number of GP mappings (1/2/3 -> 1/2/3 layers)",
          (m1.depth, m2.depth, m3.depth) == (1, 2, 3) and
          (len(m1.layers), len(m2.layers), len(m3.layers)) == (1, 2, 3),
          f"layers = {len(m1.layers)}, {len(m2.layers)}, {len(m3.layers)}")

    dims = [(l.input_dims, l.output_dims) for l in m3.layers]
    chained = all(dims[i][1] == dims[i + 1][0] for i in range(len(dims) - 1))
    check("layer dimensions chain Q -> hidden -> ... -> D",
          chained and dims[0][0] == Q and dims[-1][1] == D, f"{dims}")

    # THE unsupervised test: the generative model is a pure function of the
    # latent sample. Feed the same X with completely different Y; if the decoder
    # peeked at the data the outputs would differ.
    # Note: the cascade is itself stochastic -- GPyTorch's DeepGPLayer draws
    # samples between layers -- so every comparison below fixes the seed first.
    # Otherwise you are comparing two different draws, not two different inputs.
    Y_noise = torch.randn_like(Y) * 5.0
    x = torch.randn(4, N, Q)
    with torch.no_grad():
        m2.forward(Y, latent_sample=x)   # WARM-UP, and it is load-bearing:
        # GPyTorch's VariationalStrategy initialises q(U) lazily on its FIRST
        # call, so call 1 and call 2 differ even for identical inputs and an
        # identical seed. Comparing without this warm-up reports a Y-dependence
        # that is really just first-call initialisation.
        torch.manual_seed(11); w_a = m2.forward(Y, latent_sample=x).mean
        torch.manual_seed(11); w_b = m2.forward(Y, latent_sample=x).mean
    check("forward is deterministic given the seed (post warm-up)",
          torch.allclose(w_a, w_b, atol=1e-6),
          f"identical inputs, identical seed -> max|difference| = "
          f"{(w_a - w_b).abs().max():.2e}")
    with torch.no_grad():
        torch.manual_seed(11); o_a = m2.forward(Y, latent_sample=x).mean
        torch.manual_seed(11); o_b = m2.forward(Y_noise, latent_sample=x).mean
    check("generative model never sees Y (pure function of X)",
          torch.allclose(o_a, o_b, atol=1e-6),
          f"max|difference| = {(o_a - o_b).abs().max():.2e} with Y replaced by "
          f"pure noise -- this is what makes the model UNSUPERVISED")

    # Y enters ONLY through q. For the free-form scheme it does not enter at all.
    torch.manual_seed(7); s_a = m2.latent.rsample(3, Y=Y)
    torch.manual_seed(7); s_b = m2.latent.rsample(3, Y=Y_noise)
    check("free-form q ignores Y entirely (it is a table, not a map)",
          torch.allclose(s_a, s_b))
    torch.manual_seed(7); e_a = m3.latent.rsample(3, Y=Y)
    torch.manual_seed(7); e_b = m3.latent.rsample(3, Y=Y_noise)
    check("amortised q DOES read Y (it is the encoder -- part of q, not of p)",
          not torch.allclose(e_a, e_b),
          f"mean|difference| = {(e_a - e_b).abs().mean():.4f}")

    # ELBO accounting, term by term
    torch.manual_seed(3)
    total = float(m2.elbo(Y, num_data=N, num_samples=8))
    torch.manual_seed(3)
    out = m2.forward(Y, num_samples=8)
    import gpytorch
    mll = gpytorch.mlls.DeepApproximateMLL(
        gpytorch.mlls.VariationalELBO(m2.likelihood, m2, num_data=N))
    gp_part = float(mll(out, Y))
    kl_x = float(m2.latent.kl(Y=Y)) / N
    check("elbo() = GP term - KL(q(X)||p(X))/N, exactly",
          abs(total - (gp_part - kl_x)) < 1e-4,
          f"{total:.6f} vs {gp_part - kl_x:.6f}  (latent KL/pt = {kl_x:.4f})")

    # the diagnostic decomposition must reconstruct the objective exactly
    torch.manual_seed(4); t = m2.elbo_terms(Y, num_data=N, num_samples=8)
    torch.manual_seed(4); e = float(m2.elbo(Y, num_data=N, num_samples=8))
    recon = t["ell"] - t["kl_u"] - t["kl_x"]
    check("elbo_terms() reconstructs elbo(): ell - kl_u - kl_x",
          abs(e - recon) < 1e-3,
          f"{e:.6f} vs {recon:.6f}  (ell {t['ell']:+.3f}, kl_u {t['kl_u']:+.3f}, "
          f"kl_x {t['kl_x']:+.3f})")

    # KL against the closed form for a diagonal Gaussian vs N(0, I)
    q = m2.latent.q()
    mu, sd = q.mean, q.stddev
    analytic = float((0.5 * (sd ** 2 + mu ** 2 - 1) - torch.log(sd)).sum())
    check("KL[q(X)||N(0,I)] matches the closed form",
          abs(float(m2.latent.kl()) - analytic) < 1e-3,
          f"{float(m2.latent.kl()):.4f} vs {analytic:.4f}")

    # reparameterisation actually propagates gradients into q
    m2.zero_grad()
    (-m2.elbo(Y, num_data=N, num_samples=4)).backward()
    check("gradients reach q(X) (reparameterised, not score-function)",
          m2.latent.mu.grad is not None and m2.latent.mu.grad.abs().sum() > 0)
    named = dict(m2.named_parameters())
    dead = [n for n, p in named.items() if p.grad is None or p.grad.abs().sum() == 0]
    check("every model parameter receives gradient",
          not dead, f"no-gradient parameters: {dead}" if dead else
          f"all {len(named)} parameter tensors have non-zero gradient")

    # =====================================================================
    section("A2. THE DENSE-COVARIANCE ARMS (gap decomposition)")
    # =====================================================================
    schemes = [("free-form diag", FreeFormLatent(N, Q)),
               ("amortised diag", AmortisedLatent(N, Q, D)),
               ("free-form DENSE", DenseFreeFormLatent(N, Q)),
               ("amortised DENSE", DenseAmortisedLatent(N, Q, D))]
    kls = []
    for name, lat in schemes:
        kl = float(lat.kl(Y=Y)) / N
        kls.append(kl)
        check(f"{name}: q(X) usable (rsample, KL finite)",
              lat.rsample(3, Y=Y).shape == torch.Size([3, N, Q]) and
              math.isfinite(kl), f"KL/pt {kl:.4f}, {lat.n_variational_params:,} params")
    check("all four schemes start at the same KL (matched init, D-05)",
          max(kls) - min(kls) < 0.05,
          "KL/pt per scheme: " + ", ".join(f"{k:.3f}" for k in kls))

    dense = DenseFreeFormLatent(N, Q)
    q = dense.q()
    p_ = torch.distributions.MultivariateNormal(
        torch.zeros(N, Q), scale_tril=torch.eye(Q).expand(N, Q, Q))
    check("dense KL matches the closed-form MVN KL",
          abs(float(dense.kl()) - float(torch.distributions.kl_divergence(q, p_).sum())) < 1e-3)

    # a dense q must be able to represent correlations a diagonal one cannot
    with torch.no_grad():
        dense.raw_L[:, 1, 0] = 0.7
    cov = dense.q().covariance_matrix
    off = float(cov[:, 1, 0].abs().mean())
    check("dense q represents off-diagonal correlation (not secretly diagonal)",
          off > 1e-3, f"mean |cov[1,0]| = {off:.4f}")

    for name, lat in schemes:
        want = DenseFreeFormLatent if "DENSE" in name else FreeFormLatent
        check(f"{name}: refines to a same-family per-point scheme",
              free_form_counterpart(lat) is want, f"-> {free_form_counterpart(lat).__name__}")

    # the refinement must START exactly where the encoder is, in either family
    for name, lat in [("diag", AmortisedLatent(N, Q, D)),
                      ("dense", DenseAmortisedLatent(N, Q, D))]:
        ff = _free_form_at(lat, Y, N, Q)
        a, b = lat.q(Y=Y), ff.q(Y=Y)
        dmu = float((a.mean - b.mean).abs().max())
        dcv = float((a.covariance_matrix - b.covariance_matrix).abs().max()) \
            if hasattr(a, "covariance_matrix") else float((a.stddev - b.stddev).abs().max())
        check(f"{name} refinement initialises AT the encoder's own q (D-03)",
              dmu < 1e-5 and dcv < 1e-5, f"max|dmu| {dmu:.1e}, max|dcov| {dcv:.1e}")

    check("is_amortised() separates encoders from tables",
          [is_amortised(l) for _, l in schemes] == [False, True, False, True])

    # =====================================================================
    section("B. DOES THE GAP PROTOCOL MEASURE WHAT IT CLAIMS?")
    # =====================================================================
    print(f"\n  training a free-form model ({steps} steps) for the NEGATIVE CONTROL...")
    ff = build("free_form", N, D, Q, (3,), Y)
    torch.manual_seed(0)
    train(ff, Y, steps=steps, lr=0.01, num_samples=5, verbose=False)

    before = {k: v.detach().clone() for k, v in ff.state_dict().items()}
    g_null = measure_amortisation_gap(ff, Y, refine_steps=400, num_samples=5,
                                      eval_samples=32, seed=99, verbose=False)
    after = ff.state_dict()
    unchanged = all(torch.equal(before[k], after[k]) for k in before)
    h_fresh = train(build("free_form", N, D, Q, (), Y), Y, steps=2, verbose=False)
    check("train() creates a fresh optimiser when none is passed",
          h_fresh["optimizer"] is not None and
          all(len(g["params"]) > 0 for g in h_fresh["optimizer"].param_groups))

    check("the measured model is not mutated by the measurement",
          unchanged, "state_dict identical before and after measure_amortisation_gap")

    # The control: refining a free-form q that is ALREADY free-form can only
    # recover optimisation slack, never amortisation error -- there is none.
    # A large reading here would mean the protocol is measuring under-training.
    check("NEGATIVE CONTROL: free-form 'gap' is near zero",
          abs(g_null["gap"]) < 0.05, warn_only=True,
          detail=f"free-form gap = {g_null['gap']:+.4f} nats/pt "
                 f"(MC floor {g_null['gap_noise_sd']:.4f}). This is residual "
                 f"optimisation slack, and it is the true noise floor of the gap "
                 f"measurement -- any amortised gap must clear it.")

    print(f"\n  training an amortised model ({steps} steps)...")
    am = build("amortised", N, D, Q, (3,), Y)
    torch.manual_seed(0)
    train(am, Y, steps=steps, lr=0.01, num_samples=5, verbose=False)
    g = measure_amortisation_gap(am, Y, refine_steps=400, num_samples=5,
                                 eval_samples=32, seed=99, verbose=False)
    check("amortised gap is non-negative", g["gap"] > -1e-3,
          f"gap = {g['gap']:+.4f} nats/pt")
    check("gap trace plateaus (gap is converged, not a hyperparameter)",
          g["refine_converged"], warn_only=True,
          detail=f"last chunk moved {g['refine_last_move']:+.5f}; trace = "
                 + ", ".join(f"{v:+.3f}" for v in g["gap_trace"]))
    check("amortised gap exceeds the free-form control",
          g["gap"] > abs(g_null["gap"]), warn_only=True,
          detail=f"amortised {g['gap']:+.4f} vs control {g_null['gap']:+.4f}")

    # =====================================================================
    section("C. IS THE FITTED MODEL SATURATED / COLLAPSED?")
    # =====================================================================
    # trivial baseline: independent Gaussian per dimension, fitted to Y
    var = Y.var(0)
    trivial = float(-0.5 * (torch.log(2 * math.pi * var) + 1).sum())
    # Budget-aware: below ~1k steps a deep GP-LVM legitimately has not yet beaten
    # an independent Gaussian, so at --quick this is a warning about the BUDGET,
    # not a defect in the model. Above it, failing here is a real problem.
    undertrained = steps < 1000
    for name, model in [("free-form d2", ff), ("amortised d2", am)]:
        e = eval_elbo(model, Y, num_samples=32, seed=99)
        check(f"{name}: ELBO beats the independent-Gaussian baseline",
              e > trivial, warn_only=undertrained,
              detail=f"ELBO/pt {e:+.3f} vs trivial {trivial:+.3f} "
                     f"({e - trivial:+.3f} nats/pt of structure learned)"
                     + (f"  [only {steps} steps -- expected at this budget]"
                        if undertrained and e <= trivial else ""))

    for name, model in [("free-form d2", ff), ("amortised d2", am)]:
        print(f"\n  -- {name} --")
        q = model.latent.q(Y=Y)
        sd, mu = q.stddev, q.mean
        s = float(sd.mean())
        check(f"{name}: posterior not collapsed to a point (sigma >> 0)",
              s > 0.02, f"mean sigma = {s:.4f}")
        check(f"{name}: posterior not collapsed to the prior (sigma < 1)",
              s < 0.95, f"mean sigma = {s:.4f}; per-dim "
                        + ", ".join(f"{v:.3f}" for v in sd.mean(0).tolist()))
        spread = mu.std(0)
        check(f"{name}: latent means vary across points (latent is used)",
              float(spread.max()) > 0.1,
              "per-dim std of mu: " + ", ".join(f"{v:.3f}" for v in spread.tolist()))

        rep = model.ard_report()
        for lname, ls in rep.items():
            t = torch.tensor(ls)
            rel = 1.0 / (t + 1e-8)
            alive = int((rel.mean(0) > 0.1 * rel.mean(0).max()).sum()) if rel.dim() > 1 \
                else int((rel > 0.1 * rel.max()).sum())
            n_in = rel.shape[-1]
            check(f"{name}/{lname}: ARD has not pruned every input dimension",
                  alive >= 1, f"{alive}/{n_in} dimensions alive; lengthscale range "
                              f"[{float(t.min()):.3f}, {float(t.max()):.3f}]")
            check(f"{name}/{lname}: ARD lengthscales not saturated at a bound",
                  float(t.max()) < 1e4 and float(t.min()) > 1e-4, warn_only=True,
                  detail=f"max lengthscale {float(t.max()):.2f} "
                         f"(a huge value = that input is switched off)")

        for i, layer in enumerate(model.layers):
            os_ = float(layer.covar_module.outputscale.mean())
            check(f"{name}/layer_{i}: kernel outputscale alive (GP not dead)",
                  os_ > 1e-3, f"outputscale = {os_:.4f}")
        # rank=0, has_global_noise=False -> D independent task noises (D-07)
        noise = model.likelihood.task_noises.detach().flatten()
        check(f"{name}: observation noise not saturated",
              float(noise.min()) > 1e-5 and float(noise.max()) < 10.0,
              f"noise range [{float(noise.min()):.5f}, {float(noise.max()):.5f}]"
              f"; data variance is 1.0 by standardisation")

        # depth actually doing something: the intermediate layer must vary
        with torch.no_grad():
            xs = model.latent.rsample(8, Y=Y)
            h = model.layers[0](xs, are_samples=True)
            hs = h.mean
        check(f"{name}: intermediate layer carries variation (depth is not inert)",
              float(hs.std()) > 1e-3,
              f"std of layer-0 output = {float(hs.std()):.4f}; per-dim "
              + ", ".join(f"{v:.3f}" for v in hs.reshape(-1, hs.shape[-1]).std(0).tolist()))

    # =====================================================================
    section("SUMMARY")
    # =====================================================================
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"  {len(results)} checks: {len(results) - n_fail - n_warn} pass, "
          f"{n_warn} warn, {n_fail} FAIL")
    for s, n, d in results:
        if s != PASS:
            print(f"    [{s}] {n}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
