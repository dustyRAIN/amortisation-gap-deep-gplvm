"""
Deep GP latent variable model -- the multi-layer UNSUPERVISED deep GP.

Why this file exists: GPyTorch ships `BayesianGPLVM` (unsupervised, ONE layer)
and `DeepGP` (multi-layer, but SUPERVISED -- its forward takes observed inputs).
Neither is what we need. This wires the two together: a variational latent layer
at the bottom feeding a stack of DeepGPLayers.

Objective implemented here is the SAMPLED (doubly-stochastic) ELBO, not Damianou
& Lawrence's analytic Psi-statistics bound. See docs/decisions.md D-01 for why,
and put that reasoning on the formulation slide -- Presentation II awards 3 marks
for the formulation matching the implemented system.

    ELBO_per_point = E_q[log p(Y|F)]/N  -  KL(q(U)||p(U))/N  -  KL(q(X)||p(X))/N

The first two terms come from GPyTorch's VariationalELBO (well-tested; we do not
reimplement them). The latent KL is computed explicitly in `elbo()` below so the
accounting is visible rather than hidden in a registered loss term -- you will be
asked about this in Q&A.
"""

from typing import Sequence

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy


class GPLayer(DeepGPLayer):
    """One GP mapping in the cascade.

    mean_type: 'linear' for inner layers. Zero/constant means on inner layers
    cause representational collapse as depth grows (Duvenaud et al. 2014); the
    linear mean is Salimbeni & Deisenroth's fix. This is failure mode 1 from
    Presentation I, and we are pre-empting it rather than rediscovering it.

    The kernel is ARD (one lengthscale per input dimension). Those lengthscales
    are what prune unused latent dimensions -- inspect them for the
    probabilistic-interpretation slides.
    """

    def __init__(self, input_dims, output_dims, num_inducing=64, mean_type="linear"):
        batch_shape = torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        inducing_points = torch.randn(*batch_shape, num_inducing, input_dims)

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = (
            LinearMean(input_dims, batch_shape=batch_shape)
            if mean_type == "linear"
            else ConstantMean(batch_shape=batch_shape)
        )
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def ard_lengthscales(self):
        """(output_dims, input_dims) ARD lengthscales. Large => dimension ignored."""
        return self.covar_module.base_kernel.lengthscale.detach().squeeze(-2)


class DeepGPLVM(DeepGP):
    """Latent layer -> [GPLayer]*depth -> observed Y.

    Args:
        latent: a BaseLatent (FreeFormLatent or AmortisedLatent).
        data_dim: D, the observed dimensionality.
        hidden_dims: widths of the INTERMEDIATE layers, top to bottom.
            () -> depth 1 (a plain Bayesian GP-LVM; our correctness check)
            (Q,) -> depth 2, and so on.
        num_inducing: M per mapping.
    """

    def __init__(self, latent, data_dim, hidden_dims: Sequence[int] = (), num_inducing=64):
        super().__init__()
        self.latent = latent
        self.data_dim = data_dim
        self.depth = len(hidden_dims) + 1

        dims = [latent.latent_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(GPLayer(dims[i], dims[i + 1], num_inducing, mean_type="linear"))
        layers.append(GPLayer(dims[-1], data_dim, num_inducing, mean_type="linear"))
        self.layers = torch.nn.ModuleList(layers)

        # Final layer emits a MultitaskMultivariateNormal over (N, D), so the
        # likelihood must be multitask. rank=0 + no global noise => D independent
        # noise variances, which is the paper's assumption (eq. 1) generalised
        # to one sigma per output dimension.
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=data_dim, rank=0, has_global_noise=False
        )

    # -- forward -------------------------------------------------------------
    def forward(self, Y, num_samples=5, idx=None, latent_sample=None):
        """Push S draws of q(X) down the cascade. Returns the final MVN."""
        if latent_sample is None:
            x = self.latent.rsample(num_samples, Y=Y, idx=idx)   # (S, B, Q)
        else:
            x = latent_sample
        h = self.layers[0](x, are_samples=True)
        for layer in self.layers[1:]:
            h = layer(h)
        return h

    # -- objective -----------------------------------------------------------
    def elbo(self, Y, num_data, num_samples=5, idx=None):
        """Per-data-point ELBO in nats. This is the number we compare.

        Returned value is an average over S samples, so it carries Monte Carlo
        noise. When comparing two schemes use `common_seed` in gap.py so the
        same draws are used for both -- otherwise the noise can swamp the effect.
        """
        out = self.forward(Y, num_samples=num_samples, idx=idx)
        mll = gpytorch.mlls.DeepApproximateMLL(
            gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=num_data)
        )
        gp_part = mll(out, Y)                       # per-point, inducing KL included
        latent_kl = self.latent.kl(Y=Y, idx=idx) / num_data
        return gp_part - latent_kl

    def predictive_elbo(self, Y, num_samples=64, idx=None):
        """Per-point bound on log p(Y*) for HELD-OUT data, in nats.

        Deliberately NOT the same functional as `elbo()`. The training bound
        includes KL(q(U)||p(U)) scaled by 1/num_data; on a test set that term
        would be divided by N_test instead of N_train, inflating a fixed model
        cost purely because the split is smaller. It is also not part of a bound
        on held-out likelihood -- q(U) is already learned and held fixed here.

        So this returns the predictive bound proper:

            E_q[log p(Y*|F*)]/N*  -  KL(q(X*)||p(X*))/N*

        which is comparable across schemes, depths and split sizes. Use `elbo()`
        for training and for the gap; use this for the held-out column.
        """
        N = Y.shape[0]
        out = self.forward(Y, num_samples=num_samples, idx=idx)
        # (num_samples, N) -- expected_log_prob already sums over the D tasks
        ell = self.likelihood.expected_log_prob(Y, out).sum(-1).mean(0)
        latent_kl = self.latent.kl(Y=Y, idx=idx)
        return (ell - latent_kl) / N

    def elbo_terms(self, Y, num_data, num_samples=64, idx=None):
        """The three terms of `elbo()`, separately, in nats per point.

        Diagnostic only -- `elbo()` remains the objective. This exists to answer
        "which term did the refinement actually recover?":

            ell    E_q[log p(Y|F)]/N   -- fit. Recovering this means q's MEANS
                                          were in the wrong place.
            kl_u   KL(q(U)||p(U))/N    -- frozen during refinement, so any change
                                          here is a bug in the freeze.
            kl_x   KL(q(X)||p(X))/N    -- Recovering this means q's SPREAD was
                                          mis-set; a cheaper, more local error.

        Identity checked in verify_model.py: elbo() == ell - kl_u - kl_x.
        """
        N = Y.shape[0]
        out = self.forward(Y, num_samples=num_samples, idx=idx)
        ell = self.likelihood.expected_log_prob(Y, out).sum(-1).mean(0)
        kl_u = sum(l.variational_strategy.kl_divergence().sum() for l in self.layers)
        kl_x = self.latent.kl(Y=Y, idx=idx)
        return {"ell": float(ell) / N, "kl_u": float(kl_u) / num_data,
                "kl_x": float(kl_x) / num_data}

    # -- diagnostics ---------------------------------------------------------
    def parameter_counts(self):
        var = self.latent.n_variational_params
        total = sum(p.numel() for p in self.parameters())
        return {"variational_q_X": var, "total": total, "model_only": total - var}

    def ard_report(self):
        return {f"layer_{i}": l.ard_lengthscales().cpu().numpy() for i, l in enumerate(self.layers)}
