"""
Variational distributions q(X) over the bottom latent layer.

This module holds the ONLY thing that differs between the proposed model and the
baseline. Everything downstream -- kernels, inducing points, layers, likelihood,
optimiser, seeds -- is shared, so any measured difference is attributable to this
choice alone. That is the whole experimental design in one file.

    FreeFormLatent  -> BASELINE. One (mu_n, sigma_n) stored per data point.
                       This is Damianou & Lawrence eq. (11) as written.
    AmortisedLatent -> PROPOSED. One encoder network maps any y_n to (mu_n, sigma_n).
                       This is the VAE encoder / GP-LVM "back-constraint".
"""

import torch
import torch.nn as nn


class BaseLatent(nn.Module):
    """Common interface.

    q(Y, idx)  -> a factorised Normal over the latent layer
    rsample(S) -> reparameterised draws, shape (S, batch, latent_dim)
    kl(...)    -> total KL[ q(X) || N(0, I) ], summed over points and dimensions
    """

    def __init__(self, n: int, latent_dim: int):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim

    def q(self, Y=None, idx=None) -> torch.distributions.Normal:
        raise NotImplementedError

    def rsample(self, num_samples: int, Y=None, idx=None) -> torch.Tensor:
        return self.q(Y=Y, idx=idx).rsample(torch.Size([num_samples]))

    def kl(self, Y=None, idx=None) -> torch.Tensor:
        q = self.q(Y=Y, idx=idx)
        p = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.stddev))
        return torch.distributions.kl_divergence(q, p).sum()

    @property
    def n_variational_params(self) -> int:
        raise NotImplementedError


class FreeFormLatent(BaseLatent):
    """BASELINE. A lookup table: row n holds the parameters for data point n.

    Parameter count grows as 2 * n * latent_dim. Each row is optimised
    independently of the others, so it can reach the per-point optimum. By
    construction there is no amortisation gap -- this defines the ceiling.

    Args:
        n: number of training points.
        latent_dim: Q.
        X_init: (n, latent_dim) initial means. Use PCA of Y, as the paper does.
        init_log_sigma: pre-softplus initial spread. Keep this identical to the
            amortised model's initial bias so both schemes start comparably.
    """

    def __init__(self, n, latent_dim, X_init=None, init_log_sigma=-2.0):
        super().__init__(n, latent_dim)
        if X_init is None:
            X_init = torch.randn(n, latent_dim) * 0.1
        self.mu = nn.Parameter(X_init.clone().float())
        self.log_sigma = nn.Parameter(torch.full((n, latent_dim), float(init_log_sigma)))

    def q(self, Y=None, idx=None):
        mu, log_sigma = self.mu, self.log_sigma
        if idx is not None:
            mu, log_sigma = mu[idx], log_sigma[idx]
        return torch.distributions.Normal(mu, nn.functional.softplus(log_sigma) + 1e-6)

    @property
    def n_variational_params(self):
        return self.mu.numel() + self.log_sigma.numel()


class AmortisedLatent(BaseLatent):
    """PROPOSED. One shared encoder produces (mu, sigma) for any data point.

    Parameter count is independent of n, and a new point is embedded in one
    forward pass. But a single smooth mapping must serve every point at once, so
    it cannot in general reach the per-point optimum for all of them. That
    shortfall is the amortisation gap.

    Design note: head_log_sigma is initialised to output a constant equal to the
    free-form model's init_log_sigma, so the two schemes start from the same
    place. Without this the comparison is confounded by initialisation.
    """

    def __init__(self, n, latent_dim, data_dim, hidden=(128, 128), init_log_sigma=-2.0):
        super().__init__(n, latent_dim)
        layers, prev = [], data_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head_mu = nn.Linear(prev, latent_dim)
        self.head_log_sigma = nn.Linear(prev, latent_dim)
        nn.init.zeros_(self.head_log_sigma.weight)
        nn.init.constant_(self.head_log_sigma.bias, float(init_log_sigma))

    def q(self, Y=None, idx=None):
        if Y is None:
            raise ValueError("AmortisedLatent needs Y -- the encoder reads the data.")
        h = self.body(Y)
        return torch.distributions.Normal(
            self.head_mu(h), nn.functional.softplus(self.head_log_sigma(h)) + 1e-6
        )

    @property
    def n_variational_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# DENSE-COVARIANCE VARIANTS
#
# Why these exist. Lalchand et al. (2022) compare a per-point scheme (B-SVI) with
# an encoder (AEB-SVI) and the encoder wins on every reported metric. But their two
# arms do not use the same variational family: their Table 2 gives B-SVI 2NQ local
# parameters -- a mean and a DIAGONAL variance per point -- while section 3.3 says
# AEB-SVI "learns a dense covariance matrix ... capturing correlations across
# dimensions". Their amortised q is strictly more expressive than their free-form
# one, so their result mixes two effects and cannot separate them.
#
# With diagonal AND dense variants of both schemes, the inference gap decomposes:
#
#   amortised_diag -> free_form_diag     amortisation gap, diagonal family
#   free_form_diag -> free_form_dense    the cost of the diagonal assumption
#   amortised_dense -> free_form_dense   amortisation gap, dense family
#
# Lalchand's comparison is amortised_dense vs free_form_diag: the sum of an
# amortisation effect and a family effect, with opposite signs. That is why their
# encoder can appear to beat the per-point scheme.
# ---------------------------------------------------------------------------


def _tril_from_raw(raw, latent_dim):
    """Lower-triangular Cholesky factor with a positive diagonal, from raw values."""
    L = torch.tril(raw)
    diag = nn.functional.softplus(torch.diagonal(L, dim1=-2, dim2=-1)) + 1e-6
    return L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(diag)


class _DenseMixin:
    """q(x_n) is a full-covariance Gaussian, so the KL needs the MVN form."""

    def kl(self, Y=None, idx=None):
        q = self.q(Y=Y, idx=idx)
        p = torch.distributions.MultivariateNormal(
            torch.zeros_like(q.mean),
            scale_tril=torch.eye(self.latent_dim, device=q.mean.device).expand(
                *q.mean.shape[:-1], self.latent_dim, self.latent_dim))
        return torch.distributions.kl_divergence(q, p).sum()


class DenseFreeFormLatent(_DenseMixin, BaseLatent):
    """BASELINE, dense. One mean and one full Q x Q covariance per data point.

    The most expressive Gaussian q available per point, so it is the true ceiling
    within the Gaussian family: no amortisation gap AND no diagonal assumption.
    Parameter count grows as n * (Q + Q^2).
    """

    def __init__(self, n, latent_dim, X_init=None, init_log_sigma=-2.0):
        super().__init__(n, latent_dim)
        if X_init is None:
            X_init = torch.randn(n, latent_dim) * 0.1
        self.mu = nn.Parameter(X_init.clone().float())
        raw = torch.zeros(n, latent_dim, latent_dim)
        raw[:, range(latent_dim), range(latent_dim)] = float(init_log_sigma)
        self.raw_L = nn.Parameter(raw)

    def q(self, Y=None, idx=None):
        mu, raw = self.mu, self.raw_L
        if idx is not None:
            mu, raw = mu[idx], raw[idx]
        return torch.distributions.MultivariateNormal(
            mu, scale_tril=_tril_from_raw(raw, self.latent_dim))

    @property
    def n_variational_params(self):
        return self.mu.numel() + self.raw_L.numel()


class DenseAmortisedLatent(_DenseMixin, BaseLatent):
    """PROPOSED, dense. One encoder emits a mean AND a full Cholesky factor.

    This is Lalchand et al.'s AEB-SVI parameterisation. Compare it against
    DenseFreeFormLatent -- NOT against the diagonal FreeFormLatent, which is the
    comparison their paper makes.
    """

    def __init__(self, n, latent_dim, data_dim, hidden=(128, 128), init_log_sigma=-2.0):
        super().__init__(n, latent_dim)
        layers, prev = [], data_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head_mu = nn.Linear(prev, latent_dim)
        self.head_L = nn.Linear(prev, latent_dim * latent_dim)
        nn.init.zeros_(self.head_L.weight)
        bias = torch.zeros(latent_dim, latent_dim)
        bias[range(latent_dim), range(latent_dim)] = float(init_log_sigma)
        with torch.no_grad():
            self.head_L.bias.copy_(bias.flatten())

    def q(self, Y=None, idx=None):
        if Y is None:
            raise ValueError("DenseAmortisedLatent needs Y -- the encoder reads the data.")
        h = self.body(Y)
        raw = self.head_L(h).reshape(*h.shape[:-1], self.latent_dim, self.latent_dim)
        return torch.distributions.MultivariateNormal(
            self.head_mu(h), scale_tril=_tril_from_raw(raw, self.latent_dim))

    @property
    def n_variational_params(self):
        return sum(p.numel() for p in self.parameters())


def free_form_counterpart(latent):
    """The per-point scheme in the SAME variational family as `latent`.

    The gap protocol refines against this. Refining a dense q toward a diagonal one
    (or the reverse) would measure the family change, not amortisation.
    """
    return DenseFreeFormLatent if isinstance(latent, _DenseMixin) else FreeFormLatent


def is_amortised(latent):
    """True for the encoder schemes -- they read Y, the tables do not."""
    return hasattr(latent, "body")
