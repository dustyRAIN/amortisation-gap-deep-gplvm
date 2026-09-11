"""Datasets.

Two are provided and you should use BOTH:

  synthetic_deep_gp(...)  -- data generated from a stack of GPs of KNOWN depth.
      This is the one that makes the probabilistic-interpretation slides strong:
      you can ask whether ARD recovered the true structure, and whether the
      bound picks the true depth. Cheap, deterministic, no download.

  frey_faces(...)         -- the real dataset from the proposal. 1,965 frames of
      560 pixels (20x28), one person, varying expression and head pose.
"""

import os
import numpy as np
import torch


def synthetic_deep_gp(n=300, data_dim=20, latent_dim=2, hidden_dim=3,
                      true_depth=2, noise=0.05, seed=0):
    """Sample Y from a genuine composition of random GP-like maps.

    We use random Fourier features to draw smooth non-linear maps cheaply, which
    is a sample from a GP prior with an RBF kernel in the limit of many features.
    Returns (Y, Z_true) with Y standardised.
    """
    rng = np.random.default_rng(seed)

    def random_smooth_map(d_in, d_out, n_feat=256, lengthscale=1.0):
        W = rng.normal(0, 1.0 / lengthscale, size=(d_in, n_feat))
        b = rng.uniform(0, 2 * np.pi, size=n_feat)
        A = rng.normal(0, 1.0 / np.sqrt(n_feat), size=(n_feat, d_out))
        return lambda X: np.sqrt(2.0) * np.cos(X @ W + b) @ A

    Z = rng.normal(size=(n, latent_dim))
    h = Z
    dims = [latent_dim] + [hidden_dim] * (true_depth - 1) + [data_dim]
    for i in range(len(dims) - 1):
        f = random_smooth_map(dims[i], dims[i + 1])
        h = f(h)
        h = (h - h.mean(0)) / (h.std(0) + 1e-8)
    Y = h + noise * rng.normal(size=h.shape)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return torch.tensor(Y, dtype=torch.float32), torch.tensor(Z, dtype=torch.float32)


def frey_faces(path="data/frey_rawface.mat", n_subset=None, seed=0,
               reduce_dim=None, return_basis=False):
    """Load Frey Faces as (n, 560) float32, standardised.

    If the file is missing, download it once:
        mkdir -p data
        curl -L -o data/frey_rawface.mat \
          https://raw.githubusercontent.com/SheffieldML/GPmat/master/datasets/data/frey_rawface.mat

    The URL everyone cites -- cs.nyu.edu/~roweis/data/frey_rawface.mat -- now
    returns 403; Roweis's page has decayed. The mirror above is SheffieldML/GPmat,
    Neil Lawrence's own lab repository, which is about as authoritative a source as
    exists for this project: he is a co-author of the foundational paper. Verified
    to load as (1965, 560).

    If that also fails, `sklearn.datasets.fetch_olivetti_faces()` is an acceptable
    substitute -- 400 images, 64x64. Say so on the dataset slide if you switch; do
    not quietly change datasets.

    `n_subset` and `reduce_dim` exist because the full set is not affordable at
    the training budget this experiment needs (see docs/decisions.md D-10).
    Measured cost per Adam step, depth 2, M=50, on CPU:

        N=1572 D=560  2440 ms     N=1572 D=100   380 ms
        N= 400 D=560   679 ms     N= 400 D=100   140 ms

    Cost is dominated by D, not N: the output layer carries batch_shape=[D], so
    it runs D batched kernels and Choleskys per step. `reduce_dim` PCA-projects
    the pixels first, which is the single biggest lever.

    THIS CHANGES THE DATASET and must be declared on the dataset slide: the
    model then observes PCA coefficients, not pixels. `return_basis` hands back
    the projection so latent samples can still be rendered as faces.
    """
    from scipy.io import loadmat

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. See the docstring for the download command, "
            "or use synthetic_deep_gp() / sklearn's Olivetti faces instead."
        )
    img = loadmat(path)["ff"].T.astype(np.float64)  # (1965, 560)
    Y = (img - img.mean(0)) / (img.std(0) + 1e-8)
    if n_subset is not None:
        idx = np.random.default_rng(seed).choice(len(Y), size=n_subset, replace=False)
        Y = Y[idx]
    Y = torch.tensor(Y, dtype=torch.float32)

    basis = None
    if reduce_dim is not None and reduce_dim < Y.shape[1]:
        # Subset FIRST, then fit the projection, so the basis is estimated only
        # from data the model is allowed to see.
        n_pix = Y.shape[1]
        U, S, V = torch.pca_lowrank(Y, q=min(reduce_dim, min(Y.shape) - 1))
        basis = V[:, :reduce_dim]                       # (n_pix, reduce_dim)
        # Retained variance must be measured against the FULL spectrum. S holds
        # only the q singular values pca_lowrank was asked for, so dividing by
        # S.sum() would report 100% for any q.
        total = float(((Y - Y.mean(0)) ** 2).sum())
        var = float((S[:reduce_dim] ** 2).sum()) / total
        Y = Y @ basis
        print(f"  frey_faces: PCA {Y.shape[0]}x{n_pix} -> {tuple(Y.shape)}, "
              f"retains {var:.1%} of pixel variance")
        # per-dimension standardisation, as for the pixel version and for
        # synthetic_deep_gp, so the likelihood's per-dim noise starts calibrated
        Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    return (Y, basis) if return_basis else Y


def train_test_split(Y, test_frac=0.2, seed=0):
    """Held-out split for the secondary metric."""
    n = Y.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_test = int(round(test_frac * n))
    return Y[perm[n_test:]], Y[perm[:n_test]]


def pca_init(Y, latent_dim, basis=None, return_basis=False):
    """PCA initialisation for the latent means -- what the paper does (Sec. 4).

    Pass `basis` (the V from a previous call, obtained with return_basis=True) to
    project NEW points onto the SAME axes the training latents were initialised
    on. Test points initialised on their own basis start in a coordinate system
    the trained decoder has never seen, which shows up as a spuriously bad
    held-out number.
    """
    if basis is None:
        U, S, V = torch.pca_lowrank(Y, q=min(latent_dim, min(Y.shape) - 1))
        basis = V[:, :latent_dim]
    X = Y @ basis
    return (X, basis) if return_basis else X
