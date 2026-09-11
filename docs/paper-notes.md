# Paper notes — Damianou & Lawrence (2013)

Verified against the full PDF. Equation numbers are the paper's.

## The chain

```
Z  --f^X-->  X_{H-1}  --...-->  X_1  --f^Y-->  Y
parent        latent            latent        leaves
```

Z = X_H is the parent node with prior N(0, I). Each X_h is an output of one GP and an
input to the next. Y are the observed leaves, N x D. **H is the number of layers.**

Factorisation (two hidden layers):

    p(Y, X_1, X_2, Z) = p(Y|X_1) p(X_1|X_2) p(X_2|Z) p(Z)

## Why exact inference fails

Eq. (6): `log p(Y) = log ∫ p(Y|X) p(X|Z) p(Z)`.

Intractable because X and Z enter **nonlinearly through the GP priors** — a Gaussian
log-density contains K_NN^{-1} and log|K_NN|, so you would need the expected inverse and
expected log-determinant of a random matrix. Neither has a closed form.

## The augmentation trick (this is the key idea)

Eq. (9). Add K pseudo-inputs and their function values U. Since F and U come from the
same GP, `p(F|X) = ∫ p(F|U,X) p(U) dU` **exactly** — nothing is approximated yet.

The payoff: p(U) has covariance K_MM = k(X~, X~), which depends only on the pseudo-input
locations, **not on the latent X**.

Then choose Q (eq. 10) to contain the same conditional:

    Q = p(F^Y|U^Y,X) q(U^Y) q(X) · p(F^X|U^X,Z) q(U^X) q(Z)

In the bound's ratio p/Q, the `p(F|U,X)` factors cancel. Eq. (12). The intractable
log-determinant and inverse disappear.

**For Q&A:** augmentation is EXACT. The approximation is the *choice* of q(U) and q(X),
not the addition of U. Being able to separate those two is a good answer.

## What is left: the Psi statistics

X still enters through p(F|U,X)'s mean and covariance:

    mean = K_NM K_MM^{-1} U
    cov  = K_NN - K_NM K_MM^{-1} K_MN

Averaging over q(X) needs only three quantities:

    tr<K_NN>,   <K_NM>,   <K_MN K_NM>

These are the **Psi statistics**. K_MM^{-1} sits *outside* every expectation. The Psi
statistics are Gaussian integrals of an exponentiated quadratic, which have closed forms
— which is why eq. (5)'s ARD kernel is not a stylistic choice.

## The bound

Eq. (13), two hidden layers:

    F_v = g_Y + r_X + H_q(X) - KL(q(Z) || p(Z))

Eq. (15), general:

    F_v = Σ_m g_Y^(m) + Σ_h Σ_m r_Xh^(m) + Σ_h H_q(Xh) - KL(q(Z) || p(Z))

Terms:
- `g_Y`  leaf term — expected fit of data given the layer above, plus an inducing
         correction. Identical to the Titsias & Lawrence Bayesian GP-LVM bound.
- `r_X`  same for a hidden layer, but averaging over its **inputs** as well as its
         outputs. One per layer, which is why depth just adds terms.
- `H_q(X)` entropy — present ONLY because each layer is both a GP output and a GP input,
         so its terms do not collapse into a KL.
- `KL(q(Z)||p(Z))` the only KL, at the parent node where the prior sits. The Occam's
         razor that drives unused ARD dimensions to zero.

Maximised, jointly over model parameters and variational parameters.

## Expectation operators (asked for by the Pres I rubric)

- in `g_Y`: over `p(F^Y|U^Y,X) q(U^Y) q(X)`
- in `r_X`: over `p(F^X|U^X,Z) q(U^X) q(X) q(Z)` — averages over inputs too

## Variational distribution

Eq. (11): `q(X) = Π_q N(mu^X_q, S^X_q)`, `q(Z) = Π_q N(mu^Z_q, S^Z_q)` — Gaussian,
factorised over dimensions. q(U^Y), q(U^X) are free-form.

## Kernel

Eq. (5), ARD exponentiated quadratic:

    k(x_i, x_j) = sigma_ard^2 exp( -0.5 * Σ_q w_q (x_iq - x_jq)^2 )

One weight per latent dimension. Driving w_q to zero switches that dimension off. This is
the mechanism behind automatic structure discovery.

## Complexity

Section 3.2: sparse methods reduce each generative GP mapping from O(N^3) to O(NM^2).
Also states inducing points and q's parameters are **variational, not model
parameters**, so adding layers does not add many model parameters.

## Experiments

- **Toy (4.1):** data from a three-level stack. Deep GP recovers correct dimensionality
  per layer via ARD; stacked Isomap and stacked PCA do not.
- **Toy regression:** 25 training points from a warped process. Deep GP beats a standard
  GP over 10 repeats (fig. 3b). Nonstationarity and long-range correlations appear.
- **Motion capture (4.2):** CMU MOCAP high-five, 78 frames, 124 dims. Two-level
  hierarchy, conditionally independent per subject.
- **Digits (4.3):** 50 each of {0,1,6} from USPS, 16x16. Depths 1 to 5. Lower bound
  increased with the number of layers; nearest-neighbour errors improved. Single-layer
  made 5 mistakes; depth-5 had one. BIC considered, no effect on ranking (footnote 6).
- **Fig. 8:** sampling from layers 1-2 gives local features (is a "0" closed, how big is
  a "6"'s circle); sampling the parent node's dominant dimensions gives much more varying
  output. Lower layers local, higher layers abstract.

## Initialisation (Section 4)

Greedy layerwise: dimensionality-reduce the observations for the first hidden layer, then
repeat upward. PCA and Bayesian GP-LVM both tried; end result did not vary much. **We use
PCA — same as the paper.**

## What the paper does NOT do (our opening)

Selects depth by comparing F_v across architectures, but never checks whether the gap
between F_v and the true log p(Y) is the same size under each candidate. If the gap
widens with depth, deeper models are penalised for a reason unrelated to their quality.
