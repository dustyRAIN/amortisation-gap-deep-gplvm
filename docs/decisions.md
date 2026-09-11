# Decision log

Decisions already made and the reasoning. If you reverse one, edit the entry rather than
deleting it — the reasoning is what stops the project drifting.

---

## D-01 — Implement the sampled ELBO, not the paper's analytic bound

**Status:** decided.

Damianou & Lawrence's bound (eq. 13/15) is **analytic**. The inducing-point augmentation
makes the awkward `p(F|U,X)` terms cancel, and what remains is computable in closed form
via Ψ statistics — expectations of kernel matrices under q(X). No Monte Carlo anywhere.

We implement the **doubly-stochastic** (sampled) ELBO instead: draw from q(X), push the
draws down the cascade, average.

**Why:**
- Ψ statistics only have closed forms for specific kernels and a mean-field q. It is
  fragile, and deriving/implementing them for an arbitrary-depth latent model is a
  multi-week job on its own.
- The sampled version is what every modern deep GP library uses. It scales, it
  minibatches, and GPyTorch's `DeepGPLayer` is already built to accept samples.
- Our research question is about the amortisation gap, not about the paper's specific
  estimator. The gap exists under either bound.

**What this costs, and what to do about it:**
Every ELBO reading now carries Monte Carlo noise, and our headline result is a
*difference* of two ELBOs. Mitigated by common random numbers, paired comparisons within
a seed, and 64 evaluation samples. **State the noise floor in the results.**

**Presentation impact — important.** Presentation I showed the analytic bound.
Presentation II awards 3 marks for "final probabilistic formulation and consistency with
the implemented model." So slides 2–3 must show the **sampled** ELBO — the one we
actually run — and state plainly that we deviated from the paper and why. Showing both
and explaining the trade demonstrates understanding of each. Quietly showing eq. 13 while
running something else is the failure mode this rubric line exists to catch.

---

## D-02 — Linear mean functions on inner layers

**Status:** decided, already in `GPLayer`.

Zero mean functions on inner GPs cause representational collapse as depth grows
(Duvenaud et al., 2014). Salimbeni & Deisenroth's fix is a linear mean function. This was
**failure mode 1** on our Presentation I slide, so we are pre-empting a known problem
rather than rediscovering it.

Note the tension for Q&A: the paper uses zero-mean GPs throughout. We deviate. Say so.
If you want a bonus result, run depth 2–3 with `mean_type="constant"` and show the
collapse — that is a good failure-cases slide.

---

## D-03 — Initialise the refinement at the encoder's output

**Status:** decided, in `measure_amortisation_gap`.

When measuring the gap we attach a free-form q initialised **at the encoder's own
(μ, σ)**, not randomly. Random init would measure optimisation luck. Starting from the
encoder's answer means every nat of improvement is something the shared mapping could not
express — which is the definition of the amortisation gap.

---

## D-04 — Two datasets, synthetic first

**Status:** decided.

`synthetic_deep_gp()` generates from a stack of known depth. Cheap, deterministic, no
download, and it gives **ground truth** — so you can ask whether ARD recovered the true
structure and whether the bound picks the true depth. That is exactly the evidence the
probabilistic-interpretation criterion (2 marks) wants.

Frey Faces is the dataset from the proposal and stays as the headline. Synthetic is the
safety net if Frey misbehaves, and it strengthens the interpretation either way.

---

## D-05 — Matched initialisation across the two schemes

**Status:** decided, in `AmortisedLatent.__init__`.

`head_log_sigma` is initialised (zero weights, constant bias) to emit the same initial σ
as `FreeFormLatent`'s `init_log_sigma`. Without this the two arms start from different
places and the comparison is confounded. Verified: both start at KL ≈ 237 on a random
50×20 test.

**CORRECTION — the σ is matched, the means are not.** That verification used
`FreeFormLatent`'s *default* `X_init` (`randn * 0.1`). `run_experiment.py` does not use
the default; it passes `pca_init(Y, Q)`, whose scores have per-dimension std 2.17 … 1.12
rather than 1. Measured on the synthetic set (N=320, Q=5):

| arm | KL(q(X)‖N(0,I)) at init |
|---|---|
| free-form, raw PCA (what we actually run) | **14.6** nats/point |
| free-form, PCA scaled to unit variance | 10.4 |
| amortised encoder | **7.9** nats/point |

So the two arms start **6.7 nats/point apart**, not matched. Of the shared 7.9, all but
a rounding error is the σ term: `-log(0.127) + 0.5(0.127² - 1) = 1.572` nats per
dimension × 5 dimensions.

**Does it matter?** Not for the converged result. `why_slack.py` trained variants with
KL-at-init of 14.6, 10.4 and 2.5 and their slack at 4,000 steps was 0.071 / 0.091 /
0.065 — indistinguishable. Initialisation moves the early transient and washes out.

**But do not repeat the claim that the arms start matched.** They do not, and at an
equal-and-short budget that asymmetry is real. Raw PCA is the conventional choice
(GPyTorch's reference GP-LVM does the same) so we keep it — the fix is to train to
convergence and report the control (D-08), not to change the initialiser.

---

## D-06 — Depth 3 is a stretch goal

**Status:** decided.

Depths 1 and 2 answer the question: depth 1 is the correctness gate, depth 2 is where the
effect must appear. Depth 3 strengthens the trend but is the least stable and the most
expensive. A complete two-depth grid beats a three-depth grid with holes.

---

## D-07 — Multitask likelihood, diagonal noise

**Status:** decided, in `DeepGPLVM.__init__`.

The final layer emits a `MultitaskMultivariateNormal` over (N, D), so the likelihood must
be multitask. We use `rank=0, has_global_noise=False`, giving D independent noise
variances — the paper's single-σ Gaussian noise assumption (eq. 1) generalised to one per
output dimension. Full-rank task covariance would be `D×D` and pointless here.

---

## D-08 — Every gap reading needs a free-form negative control

**Status:** decided, in `run_experiment.py` (disable with `--no-control`).

Run the **identical** refinement protocol on the free-form arm. A free-form q has no
amortisation gap by construction, so anything the refinement recovers there is pure
optimisation slack. That is the floor the amortised gap must clear -- and it is much
larger than the Monte Carlo noise floor, so quoting only the MC floor understates the
uncertainty badly.

**Why this entry exists.** `scripts/verify_model.py` measured it. On synthetic data at
1,200 steps: amortised gap **+0.068**, free-form control **+0.062** nats/point. About
90% of the apparent "amortisation gap" was residual optimisation slack. At 300 steps the
control was **+0.77 (depth 1)** and **+2.95 (depth 2)** -- an under-trained model reports
enormous fake gaps, and they grow with depth, which would have produced a textbook-
looking confirmation of our hypothesis out of nothing but under-training.

**How to report it.** Show the raw gap, the control, and the difference. If the raw gap
does not clear the control, the honest reading is "no measurable amortisation gap at this
training budget" -- not a small positive effect. `analyse.py` draws the control as a
dashed line with the slack region shaded, and the summary prints `corrected = raw -
control`.

**This is a strong Q&A answer.** "How do you know you are measuring amortisation and not
under-training?" -- we ran the protocol on a model that cannot have an amortisation gap.

---

## D-09 — The gap must be shown to have converged, and held-out scoring mirrors deployment

**Status:** decided, in `measure_amortisation_gap` and `heldout_elbo`.

Two protocol points that were previously unguarded.

**(a) The gap is monotone in `refine_steps`.** Stop early, understate it; run longer, it
creeps up. So the refinement runs in chunks and records `gap_trace` plus
`refine_converged` (last chunk moved less than `plateau_tol` AND less than the paired MC
noise). An unplateaued gap is reported as a **lower bound**, never as a point estimate.
The original code only warned about the negative-gap direction.

**(b) Held-out scoring is deliberately asymmetric.** `predictive_elbo` -- not `elbo` --
is used for test data: the training bound divides KL(q(U)||p(U)) by `num_data`, so
scoring a smaller test split would inflate a fixed model cost, and q(U) is already
learned, so it is not part of a bound on held-out likelihood. The bound used is
`E_q[log p(Y*|F*)]/N* - KL(q(X*)||p(X*))/N*`.

Each scheme is then scored the way it would actually be deployed: the encoder embeds test
points in **one forward pass**, while the free-form arm has no mapping and must optimise
a fresh q(X*) against the frozen generative model. That asymmetry is not a confound to be
equalised away -- it is amortisation's entire selling point and belongs in the results
table.

---

## D-10 — Frey Faces is subset and PCA-reduced; the budget is the reason

**Status:** decided. `frey_faces(n_subset=..., reduce_dim=...)`, exposed as
`--n-subset` / `--reduce-dim`.

Measured cost per Adam step (depth 2, M=50, CPU):

| | D=560 (pixels) | D=100 (PCA) |
|---|---|---|
| N=1572 (full) | 2440 ms | 380 ms |
| N=400 (subset) | 679 ms | 140 ms |

Cost is dominated by **D, not N**: the output layer carries `batch_shape=[D]`, so
each step runs D batched kernels and Choleskys. At full resolution a 12-cell grid
at 16k steps is ~154 hours. Subset to 500 frames and PCA to 100 dimensions and the
same grid is ~9 hours, or ~6 hours for the 8-cell version.

**Why the budget cannot be cut instead.** D-08's control showed the gap
measurement is meaningless below roughly 1,000 steps, and the field trains far
longer than we had planned — GPyTorch's reference GP-LVM uses 10,000 iterations
for a *single* layer, and Salimbeni & Deisenroth use 20,000 for deep GPs. Steps
are the one thing we must not economise on, so the dataset gives instead.

**What the reduction actually costs.** Measured on the 500-frame subset:

| retained dims | share of pixel variance kept |
|---|---|
| 50 | 90.6% |
| **100 (chosen)** | **96.4%** |
| 200 | 99.2% |

100 dimensions buys a 17× speed-up for 3.6% of the variance. Quote that number on
the slide rather than hand-waving about "most of the signal".

**This changes the dataset and must be said out loud on the dataset slide.** The
model observes PCA coefficients, not pixels. `return_basis=True` hands back the
projection so latent samples can still be rendered as faces for the visualisation.
For scale, Damianou & Lawrence's own digit experiment used 256 dimensions and 150
points — the reduced Frey is closer to the paper's scale than the full set is.

**Order matters:** subset first, then fit the PCA basis, so the projection is
estimated only from data the model is allowed to see.

---

## D-11 — Long runs are resumable

**Status:** decided, in `run_experiment.py` (`--ckpt-every`, `--no-resume`).

A six-hour overnight run must survive losing power. Two levels:

- **Cell level.** Completed cells are already durable in `results.json`; on restart
  any `(depth, seed, scheme)` already present is skipped.
- **Within cell.** Training saves model, optimiser and **RNG state** every
  `--ckpt-every` steps, so a cell resumes mid-training. The checkpoint is removed
  once its cell reaches `results.json`.

Both `results.json` and the checkpoints are written to a temp file and renamed, so
a cut during a write cannot leave a truncated file behind.

**Why the RNG state is saved too.** Without it a resumed run continues on a
different sample stream. Reported numbers would survive that — `eval_elbo` fixes
its own seed — but the paired common-random-numbers discipline the whole project
rests on would quietly stop being reproducible across a restart.

Verified by killing a run mid-cell and restarting: it resumed at step 600 and ran
to completion, then skipped both cells on a third invocation.

---

## D-12 — The encoder gets its own learning rate, selected by ELBO

**Status:** decided. `lr_enc = 0.001` (0.003 is equivalent). Reproduce with
`scripts/encoder_lr_study.py`.

**The failure this fixes.** The 16k-step sweep failed this project's own depth-1
correctness gate: gap **0.584 ± 0.280** nats/point where it should be near zero,
with the amortised ELBO non-monotone along a single checkpointed trajectory in 3 of
6 cells (the free-form arm: 0 of 6). The largest gap readings landed exactly on the
bad checkpoints.

**The mechanism, measured.** One `lr=0.01` was applied to a ~20k-parameter tanh MLP
and to a handful of GP hyperparameters at once. Gradient norms after 1,000 steps:

| arm | q-gradient L2 | GP-gradient L2 | ratio |
|---|---|---|---|
| amortised (encoder) | 42.80 (20,490 params) | 5.76 | **7.4** |
| free-form (table) | 0.30 (3,200 params) | 1.02 | 0.3 |

The encoder's gradient is 7x the GP's and **growing** (2.96 at init to 7.43 at 1k),
while the free-form arm's have decayed to order 1 — that arm is settling and the
amortised one is not.

**Result at depth 1, 8k steps, 3 seeds:**

| config | final ELBO | gap | unstable |
|---|---|---|---|
| lr_enc=0.01 (was) | -8.648 | 0.584 ± 0.280 | 1/3 |
| lr_enc=0.003 | -7.578 | 0.110 ± 0.054 | 0/3 |
| **lr_enc=0.001** | **-7.605** | **0.075 ± 0.035** | **0/3** |
| lr_enc=0.01 + clip 1.0 | -8.677 | 0.617 ± 0.069 | 0/3 |

**Why this is a fix and not tuning toward the answer.** The ELBO *improves* along
with the gap (-8.65 to -7.6). Had only the gap moved, we would be buying the
hypothesis with a hyperparameter. Both moving means the encoder was at the wrong
operating point. The clipping row separates the two problems: clipping cures the
instability (0/3) but leaves the gap at 0.617 and the ELBO at -8.677, so
instability and under-optimisation were distinct faults and only the rate fixes
the second.

**SELECT THE ENCODER RATE BY ELBO, NEVER BY GAP.** Selecting by gap is selecting the
hyperparameter that flatters the hypothesis. By ELBO, 0.003 and 0.001 are tied
within seed noise and both pass the gate; the small gap is a consequence of that
choice, not its criterion. If you re-tune, re-tune on ELBO.

**Deviation from the design rule — declare it on the implementation slide.**
CONTRIBUTING.md requires both arms to share the optimiser and learning rate. This breaks
that deliberately: the gap is meant to measure what a shared mapping *can* express,
so an under-optimised encoder inflates it and manufactures our hypothesis. The
honest procedure is to optimise each q-parameterisation as well as it can be, then
compare. Note also that "the same learning rate" was never quite well defined
across arms — it was being applied to 20,490 encoder weights on one side and 3,200
table entries on the other.

**It reproduces the published anchor.** At 8k steps the free-form arm reaches
-7.500 and the fixed amortised arm -7.605, a shortfall of 0.105 against 1.148
before. That is Lalchand et al.'s "amortised GP-LVM converging to a similar bound"
— the result the gate exists to check.

---

## Open questions — decide these before the full run

- **Latent dimensionality Q.** Currently 5. ARD prunes downward, so Q is an upper bound.
  Check the ARD weights after the first real run and justify the choice on a slide.
- **Inducing points M.** Currently 50, matching the Presentation I feasibility slide.
- **Hidden layer width.** `HIDDEN_FOR_DEPTH` in `run_experiment.py` uses 3. Arbitrary.
  Either justify it or sweep it.
- **Training length.** Both arms must be trained to *convergence*, not to a fixed step
  count, or the gap is confounded by under-training. Check the ELBO curves.
- **Frey Faces subset size.** Full 1,965 or a subset? Affects runtime a lot.
