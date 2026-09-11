# Contributing — working context for this repo

Read this first. It is the working context for this repo.

## What this project is

A graduate research mini-project for **CSE756: Modern Probabilistic Machine Learning**.
Foundational paper: **Damianou & Lawrence, "Deep Gaussian Processes", AISTATS 2013**
(arXiv:1211.0358). Presentation I is done and marked-ready. We are now building
**Presentation II: implementation and preliminary experimental results.**

**We are NOT reproducing the paper.** We are running our own experiment, which uses a
deep GP-LVM as the substrate.

## The research question

> Does the **amortisation gap** in a deep GP latent variable model grow with the number
> of layers — and if it does, does depth selection by the variational bound differ
> between amortised and free-form inference?

### Hypothesis (three parts)

1. **At depth 1 the gap is near zero.** This is not a prediction, it is a **correctness
   gate**. If our depth-1 gap is not small, we have a bug and we stop and debug rather
   than proceeding. **It fired once and caught a real bug — see D-12.**
   *Anchor, stated precisely:* Lalchand et al. (2022) report **no amortisation-gap number
   at all**; their only relevant claim is a supplementary figure caption saying the
   amortised model reaches "a very similar convergence loss level". Do not cite them for a
   value. See `prior-art.md` — the anchor is qualitative, and their two arms use different
   variational families.
2. **The gap grows with depth.** Composing GPs makes the true posterior over the latent
   layers less Gaussian and less smoothly related to the data. A single smooth encoder is
   the wrong tool for a target like that.
3. **Consequence:** an amortised model selects fewer layers than a free-form one on
   identical data.

### Both outcomes are results — say so up front

- **Gap grows** → amortisation silently biases model selection; a caution for anyone
  pairing an encoder with a deep GP.
- **Gap flat** → amortisation is selection-neutral for deep GPs; extends a
  single-architecture finding to depth. A legitimate negative result.

Put this on the hypothesis slide *before* running anything. Do not let it look like a
consolation prize discovered afterwards.

## The experimental design in one sentence

Hold the generative model completely fixed; change **only** how q(X) is parameterised.

| | Proposed | Baseline |
|---|---|---|
| q(X) | one encoder network → (μ, σ) for any y | a table: one (μₙ, σₙ) per point |
| Class | `AmortisedLatent` | `FreeFormLatent` |
| Params | O(1) in N | O(N·Q) |
| Gap | the thing we measure | zero by construction — defines the ceiling |

Everything else — kernel, inducing points, layer widths, likelihood, optimiser, learning
rate, seeds — is identical. If you find yourself changing something else between the two
arms, that is a bug in the experiment, not an improvement.

## How the gap is measured

```
1. Train the amortised model.                                  -> ELBO_amortised
2. FREEZE every generative parameter (p.requires_grad_(False)).
3. Attach a fresh free-form q, INITIALISED AT THE ENCODER'S OUTPUT.
4. Optimise only that q, in chunks, recording the gap after each. -> ELBO_refined
5. gap = ELBO_refined - ELBO_amortised     (>= 0 by construction)
6. Run steps 2-5 again on the FREE-FORM arm. That is the negative control.
```

Two things that will silently ruin this if you get them wrong:

- **Step 3.** Random init measures optimisation luck, not the gap. Starting from the
  encoder's own answer means any improvement is exactly what the shared mapping could
  not express.
- **Step 2.** If the decoder moves, you are comparing two different posteriors and the
  number means nothing.

A **negative** gap means the refinement under-converged or the freeze failed. `train.py`
warns about this. Do not report a negative gap as a finding.

### Step 6 is not optional — read this before quoting any gap

A free-form q **cannot** have an amortisation gap. So whatever the same refinement
recovers on the free-form arm is pure optimisation slack, and the amortised gap has to
clear it before it means anything. Measured on synthetic data (`verify_model.py`):

| Training budget | Amortised gap | Free-form control | What it means |
|---|---|---|---|
| 1,200 steps | +0.068 | +0.062 | ~90% of the "gap" was slack |
| 300 steps | +0.183 (d1), +0.116 (d2) | +0.83 (d1), +2.45 (d2) | meaningless; model not converged |

Note the trap in the second row: an under-trained model produces large fake gaps **that
grow with depth**. That is our hypothesis, manufactured out of nothing but under-training.
Always report raw gap, control, and the difference. `analyse.py` draws the control as a
dashed line and shades the slack region.

### The gap is monotone in `refine_steps`

Stop early and you understate it; run longer and it creeps up. `measure_amortisation_gap`
returns `gap_trace` and `refine_converged`. If the refinement has not plateaued, the gap
is a **lower bound** and must be worded that way.

## Monte Carlo noise — read this before comparing anything

Our ELBO is **sampled**, not analytic (see `docs/decisions.md` D-01). The gap is a
*difference between two noisy numbers*. If the gap is small, noise can swamp it.

Mitigations already built in, keep them:
- `eval_elbo(..., seed=)` fixes the seed, so both arms see **common random numbers**.
- Comparisons are **paired** within a seed, never averaged across seeds first.
- `--eval-samples 64` by default for the reported numbers; training uses fewer.
- `elbo_noise_floor` / `paired_gap_noise_floor` re-evaluate a frozen model under several
  seeds. Every row carries `elbo_noise_sd` and `gap_noise_sd`; the main figure shades it.

**The MC floor is the smaller of the two floors.** The free-form control (above) is
typically an order of magnitude larger. A gap that clears the MC noise but not the
control is not a result.

If you add any new comparison, apply the same discipline.

## Notation — follow the paper, do not invent your own

| Symbol | Meaning |
|---|---|
| `Y` (N×D) | observed leaves — the data |
| `Xₕ` (N×Qₕ) | intermediate latent layer h, for h = 1 … H−1 |
| `Z` | the parent latent node, the topmost (H-th) layer, prior N(0, I) |
| `U` | inducing variables at M pseudo-inputs |
| `H` | the **number of layers**, never a layer itself |
| `θ, σ_ε` | model parameters: ARD weights, amplitude, noise |
| `μ, S` | variational parameters of q — **not** model parameters (paper §3.2) |

Presentation I uses exactly these. Do not drift.

## Repo layout

```
src/latent.py    FreeFormLatent, AmortisedLatent  <- the only experimental variable
src/model.py     DeepGPLVM: latent layer -> stack of DeepGPLayers -> Y
src/train.py     training loop + the gap measurement protocol
src/data.py      synthetic_deep_gp (known depth), frey_faces, splits, PCA init
scripts/run_experiment.py   the depth x scheme grid -> results/results.json
scripts/analyse.py          tables and plots from results.json
scripts/verify_model.py     39 correctness + degeneracy checks; run after any src/ change
docs/            project brief, prior art, decision log, paper notes, Pres II plan
```

## Why we had to write the model ourselves

GPyTorch (verified against 1.15.2) ships:
- `gpytorch.models.gplvm.BayesianGPLVM` — unsupervised, **one layer**
- `gpytorch.models.deep_gps.DeepGP` — multi-layer, but `forward(self, x)` expects
  **observed** inputs

Neither is a multi-layer *unsupervised* deep GP. GPflow/GPflux and GPy have the same
split. So there is no off-the-shelf deep GP-LVM to download.

This is good for marks: the Presentation II brief says running a complete existing
implementation without demonstrating understanding is not sufficient. We wrote the
bottom layer, which is exactly the demonstration required. **Be able to explain
`latent.py` and the ELBO accounting in `model.elbo()` line by line — that is Q&A bait.**

## Working agreements

- **Never invent numbers.** Every figure in the deck comes from a real run in
  `results/`. If a run has not happened, the slide says so.
- **Report failures.** The brief explicitly encourages reporting unsuccessful
  experiments where they explain model behaviour. Instability at depth 3 is a *result*,
  not an embarrassment.
- **Depth 3 is a stretch goal, not a promise.** Depths 1 and 2 answer the question. A
  complete small grid beats a large grid with holes.
- **Interpret probabilistically.** The rubric explicitly says do not report only
  RMSE/accuracy. Look at: did ARD prune sensibly? Did q collapse? Was the gap estimate
  stable across seeds? Did the ELBO converge or plateau? Did depth buy anything?
- **Run `--dataset synthetic --steps 800` first** to check any change end-to-end in a
  couple of minutes before committing to a long Frey Faces run.

## Immediate next steps

1. `pip install -r requirements.txt`
2. `python scripts/verify_model.py` — 39 checks that this is a genuine unsupervised deep
   GP-LVM and that the fitted model is not collapsed. All 39 pass as of the last run.
   Re-run after any change to `src/`.
3. `python scripts/run_experiment.py --dataset synthetic --depths 1 2 --seeds 0 --steps 800`
   — confirms the pipeline runs.
4. Get Frey Faces (`src/data.py:frey_faces` docstring has the URL).
5. Full run: depths 1 and 2, seeds 0/1/2, ~3000 steps. **Check two things before trusting
   anything: the depth-1 gap is small, and the free-form control is well below the
   amortised gap.** If the control is comparable, train longer — nothing else you measure
   will mean anything.
6. `python scripts/analyse.py` → table + gap-vs-depth plot.
7. Build the 14-slide deck from `docs/presentation-ii-plan.md`.

### Budget, and the Frey run

Steps are the one thing not to economise on (D-08). The field trains far longer than
this repo originally planned: GPyTorch's reference GP-LVM uses **10,000** iterations for
a *single* layer, Salimbeni & Deisenroth **20,000** for deep GPs. Treat 3,000 as a smoke
test, not a result.

Full-resolution Frey cannot be afforded at that budget (~154 h for a 12-cell grid), so it
is subset and PCA-reduced — D-10, and it must be declared on the dataset slide:

```bash
python scripts/run_experiment.py --dataset frey --n-subset 500 --reduce-dim 100 \
    --depths 1 2 --seeds 0 1 --steps 16000 --refine-steps 2000 --ckpt-every 500
```

That is roughly a six-hour overnight run. It is **resumable** (D-11): if it dies, run the
exact same command again — completed cells are skipped and an interrupted cell restarts
from its last checkpoint. `--no-resume` forces a clean start.

## Known risks

| Risk | Mitigation |
|---|---|
| Depth ≥2 unstable / posterior collapse | Linear mean functions on inner layers (already in `GPLayer`); PCA init as the paper does |
| **Amortised arm unstable / gate fails** | The encoder needs its own learning rate (D-12). At the shared `lr=0.01` its gradient is 7× the GP's and growing, the ELBO goes non-monotone, and the depth-1 gap reads 0.58 instead of ~0 |
| Gap smaller than MC noise | More eval samples, common random numbers, paired seeds; report the noise floor honestly |
| **Gap is really under-training** | The free-form negative control (D-08). This is the bigger risk of the two and it fakes our hypothesis convincingly |
| Depth 3 will not train in time | Scoped as a stretch goal from the start |
| Result is null | Pre-registered on the hypothesis slide as a valid outcome |
