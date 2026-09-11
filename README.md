# Does the amortisation gap grow with depth in deep GP-LVMs?

A measurement protocol, a negative control, and a bounded null result.

CSE756 *Modern Probabilistic Machine Learning* mini-project. Foundational paper:
Damianou & Lawrence, [*Deep Gaussian Processes*](https://arxiv.org/abs/1211.0358),
AISTATS 2013.

## The question

Deep GPs select their depth by comparing a variational lower bound across
architectures. That is only sound if the slack between the bound and the true
marginal likelihood is comparable at each depth. We ask whether one component of that
slack — the **amortisation gap**, incurred by replacing per-point variational
parameters with a shared encoder — grows with the number of layers.

## What we found

**The measurement is harder than the result.** The standard protocol for measuring an
amortisation gap (freeze the generative model, re-optimise q, report the improvement)
is confounded by residual optimisation error — severely enough to manufacture the
effect under test. On under-trained models it reports a gap that *grows with depth*,
reaching **2.45 nats/point**, while the Monte Carlo noise floor sits at 0.005–0.009
and would have declared that artifact overwhelmingly significant.

We introduce a **negative control**: run the identical refinement on a per-point
variational family, which cannot have an amortisation gap by construction. Whatever it
recovers is pure optimisation error.

With the control in place and models trained to convergence:

| | amortisation gap | free-form control | net |
|---|---|---|---|
| depth 1 | 0.041 ± 0.035 | 0.036 ± 0.008 | **+0.005** |
| depth 2 | 0.053 ± 0.071 | 0.033 ± 0.016 | **+0.020** |

The depth effect is **0.45 standard errors from zero** over 8 seeds — a *bounded*
null: any effect is smaller than ≈0.03 nats/point. Neither scheme systematically
prefers a different depth.

**On held-out data the same gap is ~1,200× larger** (29.94 nats/point at depth 1), and
it collapses **62-fold** by depth 2 — 8.5 standard errors. That collapse tracks latent
pruning: depth 1 keeps all 5 latent dimensions active, depth 2 keeps 1–2, and the
correlation between active dimensions and the held-out gap is **+0.95**.

Evidence log, with every number and its caveats: [`docs/findings.md`](docs/findings.md).
The write-up itself is a graded coursework deliverable and is not published here.

## Why we wrote the model ourselves

GPyTorch ships `BayesianGPLVM` (unsupervised, single-layer) and `DeepGP` (multi-layer,
but supervised — its forward pass takes observed inputs). Neither is a multi-layer
*unsupervised* deep GP, and GPflow/GPflux and GPy share the split. We implemented the
latent layer, both q(X) parameterisations, the cascade, and the ELBO accounting on top
of GPyTorch's variational machinery.

We verify the model is genuinely unsupervised rather than assuming it: holding a latent
draw fixed and replacing Y with pure noise changes the generative output by exactly
zero.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/verify_model.py                 # 55 correctness + degeneracy checks
python scripts/run_experiment.py --dataset synthetic --depths 1 2 --seeds 0 --steps 800
```

Frey Faces is not redistributed here. Fetch it once:

```bash
mkdir -p data && curl -L -o data/frey_rawface.mat \
  https://raw.githubusercontent.com/SheffieldML/GPmat/master/datasets/data/frey_rawface.mat
```

(The usual `cs.nyu.edu/~roweis` URL now returns 403. The mirror above is Neil
Lawrence's own lab repository.)

## Layout

| path | what |
|---|---|
| `src/latent.py` | the four q(X) parameterisations — the only experimental variable |
| `src/model.py` | `DeepGPLVM`: latent layer → stack of GP layers → Y |
| `src/train.py` | training, the gap protocol, the negative control, held-out scoring |
| `scripts/run_experiment.py` | the depth × scheme grid |
| `scripts/verify_model.py` | 55 automated correctness and degeneracy checks |
| `scripts/verify_documents.py` | recomputes ground truth from `results/` and cross-checks every number quoted in the deck and script |
| `results/` | every run behind every reported number |
| `results/slack_study_UNFIXED_lr.json` | the failure case: the run where the encoder shared a learning rate, failed the depth-1 gate, and produced a gap that grew with depth |
| `docs/decisions.md` | decision log, including the ones that turned out wrong |
| `slides/` | Presentation II deck and speaking script |

Runs are checkpointed and resumable at cell and within-cell granularity, including the
RNG state — so an interrupted run resumes on the same sample stream and the paired
common-random-numbers discipline survives a restart. This was not hypothetical; the
machine lost power twice mid-experiment.
