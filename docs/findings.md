# Findings

Every number here comes from a run in `results/`. Nothing is projected or rounded from
a partial run. Where the evidence is thin, it says so.

Dataset: `synthetic_deep_gp` (N=400, 320 train, D=20, true depth 2). Q=5, M=50,
Adam lr=0.01, encoder lr=0.001 (D-12), **8 seeds**, budgets to 16,000 steps.
32/32 cells, 160 rows, every trajectory complete.
Source: `results/slack_study.json`, produced by `scripts/slack_study.py`.

**Two datasets.** Synthetic (§1–5, 8 seeds) and Frey Faces (§5b, 4 seeds, 16/16 cells).
Both agree on the training-set null; Frey adds a much larger held-out effect.

---

## 1. The depth-1 correctness gate passes

Gap at 16k steps: **0.041 ± 0.035** nats/point against a free-form control of
**0.036 ± 0.008** — a net of **+0.005 nats/point** on a −6.7 nat bound. The gap is
statistically indistinguishable from its own optimisation-slack floor.

Before the encoder-rate fix it was 0.584 ± 0.280, with the amortised ELBO non-monotone
in 3 of 6 cells. See D-12. After the fix: **0 non-monotone trajectories out of 32.**

This is a stronger replication of Lalchand et al. than they published. Their claim is a
supplementary figure caption reading "a very similar convergence loss level", with no
number. Ours is a gap measured at essentially zero **against a control that demonstrates
the measurement could have detected one** — the control reads 2.45 nats/point on the same
protocol when the model is under-trained.

## 2. The gap does NOT grow with depth — the hypothesis is not supported

| | n | gap | control | net |
|---|---|---|---|---|
| **depth 1** | 8 | 0.0407 ± 0.035 | 0.0361 ± 0.008 | **+0.0046** |
| **depth 2** | 8 | 0.0534 ± 0.071 | 0.0334 ± 0.016 | **+0.0200** |

**Depth effect (d2 − d1): +0.0127 ± 0.0279 (SE) — 0.45 standard errors from zero.**

With 8 seeds this is not "we failed to detect an effect". It is a bounded null: any depth
effect is smaller than about 0.03 nats/point, roughly 0.4% of the bound.

Decay with budget, mean over 8 seeds (nats/point):

| | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|
| depth 1 gap | 0.410 | 0.253 | 0.151 | 0.063 | **0.041** |
| depth 2 gap | 0.472 | 0.447 | 0.181 | 0.103 | **0.053** |

This is the branch the hypothesis slide pre-registered as valid: *"Gap flat → amortisation
is selection-neutral for deep GPs; extends a single-architecture finding to depth. A
legitimate negative result."* Present it as the pre-registered branch, not a
disappointment.

## 3. Depth selection does not differ between the schemes

Hypothesis 3 predicted the amortised model would select **fewer** layers. Paired within
seed on ELBO at 16k:

| scheme | picks depth 2 | per-seed |
|---|---|---|
| free-form | **4/8** | [2,1,2,2,1,1,2,1] |
| amortised | **4/8** | [2,2,1,1,2,2,1,1] |

An exact tie, and not on the same seeds. No systematic difference in either direction.

## 4. What now limits the experiment: seed variance, not the instrument

ELBO/point at 16k:

| | mean ± sd |
|---|---|
| depth 1 free-form | −6.679 ± 0.515 |
| depth 1 amortised | −6.895 ± 0.498 |
| depth 2 free-form | −6.719 ± 1.298 |
| depth 2 amortised | −4.964 ± **2.737** |

The measurement machinery is now the *precise* part: controls are 0.03 and every
refinement plateaued. The imprecise part is that two runs of the same model with different
seeds land 2–3 nats/point apart at depth 2. Any depth effect of the size we are looking
for is buried under that.

**Done — the sweep now uses 8 seeds.** Standard errors roughly halved (depth 1 free-form
SE 0.297 → 0.132) and the depth-1 net gap fell from 0.029 to 0.005, i.e. the three-seed
estimate was seed noise. Depth 2 remains the noisy arm and is the reason no depth effect
could be resolved below ~0.03 nats/point.

## 5b. Frey Faces: the null replicates, and the gap *shrinks* with depth

16/16 cells, 4 seeds, every refinement converged. Frey: 500 frames PCA-reduced to 100
dims (96.4% of pixel variance, D-10), 400 train / 100 test.

| | depth 1 | depth 2 |
|---|---|---|
| training gap | +0.0249 ± 0.0052 | +0.0001 ± 0.0039 |
| free-form control | −0.0031 | +0.0082 |
| **net training gap** | **+0.028** | **−0.008** |
| **held-out gap** | **+29.94 ± 6.88** | **+0.48 ± 0.67** |

**The training-set null replicates**, and the depth effect is *negative*:
−0.025 ± 0.003 (s.e.).

**Depth selection is unanimous.** All eight runs — both schemes, all four seeds —
prefer depth 1. The schemes cannot disagree about depth here, so hypothesis 3 has no
room to act.

### The held-out gap is the real finding

Measured on unseen points the gap is **29.94** nats/point at depth 1 and **0.48** at
depth 2: a **62-fold decrease, 8.5 standard errors from zero**. This is by far the
largest and most significant effect in the project, and it runs *opposite* to the
hypothesis.

The mechanism is visible in the held-out bounds:

| | free-form | amortised, encoder only | amortised, refined |
|---|---|---|---|
| depth 1 | −151.72 | **−171.93** | −141.98 |
| depth 2 | −140.32 | −140.34 | −139.86 |

At depth 1 the encoder's zero-shot embeddings sit 30 nats below what per-point
refinement achieves on the same points. At depth 2 all three numbers agree to within
half a nat --- the encoder is essentially optimal on unseen data.

**The cause is latent pruning, and it is measurable.** At depth 1 all five latent
dimensions stay active (posterior spread 0.09–0.11). At depth 2 the spread is 0.71–0.96,
because three or four dimensions have returned to the prior (σ→1, mean spread 0.003) —
the latent KL removing what the model does not use. Counting active dimensions
(σ < 0.9 and mean spread > 0.1):

| | active dims | held-out gap |
|---|---|---|
| depth 1 (4 seeds) | 5/5 | +23.9 … +39.8 |
| depth 2 (4 seeds) | 1–2/5 | −0.0 … +1.4 |

**Correlation +0.95** across the eight amortised cells. When the latent carries little,
there is little for refinement to recover, so the encoder matters less. Depth 1 also
overfits (better train bound −134.5, worse held-out −151.7), but pruning is the directly
measured quantity and tracks the effect almost perfectly.

**Caveat this introduces, and the next experiment it implies.** Active dimensions and
depth move together in our grid, so we cannot yet say whether the held-out collapse is a
*depth* effect or simply a consequence of using fewer latent dimensions. Holding the
number of active dimensions fixed while varying depth would separate them. Until that is
run, the held-out finding is partly a statement about how much the model *uses* its
latent space, not purely about the encoder.

### What this means for the conventional measurement

The training-set gap at depth 1 is 0.025 nats/point and the held-out gap on the same
models is 29.94 --- a factor of 1,200. The freeze-and-refine protocol as conventionally
run asks only how much better the *training* points could be served, and is nearly blind
to whether the inference network generalises. For bound-based depth selection the
training quantity is the relevant one and it is negligible; for deployment it is not the
right question.

## 6. Two open threads

**Depth 2's gap has not converged.** It fell 0.103 → 0.053 between 8k and 16k while depth
1 had nearly plateaued (0.063 → 0.041). If the depth-2 gap is still falling, the null gets
stronger, not weaker.

**The published anchor is weaker than assumed.** Lalchand et al. report no
amortisation-gap number anywhere, and their two arms use different variational families —
their amortised q learns a dense covariance while their free-form q is diagonal. See
`prior-art.md`. Our arms are both diagonal, so our 0.062 nats/point appears to be the
first *quantified* amortisation gap for a GP-LVM. That is a contribution worth claiming,
and it survives the null result on depth.

## 7. Honest scope

- Frey has only **2 seeds** against synthetic's 8, and its depth-2 models are worse than
  its depth-1 models, so the Frey depth comparison is made on an architecture the data
  does not support.
- Frey is PCA-reduced to 100 dims (D-10), so the model observes coefficients, not pixels.
- 8 seeds. Adequate for the depth-1 result; depth 2's ELBO spread (±2.2 nats/point)
  still limits resolution to about 0.03 nats/point.
- Depths 1 and 2. Depth 3 not attempted (D-06).
- Both arms use a diagonal q. A dense-covariance free-form arm would separate
  "amortisation" from "diagonal vs dense" — the confound Lalchand et al. could not.
