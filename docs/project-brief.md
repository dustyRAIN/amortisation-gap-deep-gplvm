# Project brief

Longer background. `CONTRIBUTING.md` is the operational summary; this is the "why".

## Where the question came from

Damianou & Lawrence's headline claim is a **model-selection** claim: on 150 USPS digits,
their variational bound keeps rising as layers are added, so a five-layer hierarchy is
justified on very little data.

That claim rests entirely on the bound being a fair yardstick. The bound is a *lower*
bound: `log p(Y) = F_v + gap`. Comparing architectures by F_v is only fair if the gap
underneath each candidate is about the same size. The paper never checks this.

We are not attacking the paper. We are asking about the instrument.

## Why we narrowed to amortisation

Three candidates were considered (Presentation I slide 10):

- **A — calibration.** Is a deep GP better calibrated than a shallow one? Closed by
  Salimbeni & Deisenroth (2017) and a 2025 calibration study.
- **B — bound as ruler.** Does the bound recover a known depth? Strong, but overlaps
  published work on importance-weighted bounds for GP models.
- **C — the amortisation gap vs depth.** Selected.

C survived because the depth axis is genuinely unmeasured, and because it lands directly
on the paper's own claim: if the gap grows with depth, the paper's five layers is a
conservative *floor*, not a ceiling. We would be strengthening their conclusion by
questioning their instrument, which is a better story than "we found a flaw".

## What the amortisation gap is

Cremer et al. (2018) decompose:

    inference gap = approximation gap + amortisation gap

**Approximation gap:** the true posterior versus the best member of the variational
family. Caused by q being Gaussian and factorised when the truth is not.

**Amortisation gap:** what a shared inference network achieves versus that best member.
Caused by one smooth mapping having to serve every data point at once.

A free-form q has **no amortisation gap by construction** — every point is optimised
independently — and pays for it with O(N) parameters and no way to embed new points
without re-optimising.

## Why depth should matter

Stacking GPs makes the true posterior over the latent layers less Gaussian, more
correlated, and less smoothly related to the observed data. An encoder is a smooth
deterministic map from y to (μ, σ). A smooth map is a poor fit for a target whose shape
changes erratically across the data.

So the gap should be an increasing function of depth even if it is negligible at depth 1.

Supporting evidence, indirect: Salimbeni et al. (2019) found importance-weighted bounds
help **more for deeper models**, which is the same shape of claim — the variational
approximation strains as depth grows.

## Why it matters practically

Amortised inference is the default in essentially every modern deep latent variable
model. If it biases depth selection in deep GPs, anyone using an encoder with one is
quietly choosing a smaller model than their data supports — and the bound will not tell
them, because the bound is what is being distorted.

## Honest positioning

- **Depth 1 is replication, not discovery.** Lalchand et al. (2022) already show an
  amortised GP-LVM reaching a similar bound at a single architecture. That published
  number is our correctness anchor, and it is why depth 1 is a gate rather than a result.
- **The project lives or dies on depths 2 and 3.** The curve either bends or it does not.
- **A flat curve is a real answer** and is pre-registered as such.

## Scope discipline

Every hour spent making the deep GP-LVM prettier is an hour not spent on the actual
experiment. The model is a substrate. The deliverable is a number: the gap as a function
of depth, with an honest error bar and an honest noise floor.
