# Prior art

Read this before proposing a new direction — two obvious ones are already closed.

**Confidence note:** these summaries came from web searches during planning, not from
reading every paper end to end. Where that matters it is flagged. **Verify the load-
bearing ones before the final report.**

---

## The foundational paper

**Damianou & Lawrence (2013), "Deep Gaussian Processes", AISTATS.** Verified against the
full PDF.

- Stacks GPs so each layer is the output of one GP and the input to the next.
- Variationally marginalises the whole latent cascade, extending Titsias & Lawrence
  (2010) from one layer to any number.
- Bound is **analytic** — eq. 13 for two layers, eq. 15 in general:
  `F_v = Σ g_Y + Σ r_Xh + Σ H_q(Xh) − KL(q(Z)||p(Z))`
- Complexity O(N³) → O(NM²) per mapping (§3.2).
- Headline: on 150 USPS digits (50 each of 0/1/6, 16×16), the bound rose with every layer
  added, up to five — the deepest tried.
- Initialisation: greedy layerwise, PCA or Bayesian GP-LVM (§4).
- §3.2 states explicitly that inducing points and the parameters of q are **variational,
  not model parameters**, which is why adding layers costs little in model complexity.
  **Our project puts pressure on exactly this claim**, since an encoder replaces those
  variational parameters with model weights.

---

## What closed Candidate A (calibration)

**Salimbeni & Deisenroth (2017), "Doubly Stochastic Variational Inference for Deep
Gaussian Processes", NeurIPS.** Runs deep GP vs single-layer GP across UCI benchmarks and
reports deep GPs doing as well as or better, gaining from depth without overfitting even
on small data. Test log-likelihood is a headline metric. Also introduced the linear mean
function fix for the collapse pathology.

**"Evaluating Uncertainty in Deep Gaussian Processes" (2025), arXiv:2504.17719.** Targets
DGP/DSPP calibration specifically, with ECE and calibration plots against baselines like
deep ensembles.

Together these make "is a deep GP better calibrated than a shallow one" a settled
question. Do not go back to it.

---

## What closed Candidate B (bound as a ruler)

Importance-weighted bounds for deep GPs are taken: **Salimbeni, Dutordoir, Hensman &
Deisenroth (2019), "Deep Gaussian Processes with Importance-Weighted Variational
Inference", ICML.** Their motivation is that the Gaussian variational form gives an
inaccurate posterior; the IW objective consistently beats classical VI, especially for
deeper models. There is also an IWVI-GPLVM baseline benchmarked on Frey Faces and MNIST
in the VAIS-GPLVM paper (arXiv:2408.06710).

Note the direction of their result: **tightening the bound helps more for deeper models.**
That is consistent with our hypothesis and worth citing as support.

---

## The amortisation gap — our actual territory

**Cremer, Li & Duvenaud (2018), "Inference Suboptimality in Variational Autoencoders".**
Gives the decomposition:

```
inference gap = approximation gap + amortisation gap
```

Approximation gap = true posterior vs. the best member of the variational family.
Amortisation gap = what the inference network achieves vs. that best member.

**Caution — verify this one yourself.** Two sources characterised it differently: one
cited it as finding the gap on MNIST minimal; another said both gaps contribute
significantly. The likely reconciliation is that the finding is *conditional* — small on
MNIST, larger with harder data and more expressive decoders. That reading supports our
hypothesis with "depth" substituted for "decoder expressiveness". **Read the paper and
check whether they varied decoder capacity and plotted the gap against it.** If they did,
our framing needs adjusting; if they only varied datasets, our depth angle is cleaner.

**Kim et al., "Semi-Amortized Variational Autoencoders".** Varies training set size
(25%–100%) and inference network capacity, with a PixelCNN decoder. This is the "sweep N
and measure the gap" experiment already done — with **neural** decoders.

---

## Amortised GP-LVMs — the model exists, the depth study does not

Adding an encoder to a GP-LVM is old: back-constraints (Lawrence &
Quiñonero-Candela, 2006), then Bui & Turner (2015), Dai et al. (2016).

**Lalchand, Ravuri & Lawrence (AISTATS 2022), "Generalised GPLVM with Stochastic
Variational Inference".** VERIFIED against the full PDF (PMLR v151, 24 pages). Builds
both variants explicitly — B-SVI (per-point) and AEB-SVI (encoder).

**They never measure an amortisation gap.** The word "gap" appears **zero** times in the
paper. "Amortised" appears 11 times (§3.3 is "Amortised Inference with Encoders"), so the
idea is present, but it is never quantified. There is no ELBO number for either scheme
anywhere in the paper.

**The claim we were leaning on is one supplementary figure caption, with no number.**
Figure 10, verbatim:

> "Analysis of oilflow dimensionality reduction with ARD. Final plot shows ELBO loss for
> B-SVI (non-amortised) and the amortised NNEncoder model with the latter achieving a
> very similar convergence loss level to the non-amortised model."

Qualitative, supplementary, oilflow only. Do **not** describe this as "our depth-1 data
point, already published" — there is no published number to anchor to.

**Their numerical comparisons are test RMSE and test NLPD, and the amortised model WINS
every one of them:**

| dataset | metric | B-SVI (free-form) | AEB-SVI (amortised) |
|---|---|---|---|
| Oilflow | RMSE | 0.0925 (0.025) | **0.067 (0.0016)** |
| qPCR | RMSE | 0.554 (0.017) | **0.539 (0.004)** |
| Taxi-cab | RMSE | 249 (81) | **232 (22)** |
| Oilflow | NLPD | −11.3105 (0.243) | **−11.392 (0.147)** |
| qPCR | NLPD | 27.844 (1.429) | **25.422 (2.004)** |

**THE CONFOUND — this is the important part.** Their two arms do not use the same
variational family. §3.3 says AEB-SVI "learns a dense covariance matrix (parameterised
through a factorization) per data-point thereby capturing correlations across dimensions",
and its covariance network emits Q² outputs. But their Table 2 gives B-SVI only **2NQ**
local parameters — a mean and a *diagonal* variance per point.

So their amortised q is **strictly more expressive** than their free-form q. AEB-SVI
beating B-SVI is therefore not evidence that amortisation is free; part of it is a
full-covariance q beating a diagonal one. Their setup cannot separate the two effects.

**Our comparison is the cleaner one.** Both our arms are diagonal (`Normal` in both
`FreeFormLatent` and `AmortisedLatent`), so the only difference is per-point storage
versus a shared mapping — which is exactly what the amortisation gap is defined to be.

**Depth genuinely does not appear.** "layer"/"layers" occurs 3 times, every one of them
about the hidden layers of the encoder MLP, never about stacking GPs. Single-layer
throughout. Our depth axis is untouched by this paper — checklist item confirmed.

**Also worth citing.** Their encoders are tiny: the oilflow mean network is
(12, 10, 5, 12) ≈ 300 parameters, against our (20, 128, 128, 5) ≈ 20,490 — ours is ~70x
larger, which is relevant given D-12. And their Table 6 reports the amortised scheme is
**2x slower to train** despite O(1) test predictions, because the encoder weights are
global parameters updated every minibatch. That belongs in our wall-clock column.

There is also a 2024 paper on amortised VI for deep GPs (arXiv:2409.12301).

**The gap nobody has filled:** every amortisation-gap study uses a *single* architecture,
or varies N/encoder capacity with a *neural* decoder. Nobody has plotted the gap **against
depth** with a GP decoder. That is our contribution.

---

## Also relevant

- **Duvenaud et al. (2014), "Avoiding pathologies in very deep networks."** Diagnosed
  representational collapse in deep GPs. Basis of failure mode 1 and decision D-02.
- **"An Analysis of Posterior Collapse, Parameterization and Initialization in Variational
  Deep Gaussian Processes"** (arXiv:2606.25882). Read before debugging depth-3
  instability — it will save time.
- **Titsias & Lawrence (2010), Bayesian GP-LVM.** The single-layer bound our `g_Y` term
  is identical to.

---

## Before the final report

- [ ] Read Cremer et al. 2018 properly. It is the load-bearing citation for our
      hypothesis and we have been relying on secondhand summaries.
- [x] **DONE.** Pulled Lalchand et al. (PMLR v151) and read the full 24-page PDF. Depth
      does not appear. They report no amortisation-gap number at all, and their two arms
      use different variational families. Written up above — the anchor is much weaker
      than this document previously claimed.
- [ ] Search specifically for "amortization gap" + "Gaussian process" / "GPLVM" +
      depth/layers. If someone has run this sweep, we need to know now.
