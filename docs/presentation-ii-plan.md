# Presentation II — plan

**12–15 minutes + questions. 15 marks. 14 slides on the brief's suggested map.**

## Marking rubric — where the marks are

| Criterion | Marks | Where it is earned |
|---|---|---|
| Final formulation, **consistent with the implemented model** | 3.0 | Slides 2–3 |
| Implementation quality, reproducibility, understanding of decisions | 3.0 | Slides 4–5 + Q&A |
| Baseline choice and experimental design | 2.0 | Slide 6 |
| Preliminary results, quantitative comparison, visualisations | 3.0 | Slides 7–9 |
| Probabilistic interpretation of results | 2.0 | Slides 10–11 |
| Failure cases, remaining work, clarity, individual Q&A | 2.0 | Slides 12–14 |

## Slide map

| # | Section | Content |
|---|---|---|
| 1 | Title + recap | Question, hypothesis, model, dataset. **Brief** — do not re-run Presentation I |
| 2–3 | Final formulation | The **sampled** ELBO as implemented. Variables, distributions, parameters, latents, objective, inference, sampling |
| 4–5 | Implementation | GPyTorch + our own latent layer; why no off-the-shelf deep GP-LVM exists; preprocessing; architecture; hyperparameters; compute |
| 6 | Baseline | Free-form q; why it is the right comparison; what single difference is tested |
| 7–9 | Results | Gap vs depth plot; comparison table; ELBO curves; ARD weights; latent space |
| 10–11 | Probabilistic interpretation | Did it learn the intended distribution? Did ARD prune? Was inference stable? Did depth buy anything? |
| 12 | Failure cases | What broke, what it reveals |
| 13 | Remaining work | What is still needed for the report |
| 14 | Takeaway | Current answer to the research question |

## Slide 2–3 — read this carefully, it is 3 marks

Presentation I showed the paper's **analytic** bound. We implement the **sampled** one.
The rubric explicitly rewards the formulation matching the implemented system, so:

- Show the sampled ELBO we actually optimise:
  `ELBO/point = E_q[log p(Y|F)]/N − KL(q(U)||p(U))/N − KL(q(X)||p(X))/N`
- Point at `model.elbo()` and show the three terms in the code.
- State the deviation from the paper and why (see `docs/decisions.md` D-01).
- Show both forms side by side. Explaining the trade demonstrates understanding of each;
  quietly showing eq. 13 while running something else is what this rubric line catches.

Also cover: variables (Y, X_h, Z, U), distributions (p(Z)=N(0,I), Gaussian likelihood, GP
conditional, factorised q), parameters (ARD weights, amplitude, noise) vs variational
parameters (μ, S, inducing locations), inference (sampled variational), sampling
(reparameterised draws pushed down the cascade).

## Slides 4–5 — implementation

The brief warns that running a downloaded implementation without demonstrating
understanding is not sufficient. Our defence is strong and should be stated:

> GPyTorch ships `BayesianGPLVM` (unsupervised, one layer) and `DeepGP` (multi-layer,
> supervised). Neither is a multi-layer unsupervised deep GP, so we wrote the latent
> layer and the ELBO accounting ourselves.

Cover: framework and versions; Frey Faces preprocessing (per-pixel standardisation, the
split); architecture (latent dim, hidden widths, M inducing points, ARD RBF kernels,
linear inner means and why — D-02); training (Adam, lr, steps, S samples, PCA init as the
paper does); compute (CPU/GPU, wall-clock per run); and the major decisions from
`docs/decisions.md`.

## Slide 6 — baseline

Free-form q is the paper's own scheme and has **zero amortisation gap by construction**,
so it defines the ceiling. The single tested difference is the parameterisation of q —
everything else is byte-identical. Comparison criteria: the gap in nats/point, ELBO,
held-out log-likelihood, parameter count, wall-clock.

## Slides 7–9 — results

**Minimum bar from the brief: one numerical comparison between proposed and baseline.**
Aim higher:

1. **Main plot:** amortisation gap (nats/point) vs depth, error bars over seeds. Mark the
   MC noise floor as a shaded band — this is what makes the plot honest.
2. **Table:** depth × scheme → ELBO, held-out LL, gap, variational params, wall-clock.
3. **ELBO curves** for both schemes — evidence both were trained to convergence, which
   is what makes the comparison fair.
4. **ARD weights per layer** — how many dimensions survived.
5. **Latent space scatter** coloured by something meaningful.

## Slides 10–11 — probabilistic interpretation (2 marks)

The brief says explicitly: do not report only a predictive metric. Answer these directly:

- Did the model learn the intended distribution? On synthetic data of known depth, did
  ARD recover the true structure?
- Did approximate inference behave as expected? Did the depth-1 gap come out near zero as
  published work predicts?
- Was the ELBO estimate stable? What is the seed-to-seed spread versus the effect size?
- Did q collapse? Check the average posterior σ — near zero means collapse.
- Did depth buy anything? Did the bound improve, and did the two schemes disagree about
  which depth is best?
- Was sampling computationally reasonable? How did wall-clock scale with depth?

## Slide 12 — failure cases

The brief **encourages** reporting unsuccessful experiments where they explain model
behaviour. Candidates: depth-3 instability; negative gap from under-converged refinement;
posterior collapse with constant means (worth running deliberately as a demonstration);
gap smaller than the noise floor.

## Slide 13 — remaining work

Honest list: more seeds; depth 3; the Cremer et al. verification from
`docs/prior-art.md`; sweeping M or hidden width; and the held-out-likelihood arm if it is
not done.

## Slide 14 — takeaway

The **current** answer to the research question, with its uncertainty stated. If the
evidence is thin, say so — a hedged honest answer beats an overclaimed one.

## Timing

12–15 minutes across 14 slides is roughly 55 seconds per slide. Tighter than
Presentation I. The recap (slide 1) must be genuinely brief — resist re-explaining deep
GPs. Budget the saved time on slides 7–11, which carry 5 of the 15 marks.
