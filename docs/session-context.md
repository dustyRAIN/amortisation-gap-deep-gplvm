# Session context — what happened before this repo existed

Presentation I was built in a chat session that does not carry over. This file records
the parts of that conversation that still matter, so nothing is silently lost.

## Presentation I is done

A 15-slide deck and a 13-minute four-speaker script were produced and checked against the
rubric. If you need them and they are not in the repo, ask the team for
`CSE756_Presentation_I.pptx` and `CSE756_Presentation_I_Script.md`.

Speaker split was A: slides 1-3 and 15, B: 4-6, C: 7-10, D: 11-14. Keep the same split
for Presentation II so each member owns a consistent area.

## Errors we caught and fixed — do not reintroduce them

The first draft of the deck had four factual errors, found by checking against the actual
PDF. They are worth knowing because they are easy to make again:

1. **A generic VAE-style ELBO was presented as the paper's objective.** It is not. The
   paper's bound is eq. 13/15 with g_Y, r_X, entropy and one KL. See `paper-notes.md`.
2. **We claimed the method needs Monte Carlo sampling.** It does not — the paper's bound
   is analytic. (Our implementation IS sampled; that is a deliberate deviation, D-01.)
3. **Notation was inverted.** We had X at the top and H as hidden layers. The paper uses
   Z at the top, X_h as intermediate layers, and H as the *number* of layers.
4. **Inducing point locations were listed as model parameters.** Paper §3.2 says they are
   variational parameters. This matters — it is a claim our project puts pressure on.

Also corrected: prior work is RBM-based deep belief nets and the MAP hierarchical GP-LVM
(Lawrence & Moore 2007), not generic deep neural networks. The motivating limitation is
that MAP gives no marginal likelihood, hence no model selection.

## Claims we softened, and why

At one point we asserted "nobody has checked this". That was too strong. The accurate
version: the amortisation gap has been measured **at a single architecture** and comes
back small. What is unmeasured is how it behaves **as depth grows**. Keep this precise —
a marker who knows the literature will notice the difference.

Related: Cremer et al. (2018) is our load-bearing citation and we have only secondhand
summaries of it, which disagreed with each other. See the checklist at the end of
`prior-art.md`.

## Presentation I content that Presentation II should NOT repeat

The recap slide is one slide. Do not re-explain what a Gaussian Process is, the
augmentation trick, or the four terms of the bound. Assume the audience saw it.

What Presentation II adds: the actual implemented formulation, the code, the baseline,
numbers, and interpretation.

## Environment note

The deep GP-LVM code in `src/` was written and smoke-tested against torch 2.14 and
gpytorch 1.15.2. Depths 1, 2 and 3 all run; a full small grid on synthetic data completes
in about a minute on CPU. The numbers from those runs were throwaway smoke tests and have
been deleted deliberately — **there are no real results in this repo yet.**
