"""Training, the amortisation-gap measurement, and the honest-number helpers.

The gap protocol (this is the core of the project -- get it right):

    1. Train the amortised model to convergence.        -> ELBO_amortised
    2. FREEZE every generative parameter. The true posterior p(X|Y) is now fixed.
    3. Attach a fresh free-form q, INITIALISED AT THE ENCODER'S OUTPUT.
    4. Optimise only that q.                             -> ELBO_refined
    5. gap = ELBO_refined - ELBO_amortised   (>= 0 by construction)

Step 3 matters. If you initialise the refinement randomly you are measuring
optimisation luck, not the gap. Starting from the encoder's own answer means any
improvement is exactly what the shared mapping could not express.

Step 2 matters too. If the decoder moves, you are comparing two different
posteriors and the number is meaningless.

Step 4 has a direction of failure the original protocol did not guard: the gap
is MONOTONE INCREASING in refine_steps. Stop early and you understate it; run
longer and it creeps up. So `measure_amortisation_gap` records a gap-vs-refine-
step trace and reports whether that trace has plateaued. A gap quoted without
that check is a quote of a hyperparameter, not of the model.
"""

import copy
import time

import torch

from .latent import FreeFormLatent, free_form_counterpart, is_amortised


def make_optimizer(model, lr=0.01, lr_encoder=None):
    """Adam, with the encoder in its own parameter group when it has one.

    At a single shared rate the encoder's gradient runs ~7x the GP's and grows,
    the ELBO goes non-monotone, and the depth-1 correctness gate fails. See
    docs/decisions.md D-12 -- and select `lr_encoder` by ELBO, never by gap.

    `lr_encoder` is ignored for FreeFormLatent: a lookup table is not an encoder,
    and giving it a different rate would change the baseline for no reason.
    """
    if lr_encoder is None or not is_amortised(model.latent):
        return torch.optim.Adam(model.parameters(), lr=lr)
    enc = [p for n, p in model.named_parameters() if n.startswith("latent.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("latent.")]
    return torch.optim.Adam([{"params": rest, "lr": lr},
                             {"params": enc, "lr": lr_encoder}])


def train(model, Y, steps=2000, lr=0.01, num_samples=5, log_every=200,
          params=None, seed=None, verbose=True, optimizer=None,
          on_checkpoint=None, ckpt_every=None, start_step=0, hist=None,
          clip_grad_norm=None):
    """Maximise the ELBO. Returns a history dict.

    Pass `optimizer` to CONTINUE an existing trajectory -- Adam's moment
    estimates carry over, so a budget sweep measures one training run observed at
    checkpoints rather than several independent restarts. Omit it for a fresh
    optimiser, which is what the gap refinement wants.

    Resumability (`on_checkpoint`, `ckpt_every`, `start_step`, `hist`): a long run
    that loses power should not start over. `on_checkpoint(step, opt, hist)` is
    called every `ckpt_every` steps and once at the end; `start_step` and `hist`
    restore a previous call's position and history. The caller owns the file
    format -- see scripts/run_experiment.py.
    """
    if seed is not None:
        torch.manual_seed(seed)
    N = Y.shape[0]
    opt = optimizer if optimizer is not None else torch.optim.Adam(
        params if params is not None else model.parameters(), lr=lr)
    hist = hist if hist is not None else {"step": [], "elbo": []}
    hist.setdefault("wall_clock_s", 0.0)
    model.train()
    t0 = time.time()
    for step in range(start_step, steps):
        opt.zero_grad()
        loss = -model.elbo(Y, num_data=N, num_samples=num_samples)
        loss.backward()
        if clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], clip_grad_norm)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            hist["step"].append(step)
            hist["elbo"].append(-loss.item())
            if verbose:
                print(f"  step {step:5d}  ELBO/pt {-loss.item():+.4f}")
        if on_checkpoint is not None and ckpt_every and (step + 1) % ckpt_every == 0:
            hist["wall_clock_s"] += time.time() - t0
            t0 = time.time()
            on_checkpoint(step + 1, opt, hist)
    hist["wall_clock_s"] += time.time() - t0
    hist["optimizer"] = opt
    if on_checkpoint is not None:
        on_checkpoint(steps, opt, hist)
    return hist


@torch.no_grad()
def eval_elbo(model, Y, num_samples=64, seed=0):
    """Low-variance ELBO estimate with a FIXED seed.

    Always evaluate the two schemes with the same seed (common random numbers).
    The gap is a difference of two noisy numbers; shared draws make the noise
    cancel instead of accumulate.
    """
    was_training = model.training
    torch.manual_seed(seed)
    model.eval()
    value = float(model.elbo(Y, num_data=Y.shape[0], num_samples=num_samples))
    model.train(was_training)
    return value


# -- Monte Carlo noise floor -------------------------------------------------

@torch.no_grad()
def elbo_noise_floor(model, Y, num_samples=64, seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    """Spread of the ELBO estimator itself, holding the model fixed.

    Our ELBO is sampled (docs/decisions.md D-01), so every reported number has an
    error bar that has nothing to do with the model. This re-evaluates ONE frozen
    model under several evaluation seeds; the standard deviation is the floor
    below which no ELBO difference is meaningful.

    Report it. The main figure shades it as a band, and a gap inside the band is
    a null result, not a small effect.
    """
    vals = [eval_elbo(model, Y, num_samples=num_samples, seed=s) for s in seeds]
    return {"mean": _mean(vals), "sd": _sd(vals), "values": vals}


@torch.no_grad()
def paired_gap_noise_floor(model_a, model_b, Y, num_samples=64,
                           seeds=(0, 1, 2, 3, 4, 5, 6, 7)):
    """Residual noise in a PAIRED ELBO difference, after common random numbers.

    This -- not `elbo_noise_floor` -- is the floor that belongs on the gap plot.
    Both models are evaluated under the SAME seed, the difference is taken, and
    the spread of those differences across seeds is what survives the CRN
    cancellation. It is typically much smaller than the spread of either ELBO
    alone, which is the whole point of pairing.
    """
    diffs = [eval_elbo(model_b, Y, num_samples=num_samples, seed=s)
             - eval_elbo(model_a, Y, num_samples=num_samples, seed=s) for s in seeds]
    return {"mean": _mean(diffs), "sd": _sd(diffs), "values": diffs}


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# -- the gap -----------------------------------------------------------------

def _freeze_generative_copy(model):
    """Deep-copy the model and freeze every parameter in it."""
    frozen = copy.deepcopy(model)
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen


def _inv_softplus(y):
    return torch.log(torch.expm1(y.clamp_min(1e-6)))


def _free_form_at(latent, Y, n, latent_dim):
    """A per-point latent initialised exactly at `latent`'s current q(Y).

    The counterpart is chosen in the SAME variational family (see
    latent.free_form_counterpart): refining a dense q toward a diagonal table would
    measure the family change on top of the amortisation gap, and refining a
    diagonal q toward a dense table would let the refinement buy expressiveness the
    original scheme never had. Either way the number would stop being an
    amortisation gap.
    """
    cls = free_form_counterpart(latent)
    with torch.no_grad():
        q0 = latent.q(Y=Y)
        mu0 = q0.mean.clone()
    ff = cls(n, latent_dim, X_init=mu0)
    with torch.no_grad():
        if hasattr(ff, "raw_L"):                     # dense: match the full factor
            L = q0.scale_tril.clone()
            raw = L.clone()
            d = torch.diagonal(L, dim1=-2, dim2=-1)
            raw[..., range(latent_dim), range(latent_dim)] = _inv_softplus(d)
            ff.raw_L.copy_(raw)
        else:                                        # diagonal: match sigma exactly
            ff.log_sigma.copy_(_inv_softplus(q0.stddev.clone()))
    for p in ff.parameters():
        p.requires_grad_(True)
    return ff


def measure_amortisation_gap(model, Y, refine_steps=1500, lr=0.01,
                             num_samples=5, eval_samples=64, seed=0,
                             checkpoints=8, plateau_tol=0.01, verbose=True):
    """Freeze the generative model, re-optimise q per point, return the gap.

    The refinement is run in `checkpoints` chunks and the gap is evaluated (at
    the fixed `seed`, so every reading is comparable) after each. That trace is
    the evidence that the quoted gap is a converged quantity rather than an
    artefact of `refine_steps`:

        gap_trace     gap after each chunk, in nats/point
        refine_converged  True if the last chunk moved the gap by less than
                      `plateau_tol` nats/point AND by less than the paired MC
                      noise floor -- i.e. further refinement buys nothing
                      distinguishable from sampling noise

    If `refine_converged` is False the gap is a LOWER bound on the true gap and
    must be reported as such. Say "at least X nats/point after N steps", and
    show the trace.
    """
    N = Y.shape[0]
    elbo_before = eval_elbo(model, Y, num_samples=eval_samples, seed=seed)

    refined = _freeze_generative_copy(model)
    refined.latent = _free_form_at(model.latent, Y, N, model.latent.latent_dim)

    if verbose:
        print(f"  [gap] amortised ELBO/pt = {elbo_before:+.4f}; refining q only...")

    chunk = max(1, refine_steps // max(1, checkpoints))
    hist = {"step": [], "elbo": [], "wall_clock_s": 0.0}
    trace_steps, gap_trace = [], []
    done = 0
    while done < refine_steps:
        this = min(chunk, refine_steps - done)
        h = train(refined, Y, steps=this, lr=lr, num_samples=num_samples,
                  params=refined.latent.parameters(), verbose=False,
                  log_every=max(1, this // 2))
        done += this
        hist["step"] += [s + (done - this) for s in h["step"]]
        hist["elbo"] += h["elbo"]
        hist["wall_clock_s"] += h["wall_clock_s"]
        g = eval_elbo(refined, Y, num_samples=eval_samples, seed=seed) - elbo_before
        trace_steps.append(done)
        gap_trace.append(g)
        if verbose:
            print(f"  [gap]   after {done:5d} refine steps: gap = {g:+.4f}")

    elbo_after = eval_elbo(refined, Y, num_samples=eval_samples, seed=seed)
    gap = elbo_after - elbo_before

    # residual noise in this very difference, under common random numbers
    noise = paired_gap_noise_floor(model, refined, Y, num_samples=eval_samples)
    last_move = abs(gap_trace[-1] - gap_trace[-2]) if len(gap_trace) > 1 else float("inf")
    # Converged when a further chunk of refinement would move the gap by less than
    # we can measure. The MC noise floor is the real criterion -- you cannot resolve
    # a change smaller than your own estimator's spread -- and `plateau_tol` is only
    # a floor on that tolerance, for when the noise floor is tiny.
    #
    # This must NOT be a fixed absolute threshold: the ELBO scale is dataset
    # dependent. Synthetic sits near -7 nats/point with a noise floor of ~0.01; Frey
    # (PCA-100) sits near -150 with a floor of ~0.15. A fixed 0.01 would mark every
    # Frey refinement "not converged" at 0.007% of the bound, needlessly downgrading
    # every gap we report to a lower bound.
    tol = max(noise["sd"], plateau_tol)
    converged = last_move < tol

    if verbose:
        print(f"  [gap] refined ELBO/pt   = {elbo_after:+.4f}  ->  GAP = {gap:+.4f} nats/point")
        print(f"  [gap] paired MC noise floor = {noise['sd']:.4f} nats/point"
              f"  |  last chunk moved the gap {last_move:+.4f}")
    if gap < -1e-3:
        print("  [gap] WARNING: negative gap. Refinement under-converged or the "
              "decoder was not properly frozen. Increase refine_steps.")
    if not converged:
        print(f"  [gap] WARNING: refinement has NOT plateaued (last chunk moved "
              f"{last_move:+.4f} nats/pt, tolerance {tol:.4f}). The gap is a LOWER "
              f"BOUND -- report it as such, or raise --refine-steps.")
    if abs(gap) < noise["sd"]:
        print(f"  [gap] NOTE: |gap| ({abs(gap):.4f}) is below the paired MC noise "
              f"floor ({noise['sd']:.4f}). This is a null reading, not a small effect.")

    return {"elbo_amortised": elbo_before, "elbo_refined": elbo_after,
            "gap": gap, "gap_noise_sd": noise["sd"], "gap_trace": gap_trace,
            "gap_trace_steps": trace_steps, "refine_converged": bool(converged),
            "refine_last_move": last_move, "refine_tol": tol,
            "refine_history": hist}


# -- held-out likelihood -----------------------------------------------------

@torch.no_grad()
def eval_predictive(model, Y, num_samples=64, seed=0):
    """Fixed-seed held-out predictive bound. See DeepGPLVM.predictive_elbo."""
    was_training = model.training
    torch.manual_seed(seed)
    model.eval()
    value = float(model.predictive_elbo(Y, num_samples=num_samples))
    model.train(was_training)
    return value


def heldout_elbo(model, Y_test, steps=500, lr=0.01, num_samples=5,
                 eval_samples=64, seed=0, X_init=None, verbose=True):
    """Per-point test ELBO -- a lower bound on log p(y*) for unseen points.

    A GP-LVM cannot score a new point without a q(X*) for it, and the two schemes
    differ in exactly how they get one. That difference IS the practical stake of
    the project, so the protocol mirrors deployment rather than equalising it:

      amortised  -- the encoder maps y* to (mu*, sigma*) in ONE forward pass.
                    `embed_steps` is 0. This is amortisation's selling point.
      free-form  -- there is no mapping, so a fresh q over the test points must
                    be optimised against the FROZEN generative model. O(N_test)
                    parameters and a training loop just to score a point.

    Generative parameters are frozen in both cases, so the number reflects the
    model that was learned, not fresh fitting to the test set. Returns the test
    ELBO in nats/point, plus the embedding cost that produced it.
    """
    scored = _freeze_generative_copy(model)
    t0 = time.time()
    refined_value = None

    if is_amortised(model.latent):              # encoder: generalises for free
        embed_steps = 0
        if verbose:
            print("  [test] amortised: embedding test points with the encoder (1 pass)")
    else:                                       # FreeFormLatent: must re-optimise
        embed_steps = steps
        # Init at the TRAINING PCA basis applied to the test points, mirroring how
        # the training latents were initialised. A different basis here would make
        # the test embedding start somewhere the decoder has never seen.
        ff = FreeFormLatent(Y_test.shape[0], model.latent.latent_dim, X_init=X_init)
        for p in ff.parameters():
            p.requires_grad_(True)
        scored.latent = ff
        if verbose:
            print(f"  [test] free-form: no encoder, optimising q(X*) for "
                  f"{Y_test.shape[0]} test points ({steps} steps)")
        train(scored, Y_test, steps=steps, lr=lr, num_samples=num_samples,
              params=ff.parameters(), verbose=False, log_every=max(1, steps // 2))
        if verbose:
            print(f"  [test] free-form embedding cost: {steps} optimisation steps "
                  f"vs 1 forward pass for the encoder -- this asymmetry is a result")

    value = eval_predictive(scored, Y_test, num_samples=eval_samples, seed=seed)
    cost = time.time() - t0
    if verbose:
        print(f"  [test] held-out ELBO/pt = {value:+.4f}  ({cost:.1f}s, "
              f"{embed_steps} embedding steps)")

    # LIKE-FOR-LIKE ARM. Scoring each scheme as it deploys (above) is the honest
    # deployment comparison, but it is NOT a generalisation comparison: the
    # free-form arm optimises q(X*) against the test points for `steps` steps while
    # the encoder gets one forward pass and no fitting. Any held-out advantage for
    # free-form is therefore partly just that it was allowed to fit and the encoder
    # was not.
    #
    # So also score the amortised model WITH the same test-time refinement, started
    # at the encoder's own output. Then:
    #   free_form.heldout_elbo  vs  amortised.heldout_refined   -> like-for-like
    #   heldout_refined - heldout_elbo                          -> the amortisation
    #                                                              gap on UNSEEN data
    # That second quantity is the one the research question is really about, since
    # a shared mapping's whole purpose is to generalise to new points.
    if is_amortised(model.latent):
        r = _freeze_generative_copy(model)
        r.latent = _free_form_at(model.latent, Y_test, Y_test.shape[0],
                                 model.latent.latent_dim)
        train(r, Y_test, steps=steps, lr=lr, num_samples=num_samples,
              params=r.latent.parameters(), verbose=False, log_every=max(1, steps // 2))
        refined_value = eval_predictive(r, Y_test, num_samples=eval_samples, seed=seed)
        if verbose:
            print(f"  [test] held-out ELBO/pt with test-time refinement = "
                  f"{refined_value:+.4f}  -> held-out gap {refined_value - value:+.4f}")

    out = {"heldout_elbo": value, "heldout_embed_steps": embed_steps,
           "heldout_embed_s": cost}
    if refined_value is not None:
        out["heldout_elbo_refined"] = refined_value
        out["heldout_gap"] = refined_value - value
    return out


# -- posterior diagnostics ---------------------------------------------------

@torch.no_grad()
def latent_summary(model, Y, colour=None):
    """q(X) means and spreads, for the latent scatter and the collapse check.

    `q_sigma_mean` near zero means the posterior has collapsed to a point
    estimate; near 1 means it has collapsed to the prior and the latent carries
    no information. Both are pathologies the interpretation slides must address.
    """
    q = model.latent.q(Y=Y)
    mu, sigma = q.mean, q.stddev
    out = {"latent_mu": mu.cpu().tolist(),
           "q_sigma_mean": float(sigma.mean()),
           "q_sigma_per_dim": sigma.mean(0).cpu().tolist(),
           "mu_std_per_dim": mu.std(0).cpu().tolist()}
    if colour is not None:
        out["latent_colour"] = [float(c) for c in colour]
    return out
