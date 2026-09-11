#!/usr/bin/env python3
"""Cross-check every number quoted in the report, deck and script against results/.

A report, a deck and a speaking script drift apart the moment one is edited alone.
This recomputes the ground truth from the results JSON and asserts each claim
appears, correctly, in each document that should carry it.

Usage:  python scripts/verify_documents.py
"""
import json, os, re, sys

R = "results"
DOCS = {
    "report":  ["report/sections/%s" % f for f in os.listdir("report/sections")],
    "deck":    ["scripts/build_slides.py"],
    "script":  ["slides/CSE756_Presentation_II_Script.md"],
}
def _norm(t):
    """Normalise so formatting differences are not mistaken for content errors:
    LaTeX thin-space digits (1{,}200), Unicode minus/dashes, and line wrapping."""
    t = t.replace("{,}", ",").replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[>\s]+", " ", t)          # '>' strips the markdown quote marker
    return t

TEXT = {k: _norm("\n".join(open(f).read() for f in v)) for k, v in DOCS.items()}


def ms(v):
    if not v:
        return float("nan"), float("nan"), float("nan")
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd, sd / max(1, len(v)) ** 0.5


def load(name):
    p = f"{R}/{name}"
    return json.load(open(p))["rows"] if os.path.exists(p) else None


results, checks = [], []


def claim(label, value, fmt, where, spoken=None):
    """Assert `value` (formatted) appears in each document named in `where`."""
    s = fmt.format(value)
    for doc in where:
        hay = TEXT[doc]
        ok = s in hay
        if not ok and doc == "script" and spoken:
            ok = spoken.lower() in hay.lower()
        checks.append((label, doc, s if not (doc == "script" and spoken) else f"{s} / '{spoken}'", ok))


# ---------------- ground truth: synthetic, 8 seeds, 16k ----------------
syn = load("slack_study.json")
g1 = [r["gap"] for r in syn if r["depth"] == 1 and r["scheme"] == "amortised" and r["steps"] == 16000]
c1 = [r["gap"] for r in syn if r["depth"] == 1 and r["scheme"] == "free_form" and r["steps"] == 16000]
g2 = [r["gap"] for r in syn if r["depth"] == 2 and r["scheme"] == "amortised" and r["steps"] == 16000]
c2 = [r["gap"] for r in syn if r["depth"] == 2 and r["scheme"] == "free_form" and r["steps"] == 16000]
m1, s1, e1 = ms(g1); mc1, sc1, _ = ms(c1)
m2, s2, e2 = ms(g2); mc2, sc2, _ = ms(c2)
dse = (e1 ** 2 + e2 ** 2) ** 0.5
print("=" * 74)
print("GROUND TRUTH (recomputed from results/)")
print("=" * 74)
print(f"  synthetic n={len(g1)} seeds @16k")
print(f"    d1 gap {m1:.4f}±{s1:.4f}  control {mc1:.4f}±{sc1:.4f}  net {m1-mc1:+.4f}")
print(f"    d2 gap {m2:.4f}±{s2:.4f}  control {mc2:.4f}±{sc2:.4f}  net {m2-mc2:+.4f}")
print(f"    depth effect {m2-m1:+.4f} ± {dse:.4f} SE = {abs(m2-m1)/dse:.2f} SE")

claim("d1 gap",     m1,  "{:.3f}", ["report", "deck", "script"], spoken="zero point zero-four-one")
claim("d1 control", mc1, "{:.3f}", ["report", "deck"])
claim("d2 gap",     m2,  "{:.3f}", ["report", "deck", "script"], spoken="zero point zero-five-three")
claim("d2 control", mc2, "{:.3f}", ["report", "deck"])
claim("depth effect", m2 - m1, "{:.3f}", ["report", "deck"])
claim("depth-effect SE", dse, "{:.3f}", ["report", "deck"])

nonmono = 0
for d in (1, 2):
    for sch in ("free_form", "amortised"):
        for sd in sorted({r["seed"] for r in syn}):
            v = [next(r["elbo"] for r in syn if r["depth"] == d and r["seed"] == sd
                      and r["scheme"] == sch and r["steps"] == st)
                 for st in (1000, 2000, 4000, 8000, 16000)]
            if any(v[i+1] < v[i] - 0.5 for i in range(4)): nonmono += 1
print(f"    non-monotone trajectories: {nonmono}/32")
results.append(("synthetic trajectories all monotone", nonmono == 0, f"{nonmono}/32"))

sel = {}
for sch in ("free_form", "amortised"):
    p = [2 if next(r["elbo"] for r in syn if r["depth"] == 2 and r["seed"] == sd and r["scheme"] == sch and r["steps"] == 16000)
         > next(r["elbo"] for r in syn if r["depth"] == 1 and r["seed"] == sd and r["scheme"] == sch and r["steps"] == 16000)
         else 1 for sd in range(8)]
    sel[sch] = p.count(2)
print(f"    depth-2 preferred: free-form {sel['free_form']}/8, amortised {sel['amortised']}/8")
results.append(("synthetic depth selection is 4/8 vs 4/8 as documented",
                sel["free_form"] == 4 and sel["amortised"] == 4,
                f"{sel['free_form']}/8 vs {sel['amortised']}/8"))

# ---------------- ground truth: Frey, 4 seeds ----------------
frey = load("frey.json")
print(f"\n  frey n=4 seeds, {len(frey)}/16 cells")
fr = {}
for d in (1, 2):
    am = [r for r in frey if r["depth"] == d and r["scheme"] == "amortised"]
    ff = [r for r in frey if r["depth"] == d and r["scheme"] == "free_form"]
    fr[d] = dict(gap=ms([r["gap"] for r in am]), ctrl=ms([r["control_gap"] for r in ff]),
                 ho=ms([r["heldout_gap"] for r in am]),
                 ff_elbo=ms([r["elbo"] for r in ff]), am_elbo=ms([r["elbo"] for r in am]),
                 ff_ho=ms([r["heldout_elbo"] for r in ff]),
                 am_ho=ms([r["heldout_elbo"] for r in am]),
                 am_hor=ms([r["heldout_elbo_refined"] for r in am]))
    x = fr[d]
    print(f"    d{d} train gap {x['gap'][0]:+.4f}  ctrl {x['ctrl'][0]:+.4f}  net {x['gap'][0]-x['ctrl'][0]:+.4f}"
          f" | HELD-OUT {x['ho'][0]:+.2f}±{x['ho'][1]:.2f}")
ho_se = (fr[1]["ho"][2] ** 2 + fr[2]["ho"][2] ** 2) ** 0.5
ho_drop = fr[1]["ho"][0] - fr[2]["ho"][0]
ratio = fr[1]["ho"][0] / fr[2]["ho"][0]
tr_ratio = fr[1]["ho"][0] / fr[1]["gap"][0]
print(f"    held-out depth effect {-ho_drop:+.2f} ± {ho_se:.2f} SE = {ho_drop/ho_se:.1f} SE | {ratio:.0f}x drop")
print(f"    d1 held-out / d1 train = {tr_ratio:.0f}x")

claim("frey d1 held-out gap", fr[1]["ho"][0], "{:.2f}", ["report", "deck", "script"],
      spoken="twenty-nine point nine")
claim("frey d2 held-out gap", fr[2]["ho"][0], "{:.2f}", ["report", "deck", "script"],
      spoken="zero point four-eight")
claim("frey d1 train gap", fr[1]["gap"][0], "{:.4f}", ["report"])
claim("held-out SE multiple", ho_drop / ho_se, "{:.1f}", ["report", "deck", "script"],
      spoken="eight and a half standard errors")

for lbl, val, ok_names in [("62x collapse", ratio, ["report", "deck", "script"]),
                           ("1200x train-vs-test", tr_ratio, ["report", "deck", "script"])]:
    n = f"{val:.0f}"
    for doc in ok_names:
        hay = TEXT[doc]
        found = (n in hay) or (f"{int(round(val,-2)):,}" in hay) or (f"{int(round(val,-2))}" in hay)
        if doc == "script" and not found:
            found = ("sixty-two" in hay.lower()) or ("twelve\n> hundred" in hay) or ("twelve hundred" in hay.lower())
        checks.append((lbl, doc, n, found))

selF = {}
for sch in ("free_form", "amortised"):
    p = [2 if next(r["elbo"] for r in frey if r["depth"] == 2 and r["seed"] == sd and r["scheme"] == sch)
         > next(r["elbo"] for r in frey if r["depth"] == 1 and r["seed"] == sd and r["scheme"] == sch)
         else 1 for sd in range(4)]
    selF[sch] = p.count(2)
results.append(("frey depth selection unanimous for depth 1",
                selF["free_form"] == 0 and selF["amortised"] == 0,
                f"depth2 chosen {selF['free_form']}/4 and {selF['amortised']}/4"))
results.append(("all frey refinements converged",
                all(r.get("refine_converged", True) for r in frey if "gap" in r) and
                all(r.get("control_refine_converged", True) for r in frey if "control_gap" in r), ""))

# ---------------- encoder LR ablation ----------------
enc = load("encoder_lr_d1.json")
cfgs = {}
for r in enc: cfgs.setdefault(r["config"].split("  ")[0], []).append(r)
print("\n  encoder-lr ablation")
for k, v in cfgs.items():
    e = ms([r["elbo"] for r in v]); g = ms([r["gap"] for r in v])
    bad = sum(1 for r in v if r["non_monotone"])
    print(f"    {k:26s} ELBO {e[0]:+.3f}  gap {g[0]:.3f}±{g[1]:.3f}  unstable {bad}/{len(v)}")
    if "0.01 " in k or k.endswith("0.01"):
        claim("enc lr 1e-2 gap", g[0], "{:.3f}", ["report", "deck"])
    if k.endswith("0.001"):
        claim("enc lr 1e-3 gap", g[0], "{:.3f}", ["report", "deck"])
        claim("enc lr 1e-3 ELBO", e[0], "{:.3f}", ["report", "deck"])

# ---------------- short-budget artifact ----------------
art = load("artifact_shortbudget.json")
print("\n  short-budget artifact (the Table-1 claim)")
if art is None:
    results.append(("300-step artifact numbers are reproducible from results/", False,
                    "results/artifact_shortbudget.json MISSING — regenerate"))
else:
    want = {(d, sc, st) for d in (1, 2) for sc in ("free_form", "amortised") for st in (300, 1000)}
    have = {(r["depth"], r["scheme"], r["steps"]) for r in art}
    if not want <= have:
        results.append(("300-step artifact run complete", False,
                        f"still running: {len(have)}/{len(want)} (depth,scheme,step) combos"))
    else:
        for st in (300, 1000):
            for d in (1, 2):
                gg = ms([r["gap"] for r in art if r["depth"] == d and r["scheme"] == "amortised" and r["steps"] == st])
                cc = ms([r["gap"] for r in art if r["depth"] == d and r["scheme"] == "free_form" and r["steps"] == st])
                print(f"    {st:5d} steps depth {d}: gap {gg[0]:+.3f}  control {cc[0]:+.3f}")
        c1_ = ms([r["gap"] for r in art if r["depth"] == 1 and r["scheme"] == "free_form" and r["steps"] == 300])[0]
        c2_ = ms([r["gap"] for r in art if r["depth"] == 2 and r["scheme"] == "free_form" and r["steps"] == 300])[0]
        results.append(("300-step control grows with depth (the artifact)", c2_ > c1_,
                        f"d1 {c1_:.3f} -> d2 {c2_:.3f}"))
        claim("artifact control d1@300", c1_, "{:.2f}", ["report", "deck", "script"],
              spoken="zero point eight-three")
        claim("artifact control d2@300", c2_, "{:.2f}", ["report", "deck", "script"],
              spoken="two point four-five")

# ---------------- MC noise floor, the claim that makes the artifact look significant ----
nf = load("mc_noise_floor.json")
if nf is None:
    results.append(("MC noise-floor claim is sourced", False,
                    "results/mc_noise_floor.json MISSING"))
else:
    sds = [r["paired_gap_noise_sd"] for r in nf]
    print(f"\n  MC noise floor (paired): {min(sds):.4f} - {max(sds):.4f} over "
          f"{len(nf)} conditions")
    lo, hi = min(sds), max(sds)
    for doc in ("report", "deck", "script"):
        hay = TEXT[doc]
        ok = (f"{lo:.3f}" in hay and f"{hi:.3f}" in hay) or \
             ("zero-zero-five" in hay and "zero-zero-nine" in hay)
        checks.append(("MC noise floor range", doc, f"{lo:.3f}-{hi:.3f}", ok))
    results.append(("MC noise-floor claim is sourced to results/mc_noise_floor.json",
                    True, f"measured {lo:.4f}-{hi:.4f} over {len(nf)} conditions"))

# ---------------- posterior spread + latent pruning (slide 9/10 claims) --------
sp1=[r["q_sigma_mean"] for r in frey if r["depth"]==1]
sp2=[r["q_sigma_mean"] for r in frey if r["depth"]==2]
def active(r):
    return sum(1 for a,b in zip(r["q_sigma_per_dim"], r["mu_std_per_dim"]) if a<0.9 and b>0.1)
a1=[active(r) for r in frey if r["depth"]==1]; a2=[active(r) for r in frey if r["depth"]==2]
print(f"\n  posterior spread: d1 {min(sp1):.2f}-{max(sp1):.2f}  d2 {min(sp2):.2f}-{max(sp2):.2f}")
print(f"  active latent dims: d1 {min(a1)}-{max(a1)}/5   d2 {min(a2)}-{max(a2)}/5")
for doc in ("deck","script"):
    hay=TEXT[doc]
    if doc == "deck":            # the slide carries the numbers
        ok = "0.09" in hay and "0.71" in hay and "0.96" in hay
    else:                         # the script describes them, it does not read them aloud
        ok = "back to the prior" in hay.lower() and "five latent" in hay.lower()
    checks.append(("posterior-spread claim matches d1/d2", doc, "d1 0.09-0.11, d2 0.71-0.96", ok))
results.append(("no document claims spread stays in 0.10-0.65",
                "0.10–0.65" not in TEXT["deck"] and "0.10-0.65" not in TEXT["deck"], ""))
results.append(("Frey depth preference stated as 8 runs, not 4",
                "all four runs" not in TEXT["deck"].lower(), ""))

# ---------------- absolute claims that the data does not support --------------
floor = max(r["paired_gap_noise_sd"] for r in nf) if nf else 0.0
syn_below = sum(1 for d in (1,2) for r in syn
                if r["depth"]==d and r["scheme"]=="amortised" and r["steps"]==16000
                and abs(r["gap"]) < floor)
frey_below = sum(1 for r in frey if "gap" in r and abs(r["gap"]) <= r["gap_noise_sd"])
print(f"  per-seed gaps below the MC floor: synthetic {syn_below}/16, frey {frey_below}/8")
for doc in ("report","deck","script"):
    over = ("every gap clears" in TEXT[doc].lower()) and syn_below > 0
    results.append((f"{doc}: no overstated 'every gap clears the floor' claim", not over,
                    f"{syn_below}/16 synthetic seeds are below the floor"))
results.append(("frey gaps all clear their own recorded floor", frey_below == 0,
                f"{frey_below}/8 below"))
w = sorted(r["wall_clock_s"]/60 for r in frey)
med = w[len(w)//2]
results.append(("stated Frey cell time matches the median, not a resume-inflated mean",
                "30 min median" in TEXT["deck"] or f"{med:.0f} min" in TEXT["deck"],
                f"median {med:.0f} min, mean {sum(w)/len(w):.0f}"))

# ---------------- slide 9 and slide 13 must tell the same story ----------------
# The held-out result is explained by latent pruning (r=+0.95). If any document still
# explains it by "overfitting", slide 13's open question contradicts slide 9.
for doc in ("deck", "script"):
    hay = TEXT[doc].lower()
    explains_by_pruning = "active" in hay and "0.95" in hay.replace("zero point nine-five", "0.95")
    stale_overfit = "gap follows overfitting" in hay or "gap appears to track overfitting" in hay
    results.append((f"{doc}: held-out result explained by pruning, not the old overfitting story",
                    explains_by_pruning and not stale_overfit, ""))

# ---------------- no stale "still running" claims ----------------
# All experiment files are complete, so no document should say work is in progress.
STALE = ["running now", "finishing now", "in progress", "is still running",
         "cells are finishing", "currently running"]
for doc in ("report", "deck", "script"):
    hits = [w for w in STALE if w in TEXT[doc].lower()]
    results.append((f"{doc}: no stale 'work in progress' claims", not hits, f"{hits}"))

# ---------------- report: structural requirements ----------------
rep = TEXT["report"]
req = ["Abstract", "Introduction", "Background and Related Work", "Probabilistic Model",
       "Methodology", "Experimental Design", "Results", "Discussion", "Conclusion",
       "Team Contribution Statement"]
missing = [s for s in req if s.lower() not in rep.lower()]
results.append(("report has all required sections", not missing, f"missing: {missing}"))
pend = re.findall(r"\\pending\{([^}]*)\}", rep)
awaiting_user = [x for x in pend if "name" in x.lower()]
stale = [x for x in pend if x not in awaiting_user]
results.append(("report has no PENDING results markers", not stale, f"{stale}"))
results.append((f"only intentional placeholders remain ({len(awaiting_user)}: team names)",
                True, ""))

# ---------------- print ----------------
print("\n" + "=" * 74)
print("CROSS-DOCUMENT NUMBER CHECK")
print("=" * 74)
fails = [c for c in checks if not c[3]]
by = {}
for lbl, doc, val, ok in checks: by.setdefault(lbl, []).append((doc, ok, val))
for lbl, v in by.items():
    st = "OK  " if all(o for _, o, _ in v) else "MISS"
    print(f"  [{st}] {lbl:24s} " + "  ".join(f"{d}:{'y' if o else 'N'}" for d, o, _ in v)
          + ("" if all(o for _, o, _ in v) else f"   expected {v[0][2]}"))
print("\n" + "=" * 74)
print("CONSISTENCY ASSERTIONS")
print("=" * 74)
for lbl, ok, detail in results:
    print(f"  [{'OK  ' if ok else 'FAIL'}] {lbl}" + (f"   ({detail})" if detail and not ok else
                                                    (f"   ({detail})" if detail else "")))
nf = len(fails) + sum(1 for _, ok, _ in results if not ok)
print(f"\n{'ALL CHECKS PASS' if nf == 0 else f'{nf} ISSUE(S)'}")
sys.exit(1 if nf else 0)
