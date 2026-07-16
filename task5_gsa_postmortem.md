# Why our sensitivity analysis said "T is inert" — and why that was wrong

**Status:** corrected in `task5_sensitivity.py`. Figure and tables regenerated.
**Trigger:** Janis ran an independent GSA with real pilot API runs and got a
contradictory answer (T dominant). He was right; our analysis was wrong.

---

## 1. Summary

Our previous task-5 analysis reported three conclusions. **All three were
artefacts of the experimental design, not properties of the bioprocess.**

| Previous claim | Reality |
|---|---|
| "F3 dominates: `ST ≈ 1.00`, explains ~all variance" | True *only* of the pilot slice, which is 3 data points |
| "T is inert, `ST = 0.000`" | T swings mean yield **3.5×** in the raw data; it peaks near 37 °C |
| "No interaction is distinguishable from zero; the response is close to additive" | `F1*F3` and `pH*F3` have CIs excluding 0; `Σ(ST−S1) = 1.53` at micro |

The root cause is a single sentence:

> **We ran the Sobol analysis on the GP's pilot slice, and only 3 of our 187
> runs are pilot runs — all three sitting on top of the optimum the BO loop had
> already converged to.**

A variance-based method cannot attribute variance to an input that does not
vary. Our pilot design spans T = 34.6–40 °C out of an admissible 20–60 °C, and
pH = 8.07–8.57 out of 3–9.5. Sobol correctly reported that, *within the region
we sampled*, T explains nothing. We then read that as "T does not matter."

**`ST = 0.000` meant "we never varied T." It did not mean "T does nothing."**

There is a bitter irony here. The BO loop converged onto T ≈ 37 °C and pH ≈ 8.2
*because they are optimal*. Converging there is the loop succeeding — and it is
exactly what destroyed the variance the GSA needed in order to notice. **The
optimizer erased the evidence for its own best decision.**

---

## 2. The evidence

### 2.1 The design never moved T at pilot

All three pilot runs, in real units:

| T (°C) | pH | F3 | Y |
|---|---|---|---|
| 34.6 | 8.17 | 0.82 | 8.81 |
| 40.0 | 8.57 | 0.13 | 1.24 |
| 36.0 | 8.07 | 0.99 | 16.42 |

F3 is the only input that meaningfully moves, and Y tracks it monotonically
across all three points. A GP fitted to this *must* produce "Y = f(F3)". The
pilot Sobol result was never a measurement of the process; it was a description
of three points.

Design coverage at pilot (`task5_design_support.csv`):

| variable | design_std | pilot range | admissible | identifiable |
|---|---|---|---|---|
| T | 0.070 | 34.6 – 40.0 | 20 – 60 | **no** |
| pH | 0.041 | 8.07 – 8.57 | 3 – 9.5 | **no** |
| F1 | 0.080 | 1.43 – 1.75 | 0 – 2 | **no** |
| F2 | 0.198 | 0.94 – 1.71 | 0 – 2 | marginal |
| F3 | 0.454 | 0.27 – 1.99 | 0 – 2 | yes |

(A uniform design on [0,1] has std ≈ 0.289. Four of five inputs are nowhere
near it.)

### 2.2 The same GP knows T matters — at the scale that has data

The surrogate is not broken. Ask it for the T profile at the **micro** slice
(173 runs, full 20–60 °C sweep), holding pH = 8.2 and F3 = 2.0:

```
T =  20 C   Y_hat = 0.091
T =  28 C   Y_hat = 0.444
T =  36 C   Y_hat = 0.632   <-- peak at ~38 C
T =  44 C   Y_hat = 0.566
T =  60 C   Y_hat = -0.022
```

Total-effect index by fidelity (`task5_sobol_first_total.csv`):

| variable | micro (n=173) | bench (n=11) | pilot (n=3) |
|---|---|---|---|
| F3 | 0.675 | 0.835 | 1.003 |
| pH | 0.466 | 0.370 | 0.019 |
| F2 | 0.371 | 0.028 | 0.005 |
| T | **0.197** | 0.004 | **0.000** |
| F1 | 0.177 | 0.001 | 0.000 |

T's index does not fall from 0.197 to 0.000 because the physics changes between
a microplate and a pilot reactor. It falls because the *design* changes.

### 2.3 The raw data, with no model at all

Binned mean Y across quantile bins of each input, micro scale, n=173
(`task5_raw_effect_screen.csv`):

| variable | fold change | Spearman ρ |
|---|---|---|
| pH | **4.08×** | 0.36 |
| T | **3.51×** | **−0.12** |
| F3 | 2.69× | 0.57 |
| F1 | 2.46× | 0.49 |
| F2 | 2.15× | 0.24 |

Note T's row. Rank correlation is **−0.12** — indistinguishable from noise —
while the binned means swing 3.5×. **This is why the effect hid.** T's response
is a bump peaking around 37 °C, and a bump has no monotonic trend, so every
rank-correlation screen we ran declared T uninteresting. Spearman cannot see a
hill; it only sees slopes.

### 2.4 T's effect is not F3 in disguise

The obvious objection is confounding: the loop may have set T ≈ 37 and F3 ≈ 2
together. It did not. Restricting to the F3-saturated runs only (F3 > 1.75,
n = 105), the T bump survives intact:

| T (°C) | mean Y |
|---|---|
| 20–30 | 0.200 |
| 30–35 | 0.245 |
| **35–40** | **0.696** |
| 40–45 | 0.214 |
| 45–60 | 0.212 |

### 2.5 The response is strongly interactive

Conditioning the other way exposes the interaction we had denied:

| F3 (g/L/h) | mean Y, T ∈ 35–40 °C | mean Y, T outside |
|---|---|---|
| 0.0–1.0 | 0.097 | 0.131 |
| 1.0–1.5 | 0.461 | 0.191 |
| 1.5–1.75 | 0.512 | 0.250 |
| 1.75–2.0 | **0.696** | 0.214 |

**F3 buys a ~7× gain when T is right and ~1.6× when it isn't.** Feeding a
reactor held at the wrong temperature accomplishes little — which is
unsurprising biology, and precisely what "the response is additive" denied.

At the micro slice, `F1*F3` (S2 = 0.110 ± 0.070) and `pH*F3` (S2 = 0.104 ±
0.093) have bootstrap CIs excluding zero, and `Σ(ST − S1) = 1.53` — most of the
response lives in interactions. The old "additive" claim came from the pilot
slice, where **three points cannot express an interaction even in principle.**

---

## 3. Why our safeguards didn't catch it

This is the uncomfortable part. The previous script was *not* naive — it carried
several caveats, and they still didn't save us:

1. **It printed the right warning and we under-read it.** The old output
   already said T, pH, F1 "barely vary across the pilot runs... extrapolated,
   not measured." That warning was correct. But it was printed *below* a
   confident ranking, framed as a caveat to a finding rather than as grounds for
   refusing to report the finding. **A caveat under a headline loses to the
   headline.** The ranking is now gated: variables the design never moved are
   marked unidentifiable in the table and hatched in the figure.

2. **We checked collinearity but never checked coverage.** The old script
   tested whether inputs were correlated *with each other* (finding F1–F3,
   ρ = 0.56) — a real problem, correctly handled with grouped indices. But
   correlation between inputs is a different failure from an input having no
   range at all. We tested for the sophisticated problem and missed the basic
   one.

3. **Every correlation screen we used was monotonic.** Spearman on T gives
   −0.12. Nothing in the pipeline could see a unimodal driver. The new script
   runs a binned, model-free screen precisely because it assumes no shape.

4. **The result was plausible.** F3 dominating was consistent with the loop's
   behaviour (it did saturate F3), so it read as confirmation. A wrong answer
   that matches your expectations is the hardest kind to catch — which is
   exactly why Janis's contradictory result was valuable rather than annoying.

---

## 4. What changed in `task5_sensitivity.py`

- **Runs the GSA at every fidelity** (micro / bench / pilot) and reports them
  side by side, instead of pilot only.
- **Added `design_support()`** — per-scale spread of each input, with an
  `identifiable` flag (`design_std ≥ 0.15`). Indices for unsupported inputs are
  reported as extrapolation, not findings.
- **Added `raw_effect_screen()`** — model-free binned-mean sweep of the raw
  data, reporting fold-change alongside Spearman ρ so the gap between them is
  visible on the page.
- **Added an automatic contradiction gate.** If the pilot GSA calls a variable
  inert (`ST < 0.05`) while the raw data swings ≥ 2×, the script prints
  `*** THE PILOT-SLICE RANKING IS NOT A FINDING ***` and names the cause. It
  now fires for T, pH, F1 and F2 — i.e. the script diagnoses this exact bug
  automatically, and will do so again if it recurs.
- **Reports the defensible ranking** (micro slice, all inputs fully swept) and
  the interactions found there, instead of the pilot artefact.
- **Figure rebuilt**: ST per variable per scale with unsupported bars hatched;
  the GP's T profile with the pilot runs' coverage shaded; and the raw-data
  T×F3 interaction.

New outputs: `task5_design_support.csv`, `task5_raw_effect_screen.csv`.
Existing tables now carry a `scale` column.

---

## 5. What we can and cannot now claim

**Defensible:** at the micro scale, where all five inputs were swept across
their full ranges in 173 runs, the ranking by total effect is
**F3 (0.68) > pH (0.47) > F2 (0.37) > T (0.20) > F1 (0.18)**, and the response
is strongly interactive. By raw swing, pH (4.1×) and T (3.5×) actually exceed
F3 (2.7×) — **Janis's read that T and pH are as important as F3, if not more,
is supported.** The two measures disagree on the exact order because both are
marginal and mutually confounded (F3's marginal effect is diluted by runs where
T is wrong, and vice versa); what is *not* in doubt is that T and pH are major
drivers rather than inert.

**Not defensible, in either direction:** any per-variable ranking *at the pilot
scale*. We do not have the data. The micro ranking is "what the cheap scale
says"; transporting it to pilot assumes the scale bias doesn't reorder the
drivers, and the whole premise of this project is that the scales differ.
F1's apparent effect is also entangled with F3 (ρ = 0.56) and is not separately
identifiable.

---

## 6. Recommended next experiment

This is now a concrete, cheap ask — and it is **the one experiment the campaign
never performed**:

> Sweep T across 20–60 °C at **pilot**, holding pH ≈ 8.2, F1 ≈ 1.6, F2 ≈ 1.3,
> F3 ≈ 2.0 (the converged optimum). Then the same for pH across 3–9.5.

We are testing a 3.5× effect, so a handful of runs settles it. At 2000 EUR per
pilot run, ~6 runs ≈ 12 000 EUR — so this must be weighed against remaining
budget, but it is the only way to make the pilot ranking identifiable.

---

## 7. Note on Janis's numbers

His **conclusion was right and ours was wrong** — the direction of his result is
what prompted this fix, and it is confirmed above by data we already had.

His *reported indices*, however, still can't be read as-is. For independent
inputs Sobol guarantees `ST ≥ S1`, but 4 of his 5 rows violate it, and
`Σ S1 = 0.415 > Σ ST = 0.168`, which is impossible:

| variable | S1 (reported) | ST (reported) | ST ≥ S1? |
|---|---|---|---|
| T | 0.3012 | 0.0889 | **violated** |
| pH | 0.0275 | 0.0574 | ok |
| F1 | 0.0060 | 0.0012 | **violated** |
| F2 | 0.0077 | 0.0060 | **violated** |
| F3 | 0.0725 | 0.0140 | **violated** |

**Most likely cause: the S1 and ST columns are swapped in the print statement.**
Swapping them fixes 4 of the 5 violations and yields T dominant at ST = 0.30 —
consistent with everything in this document. Worth checking that line first.

Two further notes:

- **We tested and ruled out the obvious explanation.** Sampling a known ground
  truth at his N=16 with realistic pilot noise (σ ≈ 0.15), 10 repeats: F3 won
  every time and `Σ ST ≈ 1.2`. Small-N noise does **not** reproduce his
  signature, so something specific in his code does. We need the raw Y vector
  (all 192 values) and the sample matrix — not the summary table — to say what.

- **Budget.** `run_real_pilot_gsa()` in `Sensitivity_analysis.py` calls
  `saltelli.sample(problem, 16)`, which for 5 variables expands to
  **192 real pilot runs ≈ 384 000 EUR** against a 15 000 EUR benchmark. (`N` in
  Saltelli is a *base* sample, not a total — the small-looking 16 is the trap.)
  `basic_client.run` raises on HTTP 402, so it should have died early; we should
  find out what that run actually cost. That call is currently the **live** line
  in `Sensitivity_analysis.py`'s `__main__`, with the free surrogate version
  commented out — it should not be run again as written.

---

## 8. The transferable lesson

**A variance-based sensitivity analysis measures the design at least as much as
it measures the process.** Run one on data collected by an optimizer that has
already converged, and it will confidently report that the variables the
optimizer got right are the ones that don't matter — because it fixed them.
GSA wants a space-filling design; BO logs are the opposite of one by
construction.

Before reporting any index, ask the boring question first: **did we actually
vary this input?** The script now asks it automatically.
