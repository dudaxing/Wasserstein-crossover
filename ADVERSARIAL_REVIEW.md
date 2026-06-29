# Adversarial Review — body-fitted HF in the Wasserstein EA

Date: 2026-06-29. Scope: the L-bracket integration that uses the body-fitted-mesh
stress model (`src/bodyfitted.py`, DPTO port) as the EA's high-fidelity (HF)
objective (`src/lbracket.py`, `experiments/run_lbracket.py`).

## 0. Verdict
The body-fitted HF stress evaluator is **correct and verified** (CST patch test +
machine-precision MATLAB cross-check). But "use it as the HF objective in the
Wasserstein EA" on the L-bracket, as wired, **fails**: the EA makes **zero useful
progress** (min J₁ 0.2636 → 0.2636 over 15 generations). The diagnostics show
this is structural, not a coding bug.

## 1. The failure, plainly
- min J₁ unchanged; the representative "optimized" design *is* an initial design
  (both panels J₁=0.414, J₂=0.285 in `lbracket_compare`).
- The only HV motion (+6%) comes from **extending the front into the degenerate
  low-volume corner** (J₁≈12 at J₂≈0.24), not from relieving stress. Reporting
  "+6% HV" alone would overclaim.

## 2. Root causes — measured

**(a) The objective is noise-limited.** Same design, 8 different meshes:

| design (J₂) | J₁ mean | mesh-noise std | CV |
|---|---|---|---|
| 0.33 | 0.406 | 0.011 | 2.7% |
| 0.44 | 0.318 | 0.016 | 5.0% |
| 0.54 | 0.295 | 0.011 | 3.6% |

Design-to-design signal: std ≈ **0.051**. So mesh noise is ~25–30% of the signal
(SNR ≈ 3–5×). A real 0.02 improvement is the size of the noise — the EA cannot
reliably tell a better design from a luckier mesh, so it partly optimizes
meshing artifacts.

**(b) A singularity floor the operator can't touch.** Across all designs the max
stress sits 0.6–0.8 units from the re-entrant corner (60,60) — always *at* the
corner. The corner is the fixed passive-void corner; every load-carrying design
keeps material beside it, so J₁ has a floor that no Wasserstein blend can lower.

**(c) Raw-max is fragile, p-norm barely helps here.** Across meshes: raw-max
CV 5.0% vs p-norm(16) CV 4.1% — only marginally better, because the corner
dominates both. "Just use p-norm" is not sufficient on its own.

## 3. Methodological critique
DPTO works because it is a **gradient** method that (i) remeshes only
occasionally (when β changes), (ii) uses **smooth aggregate** constraints
(p-norm + overall stress), and (iii) lets its **design field cover the corner** so
material can be removed there. My EA does the opposite on all three: it remeshes
**every** candidate (→ noise), scores **raw max** (→ fragility), and **freezes the
corner as passive** (→ a singularity it cannot escape). The LF↔HF gap exists but
lives at the corner (unreachable by blends) and is buried under mesh noise.

## 4. Validated vs. must-not-overclaim
- ✅ Verified: CST FEA (patch test + machine-precision MATLAB cross-check), mesh
  quality, corner physics, end-to-end execution.
- ❌ Not verified: that the full random-mesh pipeline reproduces DPTO's end-to-end
  stress on a given design (different meshes, §2a); and that the HF is a *usable
  EA objective* (§1–2 show it is not). "Cross-checked against MATLAB" blesses the
  element FEA, **not** EA suitability.

## 5. Other weaknesses
- HV reference polluted by the J₁≈12 degenerate extreme.
- Single-seed EA run; no repeats.
- Cost ~0.9 s/HF eval (≈6 s at h=1) — won't scale to paper-size populations.
- `seed=0` "determinism" is illusory: different designs still get different
  meshes, so the objective is effectively stochastic and the noise is not averaged.
- CST is low-order → poor stress accuracy near concentrations.

## 6. Falsification plan
If the diagnosis is right, these flip the result:
1. **Fillet the re-entrant corner** (radius r): removes the singularity → finite,
   design-sensitive, mesh-convergent J₁. Strongest test.
2. **De-noise the HF**: average over k mesh seeds (or fixed background mesh).
3. **DPTO's actual objective** (p-norm + overall stress) instead of raw max.
4. **Free the corner** (design, not passive) so material can be removed there.
5. **Robust HV**: fixed reference + drop degenerate extremes.

## 7. Bottom line
The body-fitted HF is a faithful, MATLAB-verified stress solver. But as an EA
objective on the sharp-corner L-bracket it does not improve designs, for
structural reasons (fixed singular corner + mesh noise ≈ signal), not a bug. It
is fixable, but the fixes change the problem and must be done deliberately — not
hidden behind a +6% HV number.

---

> **RETRACTION (2026-06-29):** the L-bracket numbers in §8–§9 below were computed with a body-fitted HF that had two P0 bugs (unsupported fixed end; mesh-dependent load). They are **contaminated and retracted**. The bugs are now fixed/verified (see ADVERSARIAL_PROJECT_REVIEW.md §11); a clean re-run supersedes these.

## 8. Falsification results (diagnosis confirmed)

Applied fix #1 (fillet the re-entrant corner, radius r=10) + fix #2 (average the
HF over 3 mesh seeds), then re-ran the same EA.

**Fix #1 makes the objective well-posed.** Solid L-bracket, refining `minedge`
5→3→2:
- sharp corner: 0.294 → 0.284 → 0.243 (divergent/non-monotone — singular);
- fillet r=10: 0.214 → 0.198 → 0.195 (**mesh-convergent**), and the max moves
  onto the fillet (corner-dist ~6 — design-sensitive), not the singular corner.

**The EA now improves** (it did not before). Best max-stress at matched volume,
initial (LF) vs final:

| volume band | initial J₁ | final J₁ | change |
|---|---|---|---|
| 0.30 | 0.3044 | 0.3044 | 0% |
| 0.35 | 0.2798 | 0.2798 | 0% |
| **0.40** | 0.2768 | **0.2652** | **−4.2%** |
| **0.45** | 0.2570 | **0.2449** | **−4.7%** |
| 0.50 | 0.2420 | 0.2407 | −0.5% |

Hypervolume rises **monotonically** to **+8.7%** (vs the sharp-corner run's
noisy, non-monotone +6% with *zero* matched-volume gain). So the two root causes
identified in §2 — the fixed singular corner and mesh-to-mesh noise — were indeed
the blockers; removing them unlocks real (if modest) stress reduction.

**Honest scope (compliance-LF run).** Gains were ~4–5% at intermediate volume
only. Lesson reinforced: **report matched-volume best-J₁, not global min J₁ or
raw HV.**

## 9. Stress LF + a selection-elitism fix (further improvement)

Acting on §8(ii) ("compliance-LF is stress-blind"), the L-bracket LF was switched
from compliance to a **density-method P-norm stress** optimization (with passive
void). The stress LF both starts lower (initial min J₁ 0.222 vs compliance 0.242)
and gives the EA a better basin.

**A new flaw surfaced during this re-run** (the value of iterating): with the
stress LF the EA reached min J₁ 0.209 by t≈6 but then **drifted back up to 0.221**
— impossible under proper elitism. Cause: when Pareto front-0 exceeds the
population, the farthest-point **diversity truncation could discard the best-J₁
design**. Fix: `select` now always retains each front's per-objective extreme
points (NSGA-II gives boundary points infinite crowding distance);
`test_select_keeps_objective_extremes` guards it.

**Result with stress LF + elitism fix** (same initial population, matched-volume
best J₁, initial→final):

| volume | compliance LF | stress LF (buggy sel.) | **stress LF + elitism** |
|---|---|---|---|
| 0.30 | 0% | −22.2% | **−32.2%** |
| 0.35 | 0% | −21.9% | −19.4% |
| 0.40 | 0% | 0% | 0% |
| 0.45 | −4.7% | −4.5% | −6.1% |
| 0.50 | −0.5% | −5.0% | **−10.5%** |
| global min J₁ | flat | 0.222→0.221 (drifted, bug) | **0.222→0.208 (monotone)** |

So: stress LF >> compliance LF, and the elitism fix both makes min J₁ monotone
and improves the gains further (−32% at V≈0.30, −10.5% at V≈0.50). The
body-fitted-HF EA is now a genuinely effective optimizer on this problem.

**Remaining honesty.** Raw HV (+0.5%) is still a poor headline here (reference
dominated by a degenerate low-volume extreme) — matched-volume best-J₁ is the
right metric. Residual limiters unchanged: 3-seed averaging, structured→body-fitted
resampling noise, raw-max (not p-norm) HF, single EA seed, CST element.
