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
*Update:* falsification #1 (fillet) + #2 (seed-averaged HF) results are appended
below once run.
