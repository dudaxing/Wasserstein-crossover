"""
Multi-seed comparison: ph_wasserstein vs L2 diversity selection.

Same cached initial population, same Wasserstein crossover, same config; only
(seed, sel_mode) vary.  Reports robust metrics so the verdict does not rest on
the noise-prone raw hypervolume:
  * min J1                      -- best peak stress;
  * best J1 at matched volume   -- min J1 among designs with J2 in a fixed band;
  * HV (fixed reference point)  -- same reference for every run, so comparable.

Paired by seed; reports mean +/- std per mode and a paired t-test on the
differences.  Run in the TDA venv (needs torch_topological for ph_wasserstein):
  .ttvenv/Scripts/python experiments/multiseed_selection.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framework import StressPlateProblem, run_framework            # noqa: E402
from selection import hypervolume                                  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]
MODES = ["diversity", "ph_wasserstein"]
T_MAX = 20
V_BAND = (0.28, 0.32)        # matched-volume band for best-J1 metric


def metrics(F, ref_fixed):
    F = np.asarray(F)
    in_band = (F[:, 1] >= V_BAND[0]) & (F[:, 1] <= V_BAND[1])
    best_v = float(F[in_band, 0].min()) if in_band.any() else float("nan")
    return {
        "min_J1": float(F[:, 0].min()),
        "bestJ1_at_V": best_v,
        "hv_fixed": float(hypervolume(F, ref_fixed)),
    }


def main():
    cache = os.path.join(OUT, "init_pop_60_4x10.npz")
    designs = np.load(cache, allow_pickle=True)["designs"]
    problem = StressPlateProblem(nelx=60, R_h=0.01)
    initF = np.array([problem.hf_evaluate(g)[0] for g in designs])
    # fixed reference point from the (deterministic) initial population
    ref_fixed = 1.1 * initF.max(axis=0)
    print(f"fixed reference point = {ref_fixed}")
    print(f"initial: min J1 = {initF[:,0].min():.3f}\n")

    base = dict(n_s1=4, n_s2=10, N_pop=40, N_xo=40, t_max=T_MAX,
                R_min=0.03, R_max=0.12, V_min=0.30, V_max=0.60,
                eps_min=1e-5, eps_max=1.5e-3, wc_iter=400, wc_tol=1e-8,
                init_designs=designs, init_info=[None] * len(designs))

    runs = {m: [] for m in MODES}
    t0 = time.time()
    for mode in MODES:
        for seed in SEEDS:
            cfg = dict(base); cfg["seed"] = seed; cfg["sel_mode"] = mode
            res = run_framework(problem, cfg, crossover="wasserstein",
                                logger=lambda *a, **k: None)   # quiet
            mt = metrics(res["F"], ref_fixed)
            runs[mode].append(mt)
            print(f"[{mode:14s} seed={seed}] min_J1={mt['min_J1']:.3f}  "
                  f"bestJ1@V={mt['bestJ1_at_V']:.3f}  hv_fixed={mt['hv_fixed']:.4f}  "
                  f"({time.time()-t0:.0f}s elapsed)")

    # ---- summary ----
    summary = {"t_max": T_MAX, "seeds": SEEDS, "v_band": V_BAND,
               "ref_fixed": ref_fixed.tolist(), "runs": runs, "stats": {}}
    print("\n" + "=" * 72)
    print(f"{'metric':16s} {'diversity (mean±std)':24s} {'ph_wasserstein (mean±std)':26s}")
    print("-" * 72)
    keys = ["min_J1", "bestJ1_at_V", "hv_fixed"]
    for k in keys:
        dv = np.array([r[k] for r in runs["diversity"]])
        ph = np.array([r[k] for r in runs["ph_wasserstein"]])
        diff = ph - dv                                  # paired (same seed)
        try:
            from scipy.stats import ttest_rel
            tval, pval = ttest_rel(ph, dv)
        except Exception:
            tval, pval = float("nan"), float("nan")
        summary["stats"][k] = {
            "diversity_mean": float(dv.mean()), "diversity_std": float(dv.std(ddof=1)),
            "ph_mean": float(ph.mean()), "ph_std": float(ph.std(ddof=1)),
            "paired_diff_mean": float(diff.mean()), "paired_diff_std": float(diff.std(ddof=1)),
            "ttest_t": float(tval), "ttest_p": float(pval),
        }
        print(f"{k:16s} {dv.mean():8.3f} ± {dv.std(ddof=1):6.3f}        "
              f"{ph.mean():8.3f} ± {ph.std(ddof=1):6.3f}     "
              f"Δ(ph-div)={diff.mean():+.3f}±{diff.std(ddof=1):.3f}  p={pval:.3f}")

    path = os.path.join(OUT, "multiseed_selection.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nsaved {path}\ntotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
