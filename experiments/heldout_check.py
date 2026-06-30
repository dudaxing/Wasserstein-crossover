"""Winner's-curse check: re-evaluate the final Pareto front and the initial LF
designs on HELD-OUT mesh seeds (the EA used seeds 0,1,2) and recompute the
matched-volume best-J1 gains.  If the gains survive, they are not mesh luck."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lbracket import LBracketProblem            # noqa: E402
from selection import fast_non_dominated_sort   # noqa: E402
import bodyfitted as bf                          # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
HELD = [5, 6, 7]

res = np.load(os.path.join(OUT, "lbr_result_paper.npz"), allow_pickle=True)
pop, F = res["population"], res["F"]
lf = np.load(glob.glob(os.path.join(OUT, "lbr_initpop_paper_*.npz"))[0], allow_pickle=True)
init_designs = lf["designs"]

prob = LBracketProblem(nelx_lf=75, hf_h=2.0, hf_minedge=3.0, hf_iter=80,
                       r_fillet=0.0, hf_seeds=3, lf_method="stress")


def eval_heldout(design):
    fld = prob._to_hf_field(design)
    j1, j2 = [], []
    for s in HELD:
        J1, J2 = bf.hf_lbracket_stress(fld, prob.xn, prob.yn,
                                       geom=dict(prob.hf_geom), seed=s, n_iter=80)
        if np.isfinite(J1) and J2 > 1e-3:
            j1.append(J1); j2.append(J2)
    return np.array([np.mean(j1), np.mean(j2)]) if j1 else np.array([np.inf, np.inf])


finmask = np.where(np.isfinite(F).all(1))[0]
fr = fast_non_dominated_sort(F[finmask])[0]
front_idx = finmask[fr]
print(f"re-evaluating {len(front_idx)} front + {len(init_designs)} initial designs "
      f"on held-out seeds {HELD} ...", flush=True)
Hf = np.array([eval_heldout(pop[i]) for i in front_idx])
Hi = np.array([eval_heldout(d) for d in init_designs])
fin = lambda A: A[np.isfinite(A).all(1)]
Hf, Hi = fin(Hf), fin(Hi)


def best(A, lo, hi):
    m = (A[:, 1] >= lo) & (A[:, 1] < hi)
    return float(A[m, 0].min()) if m.any() else None


print("\nHELD-OUT seeds (5,6,7) -- matched-volume best-J1, initial -> final front:")
for lo, hi in [(0.28, 0.42), (0.42, 0.47), (0.47, 0.53), (0.53, 0.57)]:
    bi, bf_ = best(Hi, lo, hi), best(Hf, lo, hi)
    ch = f"{100*(bf_/bi-1):+.1f}%" if (bi and bf_) else "  --"
    print(f"  V {lo:.2f}-{hi:.2f}: "
          f"{round(bi,4) if bi else None} -> {round(bf_,4) if bf_ else None}   {ch}")
np.savez(os.path.join(OUT, "heldout_paper.npz"), Hf=Hf, Hi=Hi)
print("\nsaved results/heldout_paper.npz")
