"""
Honest verdict for the L-bracket Wasserstein-crossover EA.

Loads results/lbr_result_<tag>.npz and reports the *matched-volume best-J1*
(initial LF population vs final Pareto front-0), plus global min-J1 and the
hypervolume trajectory.  Matched-volume best-J1 is the right metric here: raw
min-J1 and raw HV are dominated by the degenerate low-volume corner.
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from selection import fast_non_dominated_sort   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def best_in_band(F, lo, hi):
    """min J1 among points with volume J2 in [lo,hi); None if none."""
    m = (F[:, 1] >= lo) & (F[:, 1] < hi)
    if not m.any():
        return None
    return float(F[m, 0].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sharpfix")
    args = ap.parse_args()
    d = np.load(os.path.join(OUT, f"lbr_result_{args.tag}.npz"), allow_pickle=True)
    F, initF, hv = d["F"], d["initF"], d["hv_hist"]

    # finite-only views
    fin = lambda A: A[np.isfinite(A).all(axis=1)]
    Ff, If = fin(F), fin(initF)
    fr = fast_non_dominated_sort(Ff)[0]
    front = Ff[fr]

    print(f"=== L-bracket EA verdict  (tag={args.tag}) ===")
    print(f"initial pop: {len(If)} finite designs   final pop: {len(Ff)} finite")
    print(f"global min J1: initial {If[:,0].min():.4f} -> final {Ff[:,0].min():.4f}")
    print(f"hypervolume:   {hv[0]:.4f} -> {hv[-1]:.4f}  ({100*(hv[-1]/hv[0]-1):+.1f}%)")
    monotone = bool(np.all(np.diff(hv) >= -1e-9))
    print(f"HV monotone non-decreasing: {monotone}")

    print("\nmatched-volume best-J1 (initial LF -> final front):")
    print(f"{'vol band':>12} | {'init J1':>8} | {'final J1':>8} | {'change':>8}")
    bands = [(0.275, 0.325), (0.325, 0.375), (0.375, 0.425),
             (0.425, 0.475), (0.475, 0.525), (0.525, 0.575)]
    any_gain = False
    for lo, hi in bands:
        bi = best_in_band(If, lo, hi)
        bf_ = best_in_band(front, lo, hi)
        c = bi is not None and bf_ is not None
        ch = f"{100*(bf_/bi-1):+.1f}%" if c else "  --"
        if c and bf_ < bi - 1e-6:
            any_gain = True
        print(f"{lo:.2f}-{hi:.2f} | "
              f"{(f'{bi:.4f}' if bi is not None else '  --'):>8} | "
              f"{(f'{bf_:.4f}' if bf_ is not None else '  --'):>8} | {ch:>8}")
    print(f"\nEA improves at >=1 matched volume band: {any_gain}")


if __name__ == "__main__":
    main()
