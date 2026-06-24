"""
Deterministic unit tests for the core operator and the multi-objective tools.

Complements `test_fem.py` (FEM / MMA / adjoint).  These tests pin down properties
of the Wasserstein barycenter / crossover, the NSGA-II selection, and the
hypervolume indicator on small, hand-checkable inputs -- so the conclusions in
the README do not rest on untested metric implementations.

Run:  python experiments/test_operator.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wasserstein import (convolutional_barycenter, wasserstein_crossover,   # noqa
                         eps_to_sigma, adaptive_eps, population_distance_matrix)
from selection import (dominates, fast_non_dominated_sort, crowding_distance,  # noqa
                       select, hypervolume, farthest_point_order)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _disk(n, cx, cy, r):
    y, x = np.mgrid[0:n, 0:n]
    d = ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(float)
    return d / d.sum()


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


# --------------------------------------------------------------------------- #
#  Wasserstein operator
# --------------------------------------------------------------------------- #
def test_eps_to_sigma():
    print("[eps_to_sigma]")
    # sigma = sqrt(eps/2)/h
    check(abs(eps_to_sigma(2.0, 1.0) - 1.0) < 1e-12, "eps=2,h=1 -> sigma=1")
    check(abs(eps_to_sigma(8.0, 2.0) - 1.0) < 1e-12, "eps=8,h=2 -> sigma=1")
    check(eps_to_sigma(1e-4, 0.01) > eps_to_sigma(1e-6, 0.01), "monotone in eps")


def test_adaptive_eps():
    print("[adaptive_eps]")
    emin, emax = 1e-6, 1e-4
    check(abs(adaptive_eps(2.0, 2.0, 10.0, emin, emax) - emin) < 1e-18, "d=d_min -> eps_min")
    check(abs(adaptive_eps(10.0, 2.0, 10.0, emin, emax) - emax) < 1e-18, "d=d_max -> eps_max")
    mid = adaptive_eps(6.0, 2.0, 10.0, emin, emax)
    check(emin < mid < emax, "interior strictly between")
    # degenerate d_min == d_max -> midpoint (no division by zero)
    deg = adaptive_eps(5.0, 5.0, 5.0, emin, emax)
    check(abs(deg - 0.5 * (emin + emax)) < 1e-18, "d_min==d_max -> midpoint")
    # monotone increasing in d
    ds = np.linspace(2, 10, 9)
    es = [adaptive_eps(d, 2, 10, emin, emax) for d in ds]
    check(all(np.diff(es) >= -1e-18), "monotone non-decreasing in distance")


def test_population_distance_matrix():
    print("[population_distance_matrix]")
    rng = np.random.default_rng(0)
    P = rng.random((6, 20))
    D = population_distance_matrix(P)
    check(D.shape == (6, 6), "shape")
    check(np.allclose(D, D.T, atol=1e-10), "symmetric")
    check(np.allclose(np.diag(D), 0.0, atol=1e-10), "zero diagonal")
    # match brute force
    Dbf = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            Dbf[i, j] = np.linalg.norm(P[i] - P[j])
    check(np.allclose(D, Dbf, atol=1e-8), "matches brute-force norms")


def test_barycenter_mass_nonneg():
    print("[barycenter mass + non-negativity]")
    n = 24
    A, B = _disk(n, 8, 12, 4), _disk(n, 16, 12, 5)
    bary = convolutional_barycenter(np.stack([A, B]), weights=[0.5, 0.5],
                                    sigma=1.5, n_iter=200, tol=1e-12)
    check(abs(bary.sum() - 1.0) < 1e-6, "barycenter sums to 1")
    check(bary.min() >= -1e-12, "barycenter non-negative")
    check(bary.shape == (n, n), "shape preserved")


def test_barycenter_swap_symmetry():
    print("[barycenter swap symmetry]")
    n = 24
    A, B = _disk(n, 8, 12, 4), _disk(n, 16, 12, 5)
    w = 0.3
    b1 = convolutional_barycenter(np.stack([A, B]), weights=[w, 1 - w],
                                  sigma=1.5, n_iter=200, tol=1e-12)
    b2 = convolutional_barycenter(np.stack([B, A]), weights=[1 - w, w],
                                  sigma=1.5, n_iter=200, tol=1e-12)
    check(np.allclose(b1, b2, atol=1e-6), "bary([A,B],[w,1-w]) == bary([B,A],[1-w,w])")


def test_barycenter_identical_parents():
    print("[barycenter identical parents]")
    n = 24
    A = _disk(n, 12, 12, 5)
    unrelated = _disk(n, 5, 5, 4)
    b_a = convolutional_barycenter(np.stack([A, A]), weights=[0.3, 0.7],
                                   sigma=0.6, n_iter=200, tol=1e-12)
    b_b = convolutional_barycenter(np.stack([A, A]), weights=[0.9, 0.1],
                                   sigma=0.6, n_iter=200, tol=1e-12)
    check(np.allclose(b_a, b_b, atol=1e-6), "identical parents: invariant to weights")
    check(_corr(b_a, A) > 0.95, "identical parents: barycenter ~ the parent (corr>0.95)")
    check(_corr(b_a, A) > _corr(b_a, unrelated) + 0.5,
          "identical parents: far closer to the parent than to an unrelated shape")


def test_barycenter_weight_extremes():
    print("[barycenter weight extremes]")
    n = 28
    A, B = _disk(n, 8, 14, 4), _disk(n, 20, 14, 4)
    near_A = convolutional_barycenter(np.stack([A, B]), weights=[0.98, 0.02],
                                      sigma=1.5, n_iter=300, tol=1e-12)
    near_B = convolutional_barycenter(np.stack([A, B]), weights=[0.02, 0.98],
                                      sigma=1.5, n_iter=300, tol=1e-12)
    check(_corr(near_A, A) > _corr(near_A, B), "weight on A -> closer to A")
    check(_corr(near_B, B) > _corr(near_B, A), "weight on B -> closer to B")


def test_crossover_range():
    print("[wasserstein_crossover output range]")
    n = 24
    A, B = _disk(n, 8, 12, 4).ravel(), _disk(n, 16, 12, 5).ravel()
    child = wasserstein_crossover(A, B, (n, n), lam=0.5, sigma=1.5,
                                  n_iter=200, tol=1e-12,
                                  rng=np.random.default_rng(0))
    check(child.shape == (n * n,), "flattened length")
    check(child.min() >= -1e-9 and child.max() <= 1 + 1e-9, "values in [0,1]")
    check(abs(child.min()) < 1e-6 and abs(child.max() - 1.0) < 1e-6,
          "min-max scaled to span [0,1]")


def test_crossover_endpoints_bounded():
    print("[wasserstein_crossover lambda endpoints bounded]")
    n = 24
    A, B = _disk(n, 8, 12, 4).ravel(), _disk(n, 16, 12, 5).ravel()
    for lam in (0.0, 1.0, 1e-3, 1.0 - 1e-3):
        child = wasserstein_crossover(A, B, (n, n), lam=lam, sigma=1.5,
                                      n_iter=200, tol=1e-12,
                                      rng=np.random.default_rng(0))
        check(np.all(np.isfinite(child)), f"lambda={lam}: finite (no NaN/Inf)")
        check(child.min() >= -1e-9 and child.max() <= 1 + 1e-9,
              f"lambda={lam}: bounded in [0,1]")
    # convention: lam is the weight on parent A (weights=[lam, 1-lam]),
    # so lam~1 leans toward A and lam~0 leans toward B.
    near_A = wasserstein_crossover(A, B, (n, n), lam=1.0 - 1e-3, sigma=1.5,
                                   n_iter=300, tol=1e-12, rng=np.random.default_rng(0))
    near_B = wasserstein_crossover(A, B, (n, n), lam=1e-3, sigma=1.5,
                                   n_iter=300, tol=1e-12, rng=np.random.default_rng(0))
    check(_corr(near_A.reshape(n, n), A.reshape(n, n))
          > _corr(near_A.reshape(n, n), B.reshape(n, n)), "lambda~1 -> closer to A")
    check(_corr(near_B.reshape(n, n), B.reshape(n, n))
          > _corr(near_B.reshape(n, n), A.reshape(n, n)), "lambda~0 -> closer to B")


# --------------------------------------------------------------------------- #
#  Selection
# --------------------------------------------------------------------------- #
def test_dominates():
    print("[dominates]")
    check(dominates(np.array([1, 1]), np.array([2, 2])), "(1,1) dominates (2,2)")
    check(dominates(np.array([1, 2]), np.array([1, 3])), "ties allowed if one strict")
    check(not dominates(np.array([1, 3]), np.array([2, 1])), "trade-off: no dominance")
    check(not dominates(np.array([1, 1]), np.array([1, 1])), "equal: no dominance")


def test_non_dominated_sort():
    print("[fast_non_dominated_sort]")
    F = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0],  # front 0
                  [2.0, 5.0], [5.0, 2.0]])                          # front 1
    fronts = fast_non_dominated_sort(F)
    check(set(fronts[0]) == {0, 1, 2, 3}, "front 0 is the Pareto set")
    check(set(fronts[1]) == {4, 5}, "front 1 is the dominated points")
    P = F[fronts[0]]
    mutual = all(not dominates(a, b) for a in P for b in P if not np.array_equal(a, b))
    check(mutual, "front 0 points mutually non-dominated")


def test_front_stable_under_dominated():
    print("[front unaffected by a dominated point]")
    F = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])  # all Pareto
    f0 = set(fast_non_dominated_sort(F)[0])
    # add a clearly dominated point
    F2 = np.vstack([F, [5.0, 5.0]])
    fronts2 = fast_non_dominated_sort(F2)
    check(set(fronts2[0]) == f0, "front 0 unchanged when a dominated point is added")
    check(4 in fronts2[1], "the dominated point lands in a later front")


def test_farthest_point_deterministic():
    print("[farthest_point_order determinism]")
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                  [0.5, 0.5], [0.9, 0.1]])
    front = [0, 1, 2, 3, 4, 5]
    o1 = farthest_point_order(X, front)
    o2 = farthest_point_order(X, front)
    check(o1 == o2, "deterministic: identical ordering across calls")
    check(sorted(o1) == sorted(front), "ordering is a permutation of the input")
    # with some points already selected, ordering still deterministic
    o3 = farthest_point_order(X, [1, 2, 4], already=np.array([0, 3]))
    o4 = farthest_point_order(X, [1, 2, 4], already=np.array([0, 3]))
    check(o3 == o4 and sorted(o3) == [1, 2, 4], "deterministic with `already` set")


def test_select_elitism():
    print("[select size + elitism]")
    rng = np.random.default_rng(2)
    F = rng.random((30, 2))
    X = rng.random((30, 12))
    sel = select(F, 12, X=X, mode="diversity")
    check(len(sel) == 12 and len(set(sel.tolist())) == 12, "returns N unique survivors")
    # all of front 0 retained when it fits within N
    f0 = set(fast_non_dominated_sort(F)[0])
    if len(f0) <= 12:
        check(f0.issubset(set(sel.tolist())), "entire front 0 retained (elitism)")
    sel_c = select(F, 12, X=X, mode="crowding")
    check(len(sel_c) == 12, "crowding mode also returns N")


def test_crowding_distance():
    print("[crowding_distance]")
    F = np.array([[0.0, 1.0], [0.25, 0.6], [0.5, 0.4], [1.0, 0.0]])
    cd = crowding_distance(F, list(range(4)))
    order = np.argsort(F[:, 0])
    check(np.isinf(cd[order[0]]) and np.isinf(cd[order[-1]]), "boundary points -> inf")
    check(np.all(np.isfinite(cd[order[1:-1]])), "interior points finite")


# --------------------------------------------------------------------------- #
#  Hypervolume
# --------------------------------------------------------------------------- #
def test_hypervolume_analytic():
    print("[hypervolume analytic 2D]")
    ref = np.array([1.0, 1.0])
    check(abs(hypervolume(np.array([[0.0, 0.0]]), ref) - 1.0) < 1e-9, "single (0,0) -> 1.0")
    check(abs(hypervolume(np.array([[0.5, 0.0], [0.0, 0.5]]), ref) - 0.75) < 1e-9,
          "two points -> 0.75")
    check(hypervolume(np.array([[1.0, 1.0]]), ref) == 0.0, "point at ref -> 0")
    check(hypervolume(np.array([[2.0, 2.0]]), ref) == 0.0, "point beyond ref -> 0")
    # staircase: (0.2,0.8) and (0.6,0.4) vs ref (1,1)
    hv = hypervolume(np.array([[0.2, 0.8], [0.6, 0.4]]), ref)
    expected = (1 - 0.2) * (1 - 0.8) + (1 - 0.6) * (0.8 - 0.4)
    check(abs(hv - expected) < 1e-9, "two-step staircase area")


def test_hypervolume_monotone():
    print("[hypervolume monotonicity]")
    ref = np.array([1.0, 1.0])
    base = np.array([[0.5, 0.5]])
    hv0 = hypervolume(base, ref)
    hv_better = hypervolume(np.vstack([base, [0.3, 0.3]]), ref)
    check(hv_better > hv0, "adding a dominating point increases HV")
    hv_dom = hypervolume(np.vstack([base, [0.8, 0.8]]), ref)
    check(abs(hv_dom - hv0) < 1e-9, "adding a dominated point leaves HV unchanged")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [
        test_eps_to_sigma, test_adaptive_eps, test_population_distance_matrix,
        test_barycenter_mass_nonneg, test_barycenter_swap_symmetry,
        test_barycenter_identical_parents, test_barycenter_weight_extremes,
        test_crossover_range, test_crossover_endpoints_bounded,
        test_dominates, test_non_dominated_sort,
        test_front_stable_under_dominated, test_farthest_point_deterministic,
        test_select_elitism, test_crowding_distance,
        test_hypervolume_analytic, test_hypervolume_monotone,
    ]
    for t in tests:
        t()
    print("\nALL OPERATOR TESTS PASSED")
