"""
Tests for the persistent-homology + Wasserstein diversity backend
(src/topo_selection.py, using torch_topological).

These run only when the torch_topological backend is importable (i.e. in the
dedicated TDA environment; see requirements-tda.txt).  On the default Python-3.14
environment they SKIP cleanly.

Run (in the TDA venv):  .ttvenv/Scripts/python experiments/test_topo_selection.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from selection import farthest_point_order_from_D  # noqa: E402
import topo_selection as ts  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def _shape(kind, n=64):
    img = np.zeros((n, n), np.float32)
    if kind == "square":
        img[16:48, 16:48] = 1
    elif kind == "one_hole":
        img[12:52, 12:52] = 1; img[26:38, 26:38] = 0
    elif kind == "two_holes":
        img[12:52, 12:52] = 1; img[20:30, 20:30] = 0; img[34:44, 34:44] = 0
    elif kind == "two_blocks":
        img[16:48, 8:26] = 1; img[16:48, 38:56] = 1
    return img.ravel()


def test_betti_counts():
    print("[cubical PH Betti counts]")
    n = 64
    pops = {k: _shape(k, n) for k in ["square", "one_hole", "two_holes", "two_blocks"]}
    betti = {}
    for k, v in pops.items():
        pis = ts.persistence_diagrams([v], (n, n))[0]
        b = {int(pi.dimension): len(pi.diagram) for pi in pis}
        betti[k] = b
        print(f"    {k:11s}: {b}")
    check(betti["square"].get(1, 0) == 0, "solid square: H1 = 0 holes")
    check(betti["one_hole"].get(1, 0) == 1, "one_hole: H1 = 1 hole")
    check(betti["two_holes"].get(1, 0) == 2, "two_holes: H1 = 2 holes")
    check(betti["two_blocks"].get(0, 0) == 2, "two_blocks: H0 = 2 components")


def test_ph_distance_matrix():
    print("[ph_distance_matrix]")
    n = 64
    pop = [_shape("square", n), _shape("square", n),
           _shape("one_hole", n), _shape("two_blocks", n)]
    D = ts.ph_distance_matrix(pop, (n, n))
    check(D.shape == (4, 4), "shape (4,4)")
    check(np.allclose(D, D.T, atol=1e-9), "symmetric")
    check(np.allclose(np.diag(D), 0.0, atol=1e-9), "zero diagonal")
    check(np.isfinite(D).all(), "finite entries")
    check(D[0, 1] < 1e-6, "identical designs -> distance ~ 0")
    check(D[0, 2] > 1e-3 and D[0, 3] > 1e-3, "different topology -> distance > 0")


def test_order_determinism():
    print("[farthest_point_order_from_D determinism]")
    D = np.array([[0., 1., 2., 3.],
                  [1., 0., 1.5, 2.5],
                  [2., 1.5, 0., 1.],
                  [3., 2.5, 1., 0.]])
    o1 = farthest_point_order_from_D(D, [0, 1, 2, 3])
    o2 = farthest_point_order_from_D(D, [0, 1, 2, 3])
    check(o1 == o2, "deterministic across calls")
    check(sorted(o1) == [0, 1, 2, 3], "permutation of input")
    o3 = farthest_point_order_from_D(D, [1, 2, 3], already=np.array([0]))
    check(o3 == farthest_point_order_from_D(D, [1, 2, 3], already=np.array([0])),
          "deterministic with `already` set")
    check(o3[0] == 3, "first pick is farthest from the already-selected point 0")


if __name__ == "__main__":
    if not ts.available():
        print("torch_topological backend NOT available -> SKIP "
              "(install the TDA environment; see requirements-tda.txt)")
        sys.exit(0)
    test_betti_counts()
    test_ph_distance_matrix()
    test_order_determinism()
    print("\nALL PH-SELECTION TESTS PASSED")
