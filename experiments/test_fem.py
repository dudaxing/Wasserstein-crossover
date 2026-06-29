"""
Validation of the Q4 plane-stress FEM + P-norm-stress adjoint sensitivity used
by the low-fidelity stress optimizer (src/topopt.py).

Geometry-agnostic (a simple rectangular cantilever) so it does not depend on any
particular example.  Checks:
  1. adjoint dJ/dx of the P-norm von Mises stress vs central finite differences;
  2. lf_optimize_stress actually reduces the true max stress.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from topopt import (Mesh, FEM, build_density_filter, apply_filter, filter_chain,
                    lf_optimize_stress)  # noqa: E402


def _cantilever(nelx=12, nely=8):
    """Rectangular cantilever: fix the left edge (x=0), vertical load at the
    bottom-right corner."""
    mesh = Mesh(nelx, nely, 1.0 / nelx)
    fixed = []
    for iy in range(mesh.nny):
        nd = mesh.node_id(0, iy)
        fixed += [2 * nd, 2 * nd + 1]
    fixed = np.unique(np.array(fixed, int))
    F = np.zeros(mesh.ndof)
    nd = mesh.node_id(mesh.nnx - 1, 0)
    F[2 * nd + 1] = -1.0
    fem = FEM(mesh, fixed, F)
    return mesh, fem


def test_pnorm_sensitivity():
    print("[adjoint P-norm stress sensitivity vs finite differences]")
    rng = np.random.default_rng(0)
    mesh, fem = _cantilever(12, 8)
    H, Hs = build_density_filter(mesh, R=2.5 * mesh.h)
    x = 0.5 + 0.3 * rng.random(mesh.nel)
    P, q = 6.0, 0.5
    J, dJdrho, sigma, U = fem.pnorm_stress(apply_filter(H, Hs, x), P, q)
    dJdx = filter_chain(H, Hs, dJdrho)
    idx = rng.choice(mesh.nel, 8, replace=False)
    eps = 1e-6
    err = []
    for i in idx:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        Jp = fem.pnorm_stress(apply_filter(H, Hs, xp), P, q)[0]
        Jm = fem.pnorm_stress(apply_filter(H, Hs, xm), P, q)[0]
        fd = (Jp - Jm) / (2 * eps)
        err.append(abs(fd - dJdx[i]) / (abs(fd) + 1e-12))
    print(f"  max rel error = {max(err):.2e}")
    assert max(err) < 1e-4, "adjoint sensitivity mismatch"


def test_lf_stress_reduces_max():
    print("[lf_optimize_stress reduces true max von Mises]")
    mesh, fem = _cantilever(20, 12)
    H, Hs = build_density_filter(mesh, R=2.0 * mesh.h)
    rho0 = np.full(mesh.nel, 0.4)
    s0, _, _ = fem.max_stress(rho0)
    rho, x = lf_optimize_stress(mesh, fem, H, Hs, V=0.4, P=8.0, maxiter=40,
                                move=0.1)
    s1, _, _ = fem.max_stress(rho)
    print(f"  uniform max-stress={s0:.3f} -> optimized={s1:.3f}, vol={rho.mean():.3f}")
    assert s1 < s0, "optimization did not reduce max stress"
    assert abs(rho.mean() - 0.4) < 0.05, "volume constraint not met"


if __name__ == "__main__":
    test_pnorm_sensitivity()
    test_lf_stress_reduces_max()
    print("ALL FEM TESTS PASSED")
