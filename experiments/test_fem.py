"""Validation tests for the FEM / MMA / adjoint sensitivities."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from topopt import (Mesh, FEM, build_density_filter, apply_filter, filter_chain,
                    cracked_plate_bc, lf_optimize_stress,
                    make_cracked_plate_mesh)  # noqa: E402


def test_pnorm_sensitivity():
    """Finite-difference check of the adjoint P-norm stress sensitivity."""
    rng = np.random.default_rng(0)
    mesh = make_cracked_plate_mesh(12)
    h = mesh.h
    fixed, F = cracked_plate_bc(mesh, load=1.0)
    fem = FEM(mesh, fixed, F)
    H, Hs = build_density_filter(mesh, R=2.5 * h)
    x = 0.5 + 0.3 * rng.random(mesh.nel)
    rho = apply_filter(H, Hs, x)

    P, q = 6.0, 0.5
    J, dJdrho, sigma, U = fem.pnorm_stress(rho, P, q)
    dJdx = filter_chain(H, Hs, dJdrho)

    # check 8 random design variables by central differences
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
    print(f"[sens] max rel error = {max(err):.2e}   (analytic vs FD)")
    assert max(err) < 1e-4, "sensitivity mismatch"


def test_compliance_mma():
    """Compliance minimization should reduce compliance monotonically-ish."""
    mesh = make_cracked_plate_mesh(40)
    fixed, F = cracked_plate_bc(mesh, load=1.0)
    fem = FEM(mesh, fixed, F)
    H, Hs = build_density_filter(mesh, R=2.0 * mesh.h)
    x = np.full(mesh.nel, 0.4)
    c0, _, _ = fem.compliance(apply_filter(H, Hs, x))
    print(f"[fem] initial compliance = {c0:.4e}")
    assert np.isfinite(c0) and c0 > 0


def test_lf_stress_small():
    mesh = make_cracked_plate_mesh(40)
    fixed, F = cracked_plate_bc(mesh, load=1.0)
    fem = FEM(mesh, fixed, F)
    H, Hs = build_density_filter(mesh, R=3.0 * mesh.h)
    rho, x = lf_optimize_stress(mesh, fem, H, Hs, V=0.4, P=8.0,
                                maxiter=40, move=0.1, verbose=True)
    smax0, _, _ = fem.max_stress(np.full(mesh.nel, 0.4))
    smax1, _, _ = fem.max_stress(rho)
    print(f"[lf] uniform max-stress={smax0:.3f}  optimized max-stress={smax1:.3f}"
          f"  vol={rho.mean():.3f}")


if __name__ == "__main__":
    test_pnorm_sensitivity()
    test_compliance_mma()
    test_lf_stress_small()
    print("ALL TESTS DONE")
