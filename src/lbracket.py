"""
L-bracket problem for the Wasserstein-crossover EA, with a body-fitted-mesh
high-fidelity stress model.

  * Low-fidelity (LF): compliance minimization (optimality-criteria) on a
    structured grid over the L-bracket, with the upper-right passive void.  It is
    robust and, by being a *different* objective than the HF max-stress, leaves a
    genuine LF<->HF gap for the EA to exploit.  Seeding over filter radius and
    volume fraction yields a diverse initial population.
  * High-fidelity (HF): the validated body-fitted-mesh true max von Mises stress
    (src/bodyfitted.py, ported from DPTO and cross-checked against MATLAB).

The Wasserstein crossover operates on the LF structured grid; for HF the LF
density is resampled onto the body-fitted node grid.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from topopt import Mesh, FEM, build_density_filter, apply_filter, filter_chain
import bodyfitted as bf


# --------------------------------------------------------------------------- #
#  Geometry & boundary conditions (structured LF grid)
# --------------------------------------------------------------------------- #
def make_lbracket(nelx, L=150.0, lpd=60.0):
    """Square structured grid over [0,L]^2 with the upper-right passive void."""
    h = L / nelx
    mesh = Mesh(nelx, nelx, h)
    cx, cy = mesh.ecoord[:, 0], mesh.ecoord[:, 1]
    passive = (cx > lpd) & (cy > lpd)              # passive void elements
    return mesh, passive


def lbracket_bc(mesh, L=150.0, lpd=60.0, lload=6.0, load=1.0):
    """Fixed top edge of the vertical arm (y=L, x<=lpd); downward load on the
    strip y=lpd, x in [L-lload, L].  Matches DPTO / bodyfitted.lbracket_bcs."""
    h = mesh.h
    fixed = []
    iy_top = mesh.nny - 1
    for ix in range(mesh.nnx):
        if ix * h <= lpd + 1e-9:
            nd = mesh.node_id(ix, iy_top)
            fixed += [2 * nd, 2 * nd + 1]
    fixed = np.unique(np.array(fixed, int))
    F = np.zeros(mesh.ndof)
    iy = int(round(lpd / h))
    lnodes = [mesh.node_id(ix, iy) for ix in range(mesh.nnx)
              if L - lload - 1e-9 <= ix * h <= L + 1e-9]
    for nd in lnodes:
        F[2 * nd + 1] = -load / len(lnodes)
    return fixed, F


# --------------------------------------------------------------------------- #
#  Low-fidelity: compliance minimization (OC) with passive void
# --------------------------------------------------------------------------- #
def lf_optimize_compliance(mesh, fem, H, Hs, V, passive, maxiter=40, move=0.2,
                           tol=0.01):
    """Classic compliance OC with a density filter and passive-void elements.
    `V` is the volume fraction over the *design* (non-passive) region."""
    n = mesh.nel
    free_el = ~passive
    ndesign = int(free_el.sum())
    x = np.zeros(n)
    x[free_el] = V
    for it in range(maxiter):
        xphys = apply_filter(H, Hs, x)
        xphys[passive] = 0.0
        c, dc, U = fem.compliance(xphys)           # dc = dC/dxphys (<=0)
        dv = np.ones(n)
        dc = filter_chain(H, Hs, dc)
        dv = filter_chain(H, Hs, dv)
        # OC bisection on the Lagrange multiplier
        xold = x.copy()
        l1, l2 = 1e-9, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            be = np.maximum(0.0, -dc / (dv * lmid + 1e-30))
            xnew = np.clip(x * np.sqrt(be), np.maximum(0.0, x - move),
                           np.minimum(1.0, x + move))
            xnew[passive] = 0.0
            if xnew[free_el].sum() > V * ndesign:
                l1 = lmid
            else:
                l2 = lmid
        x = xnew
        ch = np.max(np.abs(x - xold))
        if it > 5 and ch < tol:
            break
    xphys = apply_filter(H, Hs, x); xphys[passive] = 0.0
    return xphys, x


# --------------------------------------------------------------------------- #
#  Problem object for the framework
# --------------------------------------------------------------------------- #
class LBracketProblem:
    def __init__(self, nelx_lf=75, L=150.0, lpd=60.0, lload=6.0, penal=3.0,
                 hf_h=2.0, hf_minedge=3.0, hf_maxedge=40.0, hf_iter=80,
                 load=1.0):
        self.L, self.lpd, self.lload = L, lpd, lload
        self.mesh, self.passive = make_lbracket(nelx_lf, L, lpd)
        self.fixed, self.F = lbracket_bc(self.mesh, L, lpd, lload, load)
        self.fem = FEM(self.mesh, self.fixed, self.F, penal=penal)
        self.grid_shape = (self.mesh.nely, self.mesh.nelx)
        self.n = self.mesh.nel
        # HF node grid + geometry
        self.hf_h = hf_h
        self.xn, self.yn = np.meshgrid(np.arange(0, L + hf_h * 0.5, hf_h),
                                       np.arange(0, L + hf_h * 0.5, hf_h))
        self.hf_geom = dict(L=L, lpd=lpd, lload=lload, h=hf_h,
                            minedge=hf_minedge, maxedge=hf_maxedge)
        self._hf_iter = hf_iter
        # element-centroid axes for resampling LF -> HF
        self._ex = (np.arange(self.mesh.nelx) + 0.5) * self.mesh.h
        self._ey = (np.arange(self.mesh.nely) + 0.5) * self.mesh.h

    # ---- LF: generate the diverse initial population ----
    def generate_initial_population(self, n_s1, n_s2, R_min, R_max, V_min, V_max,
                                    maxiter=40, move=0.2, verbose=False):
        s1 = np.linspace(0, 1, n_s1)
        s2 = np.linspace(0, 1, n_s2)
        designs, info = [], []
        k = 0
        for a in s1:
            R = R_min + (R_max - R_min) * a
            H, Hs = build_density_filter(self.mesh, R=R)
            for b in s2:
                V = V_min + (V_max - V_min) * b
                rho, x = lf_optimize_compliance(self.mesh, self.fem, H, Hs, V,
                                                self.passive, maxiter=maxiter,
                                                move=move)
                designs.append(rho.copy())
                info.append((R, V))
                k += 1
                if verbose:
                    obj, _, _ = self.hf_evaluate(rho)
                    print(f"  LF {k:3d}/{n_s1*n_s2}: R={R:.1f} V={V:.2f} "
                          f"-> J1={obj[0]:.3f} J2={obj[1]:.3f}")
        return np.array(designs), info

    # ---- resample LF element density -> HF node grid ----
    def _to_hf_field(self, gamma):
        g = np.asarray(gamma, float).reshape(self.grid_shape)
        interp = RegularGridInterpolator((self._ey, self._ex), g,
                                         bounds_error=False, fill_value=0.0)
        pts = np.column_stack([self.yn.ravel(), self.xn.ravel()])
        field = interp(pts).reshape(self.xn.shape)
        field[(self.xn > self.lpd) & (self.yn > self.lpd)] = 0.0   # passive void
        return np.clip(field, 0.0, 1.0)

    # ---- HF: body-fitted true max von Mises stress + volume fraction ----
    def hf_evaluate(self, gamma):
        field = self._to_hf_field(gamma)
        geom = dict(self.hf_geom)
        try:
            J1, J2 = bf.hf_lbracket_stress(field, self.xn, self.yn, geom=geom,
                                           seed=0, n_iter=self._hf_iter)
            feasible = np.isfinite(J1) and J2 > 1e-3
        except Exception as e:                      # degenerate offspring
            return np.array([np.inf, np.inf]), False, None
        return np.array([J1, J2]), bool(feasible), None
