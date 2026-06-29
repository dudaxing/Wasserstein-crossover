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

from topopt import (Mesh, FEM, build_density_filter, apply_filter, filter_chain,
                    lf_optimize_stress)
import bodyfitted as bf


# --------------------------------------------------------------------------- #
#  Geometry & boundary conditions (structured LF grid)
# --------------------------------------------------------------------------- #
def make_lbracket(nelx, L=150.0, lpd=60.0, r_fillet=0.0):
    """Square structured grid over [0,L]^2 with the upper-right passive void
    (re-entrant corner optionally rounded by a fillet of radius r_fillet)."""
    h = L / nelx
    mesh = Mesh(nelx, nelx, h)
    cx, cy = mesh.ecoord[:, 0], mesh.ecoord[:, 1]
    passive = bf.passive_void_mask(cx, cy, lpd, r_fillet)
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
                 load=1.0, r_fillet=0.0, hf_seeds=5,
                 lf_method="stress", lf_P=8.0, lf_q=0.5):
        self.L, self.lpd, self.lload, self.r_fillet = L, lpd, lload, r_fillet
        self.lf_method, self.lf_P, self.lf_q = lf_method, lf_P, lf_q
        self.mesh, self.passive = make_lbracket(nelx_lf, L, lpd, r_fillet)
        self.fixed, self.F = lbracket_bc(self.mesh, L, lpd, lload, load)
        self.fem = FEM(self.mesh, self.fixed, self.F, penal=penal)
        self.grid_shape = (self.mesh.nely, self.mesh.nelx)
        self.n = self.mesh.nel
        self.hf_seeds = hf_seeds          # # of mesh seeds to average over (de-noise)
        # HF node grid + geometry
        self.hf_h = hf_h
        self.xn, self.yn = np.meshgrid(np.arange(0, L + hf_h * 0.5, hf_h),
                                       np.arange(0, L + hf_h * 0.5, hf_h))
        self.hf_geom = dict(L=L, lpd=lpd, lload=lload, h=hf_h,
                            minedge=hf_minedge, maxedge=hf_maxedge,
                            r_fillet=r_fillet)
        self._hf_iter = hf_iter
        # element-centroid axes for resampling LF -> HF
        self._ex = (np.arange(self.mesh.nelx) + 0.5) * self.mesh.h
        self._ey = (np.arange(self.mesh.nely) + 0.5) * self.mesh.h

    # ---- random initial density field (for multistart diversity) ----
    def _random_density(self, V, rng, smooth):
        """A smooth random density field with mean ~V over the design region.
        Low-frequency noise -> random *blobs* -> different topological basins,
        so the same (R,V) seed converges to a genuinely different optimum."""
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(rng.random(self.grid_shape), sigma=smooth,
                            mode="reflect").ravel()
        design = ~self.passive
        g = g - g[design].min()
        if g[design].max() > 1e-12:
            g = g / g[design].max()
        cur = g[design].mean()
        if cur > 1e-9:
            g = g * (V / cur)
        g = np.clip(g, 0.0, 1.0)
        g[self.passive] = 0.0
        return g

    # ---- LF: generate the diverse initial population ----
    def generate_initial_population(self, n_s1, n_s2, R_min, R_max, V_min, V_max,
                                    maxiter=40, move=0.2, verbose=False,
                                    random_init=True, seed=0, smooth=4.0):
        """Seed over filter radius (n_s1) x volume fraction (n_s2), as in the
        paper.  `random_init` additionally starts each LF solve from a smooth
        random density (a different basin per seed) to boost the topological
        diversity of the initial population beyond what (R,V) seeding alone
        gives; set False for the paper-faithful uniform-start behaviour."""
        s1 = np.linspace(0, 1, n_s1)
        s2 = np.linspace(0, 1, n_s2)
        rng = np.random.default_rng(seed)
        designs, info = [], []
        k = 0
        for a in s1:
            R = R_min + (R_max - R_min) * a
            H, Hs = build_density_filter(self.mesh, R=R)
            for b in s2:
                V = V_min + (V_max - V_min) * b
                x0 = self._random_density(V, rng, smooth) if random_init else None
                if self.lf_method == "stress":
                    rho, x = lf_optimize_stress(
                        self.mesh, self.fem, H, Hs, V, P=self.lf_P, q=self.lf_q,
                        x_init=x0, maxiter=maxiter, move=move, passive=self.passive)
                else:
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
        # P0 fix: the LF density lives on element CENTROIDS ([h/2, L-h/2]); the HF
        # node grid spans [0, L].  Edge-clamp the query points to the centroid
        # range (nearest-edge extrapolation) so the domain boundary -- including
        # the fixed support edge -- inherits the nearest interior density instead
        # of being zero-filled (which previously left the support unattached).
        g = np.asarray(gamma, float).reshape(self.grid_shape)
        interp = RegularGridInterpolator((self._ey, self._ex), g,
                                         bounds_error=False, fill_value=None)
        qy = np.clip(self.yn.ravel(), self._ey[0], self._ey[-1])
        qx = np.clip(self.xn.ravel(), self._ex[0], self._ex[-1])
        field = interp(np.column_stack([qy, qx])).reshape(self.xn.shape)
        field[bf.passive_void_mask(self.xn, self.yn, self.lpd, self.r_fillet)] = 0.0
        return np.clip(field, 0.0, 1.0)

    # ---- HF: body-fitted true max von Mises stress + volume fraction ----
    #  averaged over several mesh seeds to suppress mesh-to-mesh noise.
    def hf_evaluate(self, gamma):
        field = self._to_hf_field(gamma)
        j1s, j2s = [], []
        for s in range(self.hf_seeds):
            try:
                J1, J2 = bf.hf_lbracket_stress(field, self.xn, self.yn,
                                               geom=dict(self.hf_geom), seed=s,
                                               n_iter=self._hf_iter)
                if np.isfinite(J1) and J2 > 1e-3:
                    j1s.append(J1); j2s.append(J2)
            except Exception:
                continue
        if not j1s:
            return np.array([np.inf, np.inf]), False, None
        return np.array([np.mean(j1s), np.mean(j2s)]), True, None
