"""
2D density-based topology optimization (low-fidelity model of the paper).

Implements the components needed for Section 5.1 (2D cracked-plate stress
minimization), used to generate the diverse LF-optimized initial population:

  * Q4 bilinear plane-stress finite element analysis (self-consistent FEM).
  * Modified SIMP stiffness interpolation.
  * Relaxed elemental von Mises stress (q-relaxation to avoid the singularity).
  * P-norm stress aggregation  J = (sum_e sigma_e^P)^(1/P)   (paper Eq. 21, P=8).
  * Linear "hat" density filter (paper Eqs. 14-16).
  * Adjoint sensitivities of the P-norm stress and of the volume constraint.
  * LF optimization driver using MMA (paper Section 4.1, move limit 0.05).

The cracked-plate geometry (Fig. 4a): a 2x2 plate in horizontal tension on the
top strips, symmetric about the vertical centerline. We model the RIGHT HALF
(width 1, height 2): symmetry u_x = 0 on the lower half of the left edge, a free
crack on the upper half, a pin (u_y = 0) at the bottom-center, and a horizontal
traction on the top 0.1 strip of the right edge.  Stress concentrates at the
crack tip (0, 1).
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mma import mmasub


# --------------------------------------------------------------------------- #
#  Element matrices (square Q4, unit Young's modulus, plane stress)
# --------------------------------------------------------------------------- #
def _plane_stress_D(nu=0.3):
    return (1.0 / (1.0 - nu ** 2)) * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, (1.0 - nu) / 2.0]])


def _q4_matrices(h=1.0, nu=0.3):
    """Return (KE, B_centroid, D) for a square Q4 element of size h, E=1.

    Local node order (CCW): 0=BL, 1=BR, 2=TR, 3=TL.  DOFs [u0,v0,...,u3,v3].
    """
    D = _plane_stress_D(nu)
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    # node local coords
    xn = np.array([-1, 1, 1, -1]) * (h / 2.0)
    yn = np.array([-1, -1, 1, 1]) * (h / 2.0)

    def B_at(xi, eta):
        # shape function derivatives wrt xi, eta
        dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
        dN_deta = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
        J = np.array([[dN_dxi @ xn, dN_dxi @ yn],
                      [dN_deta @ xn, dN_deta @ yn]])
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        dN_xy = invJ @ np.vstack([dN_dxi, dN_deta])  # 2x4: rows d/dx, d/dy
        B = np.zeros((3, 8))
        B[0, 0::2] = dN_xy[0]
        B[1, 1::2] = dN_xy[1]
        B[2, 0::2] = dN_xy[1]
        B[2, 1::2] = dN_xy[0]
        return B, detJ

    KE = np.zeros((8, 8))
    for (xi, eta) in pts:
        B, detJ = B_at(xi, eta)
        KE += B.T @ D @ B * detJ  # weights are 1 for 2x2 Gauss
    Bc, _ = B_at(0.0, 0.0)
    return KE, Bc, D


# von Mises matrix:  sigma_vm^2 = s^T V s,   s=[sx,sy,txy]
_VM = np.array([[1.0, -0.5, 0.0],
                [-0.5, 1.0, 0.0],
                [0.0, 0.0, 3.0]])


# --------------------------------------------------------------------------- #
#  Mesh / DOF bookkeeping
# --------------------------------------------------------------------------- #
class Mesh:
    def __init__(self, nelx, nely, h):
        self.nelx, self.nely, self.h = nelx, nely, h
        self.nel = nelx * nely
        self.nnx, self.nny = nelx + 1, nely + 1
        self.nnode = self.nnx * self.nny
        self.ndof = 2 * self.nnode
        # element -> 8 global dofs
        edof = np.zeros((self.nel, 8), dtype=int)
        ecoord = np.zeros((self.nel, 2))
        e = 0
        for ey in range(nely):
            for ex in range(nelx):
                n0 = ey * self.nnx + ex          # BL
                n1 = ey * self.nnx + ex + 1      # BR
                n2 = (ey + 1) * self.nnx + ex + 1  # TR
                n3 = (ey + 1) * self.nnx + ex    # TL
                nodes = [n0, n1, n2, n3]
                dofs = []
                for nd in nodes:
                    dofs += [2 * nd, 2 * nd + 1]
                edof[e] = dofs
                ecoord[e] = [(ex + 0.5) * h, (ey + 0.5) * h]
                e += 1
        self.edof = edof
        self.ecoord = ecoord  # element centroid coords

    def node_id(self, ix, iy):
        return iy * self.nnx + ix

    def node_coords(self):
        xs = np.arange(self.nnx) * self.h
        ys = np.arange(self.nny) * self.h
        X, Y = np.meshgrid(xs, ys)
        return X.ravel(), Y.ravel()


def make_cracked_plate_mesh(nelx):
    """Right-half cracked plate: width 1, height 2, square elements.

    nely = 2*nelx, h = 1/nelx so the physical domain is exactly [0,1]x[0,2],
    matching the absolute coordinates used in `cracked_plate_bc`.
    """
    h = 1.0 / nelx
    return Mesh(nelx, 2 * nelx, h)


# --------------------------------------------------------------------------- #
#  Density (hat) filter  -- paper Eqs. (14)-(16)
# --------------------------------------------------------------------------- #
def build_density_filter(mesh: Mesh, R):
    """Linear hat filter matrix H (sparse) and row sums Hs, on element centroids.

    w_ee' = 1 - ||x_e - x_e'|| / R  for ||.|| <= R, else 0.
    """
    nelx, nely, h = mesh.nelx, mesh.nely, mesh.h
    rad = int(np.ceil(R / h))
    iH, jH, sH = [], [], []
    coords = mesh.ecoord
    # element grid index
    def eidx(ex, ey):
        return ey * nelx + ex
    for ey in range(nely):
        for ex in range(nelx):
            e = eidx(ex, ey)
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    nx, ny = ex + dx, ey + dy
                    if 0 <= nx < nelx and 0 <= ny < nely:
                        f = eidx(nx, ny)
                        dist = np.hypot(coords[e, 0] - coords[f, 0],
                                        coords[e, 1] - coords[f, 1])
                        w = R - dist
                        if w > 0:
                            iH.append(e); jH.append(f); sH.append(w)
    H = sp.csr_matrix((sH, (iH, jH)), shape=(mesh.nel, mesh.nel))
    Hs = np.array(H.sum(axis=1)).ravel()
    return H, Hs


def apply_filter(H, Hs, x):
    return np.asarray(H @ x).ravel() / Hs


def filter_chain(H, Hs, dfdxphys):
    """Back-propagate sensitivities from physical (filtered) to design field."""
    return np.asarray(H @ (dfdxphys / Hs)).ravel()


# --------------------------------------------------------------------------- #
#  Boundary conditions for the cracked plate (right-half model)
# --------------------------------------------------------------------------- #
def cracked_plate_bc(mesh: Mesh, load=1.0):
    """Return (fixed_dofs, F) for the right-half cracked-plate problem.

    Geometry: x in [0,1] (width 1), y in [0,2] (height 2).
      * symmetry u_x = 0 on left edge (x=0) for y in [0,1];
      * pin u_y = 0 at bottom-left corner (0,0);
      * horizontal traction (+x) on the top strip y in [1.9, 2.0] of right edge.
    """
    h = mesh.h
    fixed = []
    # symmetry: left edge nodes with y <= 1.0  -> fix u_x
    iy_sym = int(round(1.0 / h))
    for iy in range(0, iy_sym + 1):
        nd = mesh.node_id(0, iy)
        fixed.append(2 * nd)            # u_x = 0
    # pin bottom-left corner u_y
    nd0 = mesh.node_id(0, 0)
    fixed.append(2 * nd0 + 1)           # u_y = 0
    fixed = np.unique(np.array(fixed, dtype=int))

    # load: horizontal traction on right edge top strip y in [1.9, 2.0]
    F = np.zeros(mesh.ndof)
    iy_lo = int(round(1.9 / h))
    iy_hi = mesh.nny - 1
    strip_nodes = [mesh.node_id(mesh.nnx - 1, iy) for iy in range(iy_lo, iy_hi + 1)]
    # consistent nodal forces for a uniform edge traction: total = load
    nseg = len(strip_nodes) - 1
    fval = load / nseg
    for k, nd in enumerate(strip_nodes):
        w = 1.0 if (0 < k < len(strip_nodes) - 1) else 0.5
        F[2 * nd] += fval * w           # +x direction
    return fixed, F


# --------------------------------------------------------------------------- #
#  FEM solve
# --------------------------------------------------------------------------- #
class FEM:
    def __init__(self, mesh: Mesh, fixed, F, nu=0.3, Emin=1e-9, E0=1.0, penal=3.0):
        self.mesh = mesh
        self.KE, self.Bc, self.D = _q4_matrices(mesh.h, nu)
        self.fixed = fixed
        self.free = np.setdiff1d(np.arange(mesh.ndof), fixed)
        self.F = F
        self.Emin, self.E0, self.penal = Emin, E0, penal
        # precompute sparse assembly indices
        edof = mesh.edof
        self.iK = np.kron(edof, np.ones((8, 1), dtype=int)).flatten()
        self.jK = np.kron(edof, np.ones((1, 8), dtype=int)).flatten()
        self.KE_flat = self.KE.flatten()
        # M = D^T V D for relaxed von Mises from solid stress tau = D B u
        self.M = self.D.T @ _VM @ self.D
        self.MB = self.Bc.T @ self.M @ self.Bc   # 8x8: u_e^T MB u_e = g_e^2

    def youngs(self, rho):
        return self.Emin + rho ** self.penal * (self.E0 - self.Emin)

    def solve(self, rho):
        E = self.youngs(rho)
        sK = (self.KE_flat[None, :] * E[:, None]).flatten()
        K = sp.csc_matrix((sK, (self.iK, self.jK)),
                          shape=(self.mesh.ndof, self.mesh.ndof))
        K = (K + K.T) * 0.5
        U = np.zeros(self.mesh.ndof)
        Kff = K[self.free][:, self.free]
        U[self.free] = spla.spsolve(Kff.tocsc(), self.F[self.free])
        return U, K

    # ---- compliance (for validation) ----
    def compliance(self, rho):
        U, K = self.solve(rho)
        ce = np.einsum('ei,ij,ej->e', U[self.mesh.edof], self.KE, U[self.mesh.edof])
        c = float(self.F @ U)
        dc = -self.penal * rho ** (self.penal - 1) * (self.E0 - self.Emin) * ce
        return c, dc, U

    # ---- relaxed elemental von Mises stress ----
    def vm_stress(self, rho, q=0.5):
        U, K = self.solve(rho)
        ue = U[self.mesh.edof]                       # (nel,8)
        g2 = np.einsum('ei,ij,ej->e', ue, self.MB, ue)
        g = np.sqrt(np.maximum(g2, 0.0))             # solid von Mises
        sigma = rho ** q * g                         # relaxed von Mises
        return sigma, g, U, K, ue

    # ---- P-norm stress + adjoint sensitivity wrt physical density rho ----
    def pnorm_stress(self, rho, P=8.0, q=0.5):
        sigma, g, U, K, ue = self.vm_stress(rho, q)
        S = np.sum(sigma ** P)
        S = max(S, 1e-30)
        J = S ** (1.0 / P)

        # dJ/dS
        dJdS = (1.0 / P) * S ** (1.0 / P - 1.0)
        # explicit part: dS/drho|_explicit = qP rho^(qP-1) g^P
        dS_expl = q * P * rho ** (q * P - 1.0) * g ** P
        # adjoint: dS/du = sum_e rho^(qP) P g^(P-2) MB u_e   (scatter to global)
        coef = rho ** (q * P) * P * np.where(g > 1e-30, g ** (P - 2.0), 0.0)
        contrib = (self.MB @ ue.T).T * coef[:, None]   # (nel,8)
        dSdu = np.zeros(self.mesh.ndof)
        np.add.at(dSdu, self.mesh.edof.flatten(), contrib.flatten())
        # solve adjoint K lam = dSdu
        lam = np.zeros(self.mesh.ndof)
        Kff = K[self.free][:, self.free]
        lam[self.free] = spla.spsolve(Kff.tocsc(), dSdu[self.free])
        # implicit part: -lam^T dK/drho u = -(dE/drho)(lam_e^T KE u_e)
        lame = lam[self.mesh.edof]
        lamKu = np.einsum('ei,ij,ej->e', lame, self.KE, ue)
        dEdrho = self.penal * rho ** (self.penal - 1.0) * (self.E0 - self.Emin)
        dS_impl = -dEdrho * lamKu

        dJdrho = dJdS * (dS_expl + dS_impl)
        return J, dJdrho, sigma, U

    # ---- true max von Mises stress (HF-substitute objective) ----
    def max_stress(self, rho, q=0.5):
        sigma, g, U, K, ue = self.vm_stress(rho, q)
        return float(np.max(sigma)), sigma, U


# --------------------------------------------------------------------------- #
#  Low-fidelity optimization driver:  min P-norm stress s.t. volume <= V
# --------------------------------------------------------------------------- #
def lf_optimize_stress(mesh, fem, H, Hs, V, P=8.0, q=0.5,
                       x_init=None, maxiter=80, move=0.05, tol=1e-3,
                       passive=None, verbose=False):
    """Density-based stress minimization with a volume constraint via MMA.

    minimize  (sum_e sigma_e^P)^(1/P)
    s.t.      vol_fraction(design region) - V <= 0,  0<=gamma<=1.

    `passive` (bool mask) forces those elements to void (rho=0); the volume
    constraint and seeding are taken over the *design* (non-passive) region.
    Returns (rho_phys, x) flattened.
    """
    n = mesh.nel
    if passive is None:
        passive = np.zeros(n, dtype=bool)
    design = ~passive
    ndesign = max(int(design.sum()), 1)
    if x_init is None:
        x = np.zeros(n)
        x[design] = V
    else:
        x = x_init.copy()
    x[passive] = 0.0
    xmin = np.zeros((n, 1))
    xmax = np.ones((n, 1))
    xmax[passive, 0] = 1e-6              # pin passive elements to (near) void
    xold1 = x.reshape(-1, 1).copy()
    xold2 = x.reshape(-1, 1).copy()
    low = xmin.copy()
    upp = xmax.copy()
    m = 1
    a0, a = 1.0, np.zeros((m, 1))
    c = 1e3 * np.ones((m, 1))
    d = np.zeros((m, 1))

    # normalize objective by its initial value for numerical conditioning
    rho0 = apply_filter(H, Hs, x); rho0[passive] = 0.0
    J0, _, _, _ = fem.pnorm_stress(rho0, P, q)
    J0 = max(J0, 1e-12)

    Jprev = None
    for it in range(maxiter):
        rho = apply_filter(H, Hs, x)
        rho[passive] = 0.0                       # enforce passive void
        J, dJdrho, sigma, U = fem.pnorm_stress(rho, P, q)
        f0 = J / J0
        df0 = filter_chain(H, Hs, dJdrho) / J0

        vol = rho[design].sum() / ndesign        # volume fraction over design region
        g1 = vol / V - 1.0
        dg1 = filter_chain(H, Hs, design.astype(float) / (ndesign * V))

        xmma, *_rest, low, upp = mmasub(
            m, n, it + 1, x.reshape(-1, 1), xmin, xmax, xold1, xold2,
            f0, df0.reshape(-1, 1), np.array([[g1]]), dg1.reshape(1, n),
            low, upp, a0, a, c, d, move=move)
        xold2 = xold1
        xold1 = x.reshape(-1, 1).copy()
        x = xmma.ravel()

        change = 1.0 if Jprev is None else abs(J - Jprev) / max(abs(J), 1e-12)
        Jprev = J
        if verbose and (it % 5 == 0 or it == maxiter - 1):
            print(f"   it={it:3d}  J(pnorm)={J:9.4f}  vol={vol:6.3f}  ch={change:.4f}")
        if it > 10 and change < tol:
            break
    rho_final = apply_filter(H, Hs, x); rho_final[passive] = 0.0
    return rho_final, x  # return physical and design fields
