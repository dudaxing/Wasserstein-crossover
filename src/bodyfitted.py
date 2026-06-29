"""
Body-fitted-mesh stress evaluation (high-fidelity model), ported to Python from
the DPTO method:

    Z. Zhuang, Y. Xiong, Y. He, Y.M. Xie, "A novel topology optimization method
    for enhanced stress distribution using density projection and body-fitted
    mesh", Engineering Structures 349 (2026) 121854.  (DPTO_STR.m)

This module provides the pieces needed to evaluate the *true* maximum von Mises
stress of a material layout on a boundary-conforming triangular mesh:

  * linear constant-strain triangle (CST) plane-stress FEA  (this file, part 1);
  * 0.5 iso-contour extraction + cleaning                    (part 2);
  * body-fitted mesh generation (rejection + DistMesh)       (part 3);
  * L-bracket HF evaluation: density -> contour -> mesh -> FEA -> max vM, volume.

Part 1 (FEA) is implemented and patch-test verified first; later parts build on
it.  All linear-elastic, isotropic, plane stress.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------- #
#  Part 1: linear constant-strain triangle (CST) plane-stress FEA
#  (port of ElementMatrixKe / FEA in DPTO_STR.m)
# --------------------------------------------------------------------------- #
def elasticity_matrix(E0=1.0, nu=0.3):
    return E0 / (1 - nu ** 2) * np.array([[1.0, nu, 0.0],
                                          [nu, 1.0, 0.0],
                                          [0.0, 0.0, (1 - nu) / 2.0]])


def cst_BkeS(coords, D):
    """Strain matrix B (3x6), stiffness Ke (6x6), stress matrix S=D*B (3x6),
    and signed area, for one constant-strain triangle.

    `coords` is a (3,2) array of node coordinates; DOF order [u1,v1,u2,v2,u3,v3].
    Exactly mirrors DPTO's ElementMatrixKe.
    """
    X = coords[:, 0]
    Y = coords[:, 1]
    J = np.array([[X[0] - X[2], Y[0] - Y[2]],
                  [X[1] - X[2], Y[1] - Y[2]]])
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    J11, J12, J21, J22 = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    B = (1.0 / detJ) * np.array([
        [J22, 0, -J12, 0, -J22 + J12, 0],
        [0, -J21, 0, J11, 0, J21 - J11],
        [-J21, J22, J11, -J12, J21 - J11, -J22 + J12]])
    Ke = 0.5 * detJ * (B.T @ D @ B)
    S = D @ B
    area = 0.5 * detJ
    return B, Ke, S, area


def _edofs(t):
    """(NT,6) DOF map for triangles `t` (NT,3); node n -> dofs [2n, 2n+1]."""
    e = np.empty((len(t), 6), dtype=int)
    e[:, [0, 2, 4]] = 2 * t
    e[:, [1, 3, 5]] = 2 * t + 1
    return e


def fea_t3(p, t, rho, fixed_dofs, F, E0=1.0, Emin=1e-9, nu=0.3, q_relax=0.0,
           alpha=0.03):
    """Solve plane-stress FEA on a triangular mesh and return per-element von
    Mises stress.

    Parameters
    ----------
    p : (NP,2) node coordinates
    t : (NT,3) triangle node indices
    rho : (NT,) element densities in [0,1] (1=solid, ~0=void)
    fixed_dofs : array of constrained global DOF indices
    F : (2*NP,) global load vector
    q_relax : if >0, multiply the *reported* von Mises by rho**q_relax (kept 0
              for a near-binary body-fitted design); the DPTO relaxation
              rho/(rho+alpha*(1-rho)) is also available via `alpha`.

    Returns
    -------
    U : (2*NP,) displacements
    vm : (NT,) von Mises stress per element (solid-material stress)
    sig : (NT,3) [sx, sy, txy] per element
    area : (NT,) element areas
    """
    p = np.asarray(p, float)
    t = np.asarray(t, int)
    NP, NT = len(p), len(t)
    D = elasticity_matrix(E0, nu)
    edof = _edofs(t)

    # linear material interpolation E_e = Emin + rho*(E0-Emin)  (DPTO uses x*Ke)
    Evec = Emin + rho * (E0 - Emin)

    iK = np.empty(36 * NT); jK = np.empty(36 * NT); sK = np.empty(36 * NT)
    Ke_list = np.empty((NT, 6, 6)); S_list = np.empty((NT, 3, 6))
    area = np.empty(NT)
    for e in range(NT):
        _, Ke, S, a = cst_BkeS(p[t[e]], D)
        Ke_list[e] = Ke; S_list[e] = S; area[e] = a
        ed = edof[e]
        iK[36 * e:36 * e + 36] = np.repeat(ed, 6)
        jK[36 * e:36 * e + 36] = np.tile(ed, 6)
        sK[36 * e:36 * e + 36] = (Evec[e] * Ke).ravel()
    K = sp.csc_matrix((sK, (iK, jK)), shape=(2 * NP, 2 * NP))
    K = 0.5 * (K + K.T)

    alldof = np.arange(2 * NP)
    free = np.setdiff1d(alldof, np.asarray(fixed_dofs, int))
    U = np.zeros(2 * NP)
    U[free] = spla.spsolve(K[free][:, free].tocsc(), np.asarray(F, float)[free])

    # per-element stress (solid-material stress: S @ u_e) and von Mises
    Ue = U[edof]                                  # (NT,6)
    sig = np.einsum('eij,ej->ei', S_list, Ue)     # (NT,3) [sx,sy,txy]
    vm = np.sqrt(sig[:, 0] ** 2 + sig[:, 1] ** 2
                 - sig[:, 0] * sig[:, 1] + 3 * sig[:, 2] ** 2)
    if q_relax > 0:
        vm = rho ** q_relax * vm
    return U, vm, sig, area


# --------------------------------------------------------------------------- #
#  Patch test: a linear displacement field must be reproduced exactly by CST,
#  giving a constant, exact stress state on every element.
# --------------------------------------------------------------------------- #
def _patch_test(verbose=True):
    import scipy.spatial
    rng = np.random.default_rng(0)
    # random points in the unit square + corners, Delaunay triangulated
    pts = np.vstack([[[0, 0], [1, 0], [1, 1], [0, 1]], rng.random((25, 2))])
    tri = scipy.spatial.Delaunay(pts)
    p, t = pts, tri.simplices
    E0, nu = 200.0, 0.3
    # exact linear displacement field  u = a x + b y,  v = c x + d y
    a, b, c, d = 1e-3, 2e-4, -3e-4, 5e-4
    uex = a * p[:, 0] + b * p[:, 1]
    vex = c * p[:, 0] + d * p[:, 1]
    # constant strain -> constant stress (analytical)
    eps = np.array([a, d, b + c])               # [ex, ey, gxy]
    sig_exact = elasticity_matrix(E0, nu) @ eps
    vm_exact = np.sqrt(sig_exact[0] ** 2 + sig_exact[1] ** 2
                       - sig_exact[0] * sig_exact[1] + 3 * sig_exact[2] ** 2)

    # prescribe the exact field on boundary nodes, solve interior
    on_b = (np.isclose(p[:, 0], 0) | np.isclose(p[:, 0], 1)
            | np.isclose(p[:, 1], 0) | np.isclose(p[:, 1], 1))
    bnodes = np.where(on_b)[0]
    NP = len(p)
    # build a system with prescribed boundary dofs via penalty-free substitution:
    D = elasticity_matrix(E0, nu)
    edof = _edofs(t)
    rows = []; cols = []; vals = []
    for e in range(len(t)):
        _, Ke, _, _ = cst_BkeS(p[t[e]], D)
        ed = edof[e]
        for i in range(6):
            for j in range(6):
                rows.append(ed[i]); cols.append(ed[j]); vals.append(Ke[i, j])
    K = sp.csc_matrix((vals, (rows, cols)), shape=(2 * NP, 2 * NP))
    U = np.zeros(2 * NP)
    U[2 * bnodes] = uex[bnodes]
    U[2 * bnodes + 1] = vex[bnodes]
    presc = np.concatenate([2 * bnodes, 2 * bnodes + 1])
    free = np.setdiff1d(np.arange(2 * NP), presc)
    rhs = -K[free][:, presc] @ U[presc]
    U[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)

    # recover element stresses
    rho = np.ones(len(t))
    _, vm, sig, _ = _fea_recover(p, t, U, E0, nu)
    err_u = max(np.max(np.abs(U[0::2] - uex)), np.max(np.abs(U[1::2] - vex)))
    err_s = np.max(np.abs(sig - sig_exact[None, :]))
    if verbose:
        print(f"  patch test: max|U-Uexact|={err_u:.2e}, "
              f"max|sigma-exact|={err_s:.2e}, vm_exact={vm_exact:.4f}, "
              f"vm range=[{vm.min():.4f},{vm.max():.4f}]")
    assert err_u < 1e-9 and err_s < 1e-7, "CST patch test failed"
    return True


def _fea_recover(p, t, U, E0, nu):
    D = elasticity_matrix(E0, nu)
    edof = _edofs(t)
    S_list = np.empty((len(t), 3, 6))
    for e in range(len(t)):
        _, _, S, _ = cst_BkeS(p[t[e]], D)
        S_list[e] = S
    Ue = U[edof]
    sig = np.einsum('eij,ej->ei', S_list, Ue)
    vm = np.sqrt(sig[:, 0] ** 2 + sig[:, 1] ** 2
                 - sig[:, 0] * sig[:, 1] + 3 * sig[:, 2] ** 2)
    return None, vm, sig, None


# --------------------------------------------------------------------------- #
#  Part 2: 0.5 iso-contour extraction + cleaning
#  (port of contour(...) + ContourPoints in DPTO_STR.m)
# --------------------------------------------------------------------------- #
def extract_contour(xn, yn, field, level=0.5):
    """Return a list of (n,2) polylines of the `level` iso-contour of `field`
    on the node grid (xn, yn) (both 2D meshgrids, field same shape)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    cs = plt.contour(xn, yn, field, levels=[level])
    segs = [np.asarray(s) for s in cs.allsegs[0]] if cs.allsegs else []
    plt.close(fig)
    return segs


def clean_contour(segs, d1, d2):
    """Resample contour polylines to spacing roughly in [d1, d2]: drop points
    closer than d1, insert midpoints in gaps larger than d2 (port of
    ContourPoints' intent).  Returns a single (M,2) point array."""
    out = []
    for ck in segs:
        ck = np.asarray(ck, float)
        if len(ck) < 5:
            continue
        closed = np.linalg.norm(ck[0] - ck[-1]) < 1e-9
        # drop too-close points
        keep = [ck[0]]
        for pt in ck[1:]:
            if np.linalg.norm(pt - keep[-1]) >= d1:
                keep.append(pt)
        ck = np.array(keep)
        if len(ck) < 3:
            continue
        # insert midpoints in long gaps
        newpts = [ck[0]]
        for i in range(len(ck) - 1):
            if np.linalg.norm(ck[i + 1] - ck[i]) > d2:
                newpts.append(0.5 * (ck[i] + ck[i + 1]))
            newpts.append(ck[i + 1])
        ck = np.array(newpts)
        if closed:
            ck = ck[:-1]                       # drop duplicate closing point
        out.append(ck)
    if out:
        return np.vstack(out)
    return np.zeros((0, 2))


# --------------------------------------------------------------------------- #
#  Part 3: body-fitted mesh generation (rejection + DistMesh)
#  (port of GenerateMesh's body-fitted branch in DPTO_STR.m, L-bracket)
# --------------------------------------------------------------------------- #
def _nearest_dist(pts, ref):
    """min Euclidean distance from each row of `pts` to the set `ref`."""
    from scipy.spatial import cKDTree
    if len(ref) == 0:
        return np.full(len(pts), np.inf)
    return cKDTree(ref).query(pts, k=1)[0]


def lbracket_fixed_passive(BDY, lpd, lload, h, r_fillet=0.0):
    """Fixed boundary nodes for the L-bracket: the L outer perimeter plus the
    internal passive-void boundary (edges truncated by the fillet + the fillet
    arc), sampled at spacing h.  Guarantees straight edges and that BC/load nodes
    exist."""
    x0, y0 = BDY[0]; x1, y1 = BDY[1]
    def seg(val, lo, hi, axis):
        s = np.arange(lo, hi + 1e-9, h)
        col = np.full_like(s, val)
        return np.column_stack([col, s]) if axis == "x" else np.column_stack([s, col])
    parts = [
        seg(x0, y0, y1, "x"),                    # left edge x=0
        seg(y0, x0, x1, "y"),                    # bottom edge y=0
        seg(y1, x0, lpd, "y"),                   # top of vertical arm y=L, x<=lpd
        seg(x1, y0, lpd, "x"),                   # right of horizontal arm x=L, y<=lpd
        _internal_boundary(lpd, x1, r_fillet, h),  # internal void boundary (+ fillet)
    ]
    fp = np.vstack(parts)
    return np.unique(fp, axis=0)


def lbracket_bcs(p, BDY, lpd, lload, F0=1.0, h=1.0):
    """L-bracket BCs (port of DPTO FEA): fix the top edge of the vertical arm
    (y=L, x<=lpd) in both directions; apply a downward distributed load on the
    strip y=lpd, x in [L-lload, L].

    P0 fix: consistent nodal loads for a uniform traction F0/lload over the strip
    -- interior node gets (F0/lload)*h, end nodes (F0/lload)*h/2 -- so the TOTAL
    applied force is F0 regardless of the mesh spacing h (previously it scaled
    with the node count, e.g. -0.5 at h=2 vs -1.0 at h=1)."""
    x1, y1 = BDY[1]
    fixed_nodes = np.where((p[:, 0] <= lpd + 1e-9) & np.isclose(p[:, 1], y1))[0]
    fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
    F = np.zeros(2 * len(p))
    t = F0 / lload                                   # traction (force per length)
    fn = np.where((p[:, 0] < x1) & (p[:, 0] > x1 - lload) & np.isclose(p[:, 1], lpd))[0]
    fn1 = np.where(((np.isclose(p[:, 0], x1)) | (np.isclose(p[:, 0], x1 - lload)))
                   & np.isclose(p[:, 1], lpd))[0]
    F[2 * fn + 1] = -t * h                            # interior tributary length h
    F[2 * fn1 + 1] = -0.5 * t * h                     # end tributary length h/2
    return fixed_dofs, F, fixed_nodes, np.union1d(fn, fn1)


def generate_bodyfitted_mesh(contour_pts, BDY, lpd, lload, h, minedge, maxedge,
                             dens_interp=None, rng_vec=None, n_iter=200,
                             seed=0, r_fillet=0.0):
    """Body-fitted triangular mesh of the L-bracket design box conforming to the
    material contour.  Mirrors DPTO's GenerateMesh body-fitted branch.

    Parameters
    ----------
    contour_pts : (M,2) cleaned 0.5-contour points
    BDY : [[x0,y0],[x1,y1]] bounding box
    lpd, lload : passive-void and load-region sizes
    dens_interp : callable (pts(N,2)) -> density at those points (for solid/void
        classification of triangle centroids); if None all design triangles solid
    rng_vec : optional precomputed uniform vector for the rejection method
        (determinism); else drawn from `seed`

    Returns
    -------
    p : (NP,2) nodes,  t : (NT,3) triangles,
    pv, ps, pd : boolean masks over triangles (passive-void / passive-solid / design),
    rho : (NT,) element density (1 solid, ~0 void) honoring passive regions,
    centroids : (NT,2)
    """
    from scipy.spatial import Delaunay
    x0, y0 = BDY[0]; x1, y1 = BDY[1]
    fixed_passive = lbracket_fixed_passive(BDY, lpd, lload, h, r_fillet)

    # background hex-like point grid
    xs = np.arange(x0, x1 + 1e-9, h)
    ys = np.arange(y0, y1 + 1e-9, np.sqrt(3) / 2 * h)
    X, Y = np.meshgrid(xs, ys)
    X[1::2, :] += 0.5 * h
    pi = np.column_stack([X.ravel(), Y.ravel()])
    pi = pi[(pi[:, 0] <= x1) & (pi[:, 1] <= y1)]

    # remove contour points too close to the internal passive (void) boundary
    c = contour_pts
    if len(c):
        ib = _internal_boundary(lpd, x1, r_fillet, h)
        c = c[_nearest_dist(c, ib) > 1.2 * h]

    cset = np.vstack([fixed_passive, c]) if len(c) else fixed_passive
    cset = np.unique(cset, axis=0)

    # size function -> rejection of background points (finer near contour)
    d = _nearest_dist(pi, cset)
    r0 = 1.0 / np.minimum(np.maximum(minedge, d), maxedge) ** 2
    if rng_vec is None:
        rng_vec = np.random.default_rng(seed).random(len(pi))
    keep = rng_vec[:len(pi)] < (r0 / r0.max())

    corners = np.array([[x1, y0], [x0, y1], [x0, y0], [x1, y1]])
    pfix = np.unique(np.vstack([fixed_passive, c, corners]) if len(c)
                     else np.vstack([fixed_passive, corners]), axis=0)
    p = np.vstack([pfix, pi[keep]])
    p = np.unique(p, axis=0)
    n_fix = len(pfix)
    # ensure fixed nodes occupy the first rows (for pinning)
    # (rebuild p as [pfix; others not duplicating pfix])
    from scipy.spatial import cKDTree
    tree = cKDTree(pfix)
    others = pi[keep]
    far = tree.query(others, k=1)[0] > 1e-9
    p = np.vstack([pfix, others[far]])

    # DistMesh node-moving
    t = Delaunay(p).simplices
    edges = _unique_edges(t)
    p_prev = np.full_like(p, 1e9)
    L1 = None
    for it in range(n_iter):
        if np.max(np.sum((p - p_prev) ** 2, axis=1)) > 0.01 * h:
            t = Delaunay(p).simplices
            edges = _unique_edges(t)
            p_prev = p.copy()
            mid = 0.5 * (p[edges[:, 0]] + p[edges[:, 1]])
            dmid = _nearest_dist(mid, cset)
            L1 = np.minimum(np.maximum(minedge, dmid), maxedge)
        bars = p[edges[:, 0]] - p[edges[:, 1]]
        L = np.sqrt(np.sum(bars ** 2, axis=1)) + 1e-12
        L0 = 1.2 * L1 * np.sqrt(np.sum(L ** 2) / np.sum(L1 ** 2))
        f = np.maximum(L0 - L, 0.0) / L
        Fbar = f[:, None] * bars
        Fp = np.zeros_like(p)
        np.add.at(Fp, edges[:, 0], Fbar)
        np.add.at(Fp, edges[:, 1], -Fbar)
        Fp[:n_fix] = 0.0
        p = p + 0.2 * Fp
        p[:, 0] = np.clip(p[:, 0], x0, x1)
        p[:, 1] = np.clip(p[:, 1], y0, y1)
        if np.mean(np.sqrt(np.sum((0.2 * Fp) ** 2, axis=1))) / h < 0.01:
            break

    t = Delaunay(p).simplices
    cen = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
    pv = passive_void_mask(cen[:, 0], cen[:, 1], lpd, r_fillet)
    ps = (cen[:, 0] > x1 - lload) & (cen[:, 1] > lpd - 1) & (cen[:, 1] < lpd)
    pd = ~(pv | ps)
    rho = np.ones(len(t))
    rho[pv] = 0.0
    if dens_interp is not None:
        dE = dens_interp(cen)
        design_void = pd & (dE < 0.5)
        rho[design_void] = 0.0
    return p, t, pv, ps, pd, rho, cen


def _unique_edges(t):
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


# --------------------------------------------------------------------------- #
#  L-bracket high-fidelity stress evaluation (body-fitted)
# --------------------------------------------------------------------------- #
LBRACKET = dict(L=150.0, lpd=60.0, lload=6.0, h=1.0, minedge=2.0, maxedge=40.0,
                d1=0.5, d2=1.0, E0=1.0, Emin=1e-9, nu=0.3, F0=1.0, r_fillet=0.0)


def passive_void_mask(x, y, lpd, r_fillet=0.0):
    """Upper-right passive void, with the re-entrant corner (lpd,lpd) rounded by
    a fillet of radius `r_fillet` (material added in the corner wedge)."""
    x = np.asarray(x); y = np.asarray(y)
    base = (x > lpd) & (y > lpd)
    if r_fillet <= 0:
        return base
    cx, cy = lpd + r_fillet, lpd + r_fillet
    wedge = (x < cx) & (y < cy) & ((x - cx) ** 2 + (y - cy) ** 2 > r_fillet ** 2)
    return base & ~wedge


def _internal_boundary(lpd, L, r_fillet, h):
    """Void-domain internal boundary: the two L edges (truncated by the fillet)
    plus the fillet arc, sampled at spacing h."""
    pts = []
    yy = np.arange(lpd + r_fillet, L + 1e-9, h)
    pts.append(np.column_stack([np.full_like(yy, lpd), yy]))        # x=lpd edge
    xx = np.arange(lpd + r_fillet, L + 1e-9, h)
    pts.append(np.column_stack([xx, np.full_like(xx, lpd)]))        # y=lpd edge
    if r_fillet > 0:
        cx, cy = lpd + r_fillet, lpd + r_fillet
        th = np.linspace(np.pi, 1.5 * np.pi, max(4, int(round(r_fillet / h)) + 1))
        pts.append(np.column_stack([cx + r_fillet * np.cos(th),
                                    cy + r_fillet * np.sin(th)]))    # fillet arc
    return np.vstack(pts)


def hf_lbracket_stress(field, xn, yn, geom=None, rng_vec=None, seed=0,
                       return_mesh=False, n_iter=200):
    """Body-fitted high-fidelity evaluation for the L-bracket.

    field : (ny,nx) density on the node grid (xn,yn); 1=solid, 0=void.
    Returns (J1, J2) = (max von Mises over solid material, volume fraction),
    where volume fraction = solid area / L-shaped design-domain area.
    """
    g = dict(LBRACKET);
    if geom: g.update(geom)
    L, lpd, lload, h = g["L"], g["lpd"], g["lload"], g["h"]
    BDY = np.array([[0, 0], [L, L]], float)
    from scipy.interpolate import RegularGridInterpolator
    di = RegularGridInterpolator((yn[:, 0], xn[0, :]), field,
                                 bounds_error=False, fill_value=0.0)
    def dens_interp(pts):
        return di(np.column_stack([pts[:, 1], pts[:, 0]]))

    segs = extract_contour(xn, yn, field, 0.5)
    c = clean_contour(segs, g["d1"], g["d2"])
    p, t, pv, ps, pd, rho, cen = generate_bodyfitted_mesh(
        c, BDY, lpd, lload, h, g["minedge"], g["maxedge"],
        dens_interp=dens_interp, rng_vec=rng_vec, seed=seed, n_iter=n_iter,
        r_fillet=g.get("r_fillet", 0.0))
    # passive solid (load block) forced solid
    rho = rho.copy(); rho[ps] = 1.0
    fixed_dofs, F, fixed_nodes, load_nodes = lbracket_bcs(p, BDY, lpd, lload, g["F0"], h=h)
    solid = rho > 0.5
    # element areas (CST), computed from the mesh so J2 is available even when
    # the solve is skipped by the robustness guard below.
    v1 = p[t[:, 1]] - p[t[:, 0]]
    v2 = p[t[:, 2]] - p[t[:, 0]]
    area = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    design_area = L * L - (L - lpd) * (L - lpd)        # L-shape area
    J2 = float(area[solid].sum() / design_area) if solid.any() else 0.0

    # Robustness guard (prevents C-level solver segfaults on degenerate
    # offspring): only solve if the SOLID material connects the load to the
    # support through a single connected component.  Otherwise the global
    # stiffness is (near-)singular -> mark infeasible without solving.
    if not _load_path_ok(p, t, solid, fixed_nodes, load_nodes):
        J1 = float("inf")
        if return_mesh:
            return J1, J2, dict(p=p, t=t, vm=np.zeros(len(t)), rho=rho,
                                U=np.zeros(2 * len(p)), area=area, cen=cen)
        return J1, J2

    U, vm, sig, area = fea_t3(p, t, rho, fixed_dofs, F,
                              g["E0"], g["Emin"], g["nu"])
    J1 = float(vm[solid].max()) if solid.any() else float("inf")
    if return_mesh:
        return J1, J2, dict(p=p, t=t, vm=vm, rho=rho, U=U, area=area, cen=cen)
    return J1, J2


def _load_path_ok(p, t, solid, fixed_nodes, load_nodes):
    """True iff some fixed node and some load node lie in the same connected
    component of the solid sub-mesh (i.e. a real load path exists)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not solid.any():
        return False
    st = t[solid]
    ii = np.concatenate([st[:, 0], st[:, 1], st[:, 2]])
    jj = np.concatenate([st[:, 1], st[:, 2], st[:, 0]])
    NP = len(p)
    A = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(NP, NP))
    _, lab = connected_components(A + A.T, directed=False)
    sol = set(np.unique(st).tolist())
    fsup = [n for n in fixed_nodes if n in sol]
    fload = [n for n in load_nodes if n in sol]
    if not fsup or not fload:
        return False
    return not set(lab[fsup]).isdisjoint(set(lab[fload]))


if __name__ == "__main__":
    print("[bodyfitted] CST FEA self-test")
    _patch_test()
    print("  PASS")
