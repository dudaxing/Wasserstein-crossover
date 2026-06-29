"""
Validation for the body-fitted-mesh stress HF model (src/bodyfitted.py),
ported from DPTO (Zhuang et al., Eng. Struct. 2026).

  1. CST plane-stress patch test (exact constant-stress reproduction);
  2. body-fitted mesh quality (DistMesh -> well-shaped triangles);
  3. L-bracket physics: max von Mises lands at the re-entrant corner.

Run:  python experiments/test_bodyfitted.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import bodyfitted as bf  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def _lbracket_field(L=150, lpd=60, h=1.0, holes=()):
    xn, yn = np.meshgrid(np.arange(0, L + 1, h), np.arange(0, L + 1, h))
    field = np.ones_like(xn, float)
    field[(xn > lpd) & (yn > lpd)] = 0.0
    for (cx, cy, r) in holes:
        field[(xn - cx) ** 2 + (yn - cy) ** 2 < r ** 2] = 0.0
    return xn, yn, field


def test_patch():
    print("[CST patch test]")
    bf._patch_test(verbose=True)
    print("  ok: CST reproduces constant stress exactly")


def test_mesh_quality():
    print("[body-fitted mesh quality]")
    xn, yn, field = _lbracket_field(holes=[(95, 30, 16)])
    from scipy.interpolate import RegularGridInterpolator
    di = RegularGridInterpolator((yn[:, 0], xn[0, :]), field,
                                 bounds_error=False, fill_value=0.0)
    segs = bf.extract_contour(xn, yn, field, 0.5)
    c = bf.clean_contour(segs, 0.5, 1.0)
    BDY = np.array([[0, 0], [150, 150]], float)
    p, t, pv, ps, pd, rho, cen = bf.generate_bodyfitted_mesh(
        c, BDY, 60, 6, 1.0, 2.0, 40.0,
        dens_interp=lambda q: di(np.column_stack([q[:, 1], q[:, 0]])), seed=0)
    # min interior angle of every triangle
    a, b, cc = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    def angle(u, v):
        cv = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-12)
        return np.degrees(np.arccos(np.clip(cv, -1, 1)))
    A = angle(b - a, cc - a); B = angle(a - b, cc - b); C = 180 - A - B
    mn = np.minimum(np.minimum(A, B), C)
    print(f"  min-angle: median={np.median(mn):.1f}, 1st pct={np.percentile(mn,1):.1f}, "
          f"<15deg frac={(mn<15).mean():.3f}")
    check(np.median(mn) > 35, "median min-angle > 35 deg (well-shaped)")
    check((mn < 15).mean() < 0.02, "<2% sliver triangles below 15 deg")
    check(len(t) > 2000, "mesh has a reasonable number of triangles")


def test_lbracket_physics():
    print("[L-bracket stress physics]")
    xn, yn, field = _lbracket_field()              # solid L
    J1, J2, mesh = bf.hf_lbracket_stress(field, xn, yn, return_mesh=True, seed=0)
    solid = mesh["rho"] > 0.5
    vm = mesh["vm"].copy(); vm[~solid] = 0
    cen = mesh["cen"][np.argmax(vm)]
    d_corner = np.hypot(cen[0] - 60, cen[1] - 60)
    print(f"  J1={J1:.4f}, J2={J2:.4f}, max-stress at ({cen[0]:.1f},{cen[1]:.1f}), "
          f"dist to re-entrant corner={d_corner:.1f}")
    check(np.isfinite(J1) and J1 > 0, "finite positive max von Mises")
    check(abs(J2 - 1.0) < 0.02, "solid L -> volume fraction ~ 1")
    check(d_corner < 4.0, "max stress at the re-entrant corner (60,60)")


if __name__ == "__main__":
    test_patch()
    test_mesh_quality()
    test_lbracket_physics()
    print("\nALL BODY-FITTED HF TESTS PASSED")
