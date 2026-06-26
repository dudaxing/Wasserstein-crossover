"""
Persistent-homology + Wasserstein-distance diversity backend (paper-faithful).

This implements the *second stage* of the paper's selection (Kii et al. [59]):
within a Pareto rank, preserve the intrinsic diversity of material distributions
by (1) extracting topological features of each design via persistent homology,
and (2) measuring pairwise differences with the Wasserstein distance between the
resulting persistence diagrams.

It uses **torch_topological** for both steps, exactly as suggested:
    from torch_topological.nn import CubicalComplex      # persistent homology
    from torch_topological.nn import WassersteinDistance # diagram Wasserstein

Notes / environment:
  * torch_topological requires PyTorch + POT, which currently have no Python-3.14
    wheels, so this backend runs in the dedicated TDA environment (Python 3.13;
    see requirements-tda.txt).  The rest of the project still runs on 3.14 with
    the default L2 diversity; this module is imported lazily only when
    ``sel_mode="ph_wasserstein"`` is requested.
  * Two small, isolated compatibility shims are applied here (they do NOT modify
    torch_topological):
      - stub ``gph`` (giotto-ph) so ``torch_topological.nn``'s package __init__
        imports without that compiled dependency (we never use VietorisRips);
      - wrap ``gudhi.CubicalComplex`` so a torch.Tensor argument from
        torch_topological 0.1.9 is converted to numpy for current gudhi (>=3.11).
        This only affects the (detached) index-pairing computation.
  * The persistence filtration is the **signed distance function (SDF)** of the
    binarized design, so features carry geometric scale (hole size, bar
    thickness) -- a physically meaningful filtration for topology optimization.
"""
from __future__ import annotations
import sys
import types
import warnings
import numpy as np
from scipy.ndimage import distance_transform_edt

_BACKEND = None  # cached (torch, CubicalComplex instance, WassersteinDistance instance)


def available() -> bool:
    """True if the torch_topological backend can be loaded."""
    try:
        _load()
        return True
    except Exception:
        return False


def _load():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # (1) stub giotto-ph: only VietorisRips needs it, and we do not use that.
    if "gph" not in sys.modules:
        m = types.ModuleType("gph")
        m.ripser_parallel = None
        sys.modules["gph"] = m

    import torch
    import gudhi

    # (2) gudhi>=3.11 compatibility: torch_topological 0.1.9 passes a torch.Tensor
    #     to gudhi.CubicalComplex(top_dimensional_cells=...); convert to numpy.
    if not getattr(gudhi.CubicalComplex, "_wc_compat", False):
        _orig = gudhi.CubicalComplex

        def _cc(*args, **kwargs):
            v = kwargs.get("top_dimensional_cells", None)
            if isinstance(v, torch.Tensor):
                kwargs["top_dimensional_cells"] = v.detach().cpu().numpy()
            if "dimensions" in kwargs:
                kwargs["dimensions"] = list(kwargs["dimensions"])
            return _orig(*args, **kwargs)

        _cc._wc_compat = True
        gudhi.CubicalComplex = _cc

    from torch_topological.nn import CubicalComplex, WassersteinDistance

    cc = CubicalComplex(superlevel=True, dim=2)
    wd = WassersteinDistance(q=2)
    _BACKEND = (torch, cc, wd)
    return _BACKEND


def _sdf(binary: np.ndarray) -> np.ndarray:
    """Signed distance: + inside material, - outside (feature-scale filtration)."""
    b = binary.astype(bool)
    return (distance_transform_edt(b) - distance_transform_edt(~b)).astype(np.float32)


def persistence_diagrams(population, grid_shape, thresh=0.5):
    """Cubical persistent homology (via torch_topological) of each design.

    Returns a list of per-design ``PersistenceInformation`` lists (one entry per
    homology dimension).
    """
    torch, cc, _ = _load()
    pis = []
    for g in population:
        b = (np.asarray(g, dtype=float).reshape(grid_shape) > thresh).astype(np.float32)
        field = torch.as_tensor(_sdf(b))
        pis.append(cc(field))
    return pis


def ph_distance_matrix(population, grid_shape):
    """Pairwise Wasserstein distance between persistence diagrams of the designs.

    Uses ``torch_topological.nn.WassersteinDistance`` (which aggregates across
    homology dimensions).  Returns a symmetric (M, M) numpy matrix.
    """
    torch, cc, wd = _load()
    pis = persistence_diagrams(population, grid_shape)
    M = len(pis)
    D = np.zeros((M, M))
    # WassersteinDistance uses POT internally; empty/degenerate diagrams (e.g. a
    # design with no holes -> empty H1) can trigger benign POT warnings.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(M):
            for j in range(i + 1, M):
                D[i, j] = D[j, i] = float(wd(pis[i], pis[j]))
    return D
