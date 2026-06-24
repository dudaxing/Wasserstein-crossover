"""
Multi-objective selection and the hypervolume indicator.

Reproduces the "Selection" and "Convergence check" steps of the framework
(paper Section 3):

  * Non-dominated sorting (NSGA-II, Deb et al. [60,61]) -- the first stage of
    the paper's two-stage ranking.
  * Within-rank ordering.  The paper's second stage uses a persistent-homology
    based "Wasserstein distance sorting" of material distributions (Kii et al.
    [59]) to preserve the *intrinsic diversity of the material distributions*.
    Persistent-homology libraries are not assumed here, so we provide two
    faithful substitutes that pursue the same goal:
        - "crowding"  : classic NSGA-II crowding distance (objective space);
        - "diversity" : farthest-point ordering in design (density) space,
                        which directly preserves geometric diversity of the
                        material distributions (closest in spirit to [59]).
  * Hypervolume indicator (paper Eq. 9) for the convergence check.

All objectives are MINIMIZED.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- #
#  Pareto dominance & non-dominated sorting
# --------------------------------------------------------------------------- #
def dominates(a, b):
    """True if a dominates b (minimization): a<=b in all, a<b in at least one."""
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F):
    """Return a list of fronts; each front is a list of indices into F.

    F : (M, n_obj) array of objective values (minimization).
    """
    M = len(F)
    S = [[] for _ in range(M)]
    n = np.zeros(M, dtype=int)
    fronts = [[]]
    for p in range(M):
        for q in range(M):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


# --------------------------------------------------------------------------- #
#  Within-rank ordering
# --------------------------------------------------------------------------- #
def crowding_distance(F, front):
    """NSGA-II crowding distance for the points indexed by `front`."""
    l = len(front)
    if l == 0:
        return np.array([])
    if l <= 2:
        return np.full(l, np.inf)
    Ff = F[front]
    dist = np.zeros(l)
    n_obj = Ff.shape[1]
    for m in range(n_obj):
        order = np.argsort(Ff[:, m])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = Ff[order[0], m], Ff[order[-1], m]
        if fmax - fmin < 1e-30:
            continue
        for k in range(1, l - 1):
            dist[order[k]] += (Ff[order[k + 1], m] - Ff[order[k - 1], m]) / (fmax - fmin)
    return dist


def farthest_point_order(X, front, already=None):
    """Greedy farthest-point ordering of `front` in design space X (preserve
    geometric diversity of material distributions).  Returns indices of `front`
    ordered from most to least 'spread-adding'.

    X : (M, n) design (density) matrix.  `already` : indices already selected
    (their presence reduces the marginal diversity of close candidates).
    """
    front = list(front)
    Xf = X[front]
    if already is not None and len(already) > 0:
        ref = X[already]
        mind = np.full(len(front), np.inf)
        for r in ref:
            mind = np.minimum(mind, np.linalg.norm(Xf - r, axis=1))
    else:
        mind = np.full(len(front), np.inf)
    order = []
    remaining = list(range(len(front)))
    # seed: the point with the largest distance to the already-selected set
    while remaining:
        if np.all(np.isinf(mind[remaining])):
            # nothing selected yet: start from an arbitrary extreme
            pick = remaining[int(np.argmax(np.linalg.norm(Xf[remaining], axis=1)))]
        else:
            pick = remaining[int(np.argmax(mind[remaining]))]
        order.append(front[pick])
        remaining.remove(pick)
        # update min-distances with the newly picked point
        d = np.linalg.norm(Xf - Xf[pick], axis=1)
        mind = np.minimum(mind, d)
    return order


# --------------------------------------------------------------------------- #
#  Environmental selection (truncate population to size Npop)
# --------------------------------------------------------------------------- #
def select(F, Npop, X=None, mode="diversity"):
    """Select Npop survivors from a candidate set by two-stage ranking.

    Stage 1: non-dominated sorting (Pareto rank).
    Stage 2: within the last (partial) front, order by `mode`:
        "crowding"  -> NSGA-II crowding distance (objective space),
        "diversity" -> farthest-point in design space (needs X).
    Returns the indices of the selected survivors.
    """
    F = np.asarray(F, dtype=float)
    M = len(F)
    if M <= Npop:
        return np.arange(M)
    fronts = fast_non_dominated_sort(F)
    selected = []
    for fr in fronts:
        if len(selected) + len(fr) <= Npop:
            selected.extend(fr)
        else:
            need = Npop - len(selected)
            if mode == "diversity" and X is not None:
                order = farthest_point_order(X, fr, already=np.array(selected, dtype=int))
                selected.extend(order[:need])
            else:
                cd = crowding_distance(F, fr)
                order = [fr[i] for i in np.argsort(-cd)]
                selected.extend(order[:need])
            break
    return np.array(selected, dtype=int)


# --------------------------------------------------------------------------- #
#  Hypervolume indicator (paper Eq. 9)
# --------------------------------------------------------------------------- #
def hypervolume(F, ref):
    """Hypervolume dominated by point set F w.r.t. reference point `ref`
    (minimization).  Exact for 2 objectives; Monte Carlo for >2.
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.shape[1] == 2:
        # keep only points that dominate the reference
        mask = np.all(F < ref, axis=1)
        P = F[mask]
        if len(P) == 0:
            return 0.0
        # Pareto front
        fronts = fast_non_dominated_sort(P)
        P = P[fronts[0]]
        P = P[np.argsort(P[:, 0])]      # ascending in obj1
        hv = 0.0
        prev_y = ref[1]
        for x, y in P:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
        return float(hv)
    # >2 objectives: Monte Carlo estimate
    lo = F.min(axis=0)
    n = 200000
    rng = np.random.default_rng(0)
    pts = rng.uniform(lo, ref, size=(n, F.shape[1]))
    dominated = np.zeros(n, dtype=bool)
    for f in F:
        dominated |= np.all(f <= pts, axis=1)
    vol = np.prod(ref - lo)
    return float(dominated.mean() * vol)
