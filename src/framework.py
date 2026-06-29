"""
Proposed EA-based topology optimization framework with Wasserstein crossover.

Implements Algorithm 3 of the paper:

    1. for k = 1..N_lf:  solve the LF optimization problem  -> initial designs
    2. for t = 0..t_max:
         a. HF-evaluate every candidate in the temporary set
         b. drop constraint-violating candidates
         c. merge with the running population (t>0) and run selection
         d. convergence check on the hypervolume
         e. generate N_xo offspring by Wasserstein crossover of random parents
         f. the offspring become the next temporary set

The Wasserstein-crossover step uses the adaptive entropic regularization of
Eqs. (18)-(19): the L2 distance between the two selected parents is mapped to an
entropic coefficient in [eps_min, eps_max].

Problem-specific pieces (LF optimization and HF evaluation) are provided by a
`Problem` object; `StressPlateProblem` below implements the 2D cracked-plate
example of Section 5.1.
"""
from __future__ import annotations
import time
import numpy as np

from selection import select, hypervolume, fast_non_dominated_sort
from wasserstein import (wasserstein_crossover, population_distance_matrix,
                         adaptive_eps, eps_to_sigma)


# --------------------------------------------------------------------------- #
#  Generic framework loop (Algorithm 3)
# --------------------------------------------------------------------------- #
def run_framework(problem, cfg, crossover="wasserstein", rng=None, logger=print):
    """Run the EA-based framework.  `crossover` is "wasserstein" or a callable
    crossover_fn(parent_i, parent_j, grid_shape, lam, eps, rng) -> offspring.
    Returns a results dict.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed", 0))
    t0 = time.time()

    # ---- 1. LF optimization -> initial designs (or reuse precomputed) ----
    if cfg.get("init_designs") is not None:
        designs = np.asarray(cfg["init_designs"])
        info = cfg.get("init_info", [None] * len(designs))
        logger(f"[1] Using precomputed initial population ({len(designs)} designs)")
    else:
        logger("[1] Low-fidelity optimization (initial population)...")
        designs, info = problem.generate_initial_population(
            cfg["n_s1"], cfg["n_s2"], cfg["R_min"], cfg["R_max"],
            cfg["V_min"], cfg["V_max"], maxiter=cfg.get("lf_maxiter", 60),
            move=cfg.get("lf_move", 0.05), verbose=cfg.get("lf_verbose", False))
        logger(f"    generated {len(designs)} LF designs in {time.time()-t0:.1f}s")

    Theta = list(designs)                  # current population (density vectors)
    Theta_tmp = list(designs)              # newly added candidates to evaluate
    F_pop = None                           # objective values of Theta

    hv_hist, pareto_hist = [], []
    ref = None
    init_objs = None

    for t in range(cfg["t_max"] + 1):
        # ---- (a) HF evaluation of the temporary set ----
        F_tmp = []
        for g in Theta_tmp:
            obj, feasible, _ = problem.hf_evaluate(g)
            F_tmp.append(obj if feasible else np.array([np.inf, np.inf]))
        F_tmp = np.array(F_tmp)

        # ---- (c) merge with running population ----
        if t == 0:
            cand = list(Theta_tmp)
            Fc = F_tmp
            init_objs = F_tmp.copy()
            # reference point (Eq. 9): base it on the FEASIBLE (finite) initial
            # designs only -- a single infeasible seed would otherwise push the
            # reference to +inf and make every hypervolume inf.
            fin = np.isfinite(init_objs).all(axis=1)
            base = init_objs[fin] if fin.any() else init_objs
            ref = 1.1 * np.max(base, axis=0)
        else:
            cand = list(Theta) + list(Theta_tmp)
            Fc = np.vstack([F_pop, F_tmp])

        # ---- selection -> keep Npop ----
        Xc = np.array(cand)
        sel = select(Fc, cfg["N_pop"], X=Xc, mode=cfg.get("sel_mode", "diversity"),
                     grid_shape=problem.grid_shape)
        Theta = [cand[i] for i in sel]
        F_pop = Fc[sel]

        # ---- (d) convergence check (hypervolume) ----
        hv = hypervolume(F_pop, ref)
        hv_hist.append(hv)
        front = fast_non_dominated_sort(F_pop)[0]
        pareto_hist.append(F_pop[front].copy())
        logger(f"[t={t:3d}] |pop|={len(Theta)}  HV={hv:.5g}  "
               f"minJ1={F_pop[:,0].min():.3f}  minJ2={F_pop[:,1].min():.3f}")

        if t == cfg["t_max"]:
            break
        # optional HV-based convergence
        if cfg.get("hv_tol") and t > 5:
            rel = (hv_hist[-1] - hv_hist[-6]) / (abs(hv_hist[-6]) + 1e-12)
            if rel < cfg["hv_tol"]:
                logger(f"    converged (HV rel increase {rel:.2e} < {cfg['hv_tol']})")
                break

        # ---- (e) crossover -> N_xo offspring ----
        Xpop = np.array(Theta)
        D = population_distance_matrix(Xpop)
        Dmax = D.max() if D.size else 1.0
        Dmin = D[D > 0].min() if np.any(D > 0) else 0.0

        # generative crossovers (e.g. VAE) are retrained on the population
        is_trainable = hasattr(crossover, "fit") and hasattr(crossover, "crossover")
        if is_trainable:
            crossover.fit(Xpop)

        offspring = []
        for _ in range(cfg["N_xo"]):
            i, j = rng.choice(len(Theta), size=2, replace=False)
            lam = float(rng.uniform(0, 1))
            if crossover == "wasserstein":
                eps = adaptive_eps(D[i, j], Dmin, Dmax,
                                   cfg["eps_min"], cfg["eps_max"])
                sigma = eps_to_sigma(eps, h=problem.mesh.h)
                child = wasserstein_crossover(
                    Theta[i], Theta[j], problem.grid_shape, lam=lam,
                    sigma=sigma, n_iter=cfg.get("wc_iter", 500),
                    tol=cfg.get("wc_tol", 1e-9), rng=rng)
            elif is_trainable:
                child = crossover.crossover(Theta[i], Theta[j],
                                            problem.grid_shape, lam, rng)
            else:
                child = crossover(Theta[i], Theta[j], problem.grid_shape, lam, rng)
            offspring.append(child)
        Theta_tmp = offspring

    return dict(
        problem=problem, population=np.array(Theta), F=F_pop,
        init_designs=np.array(designs), init_F=init_objs,
        hv_hist=np.array(hv_hist), pareto_hist=pareto_hist, ref=ref,
        info=info, wall_time=time.time() - t0)


# --------------------------------------------------------------------------- #
#  Baseline crossovers for comparison
# --------------------------------------------------------------------------- #
def linear_crossover(gi, gj, grid_shape, lam, rng):
    """Euclidean (linear) interpolation -- the simplest baseline."""
    return lam * np.asarray(gi) + (1 - lam) * np.asarray(gj)
