"""
L-bracket Wasserstein-crossover EA with the body-fitted-mesh HF model.

LF = compliance OC (structured grid, passive void); HF = true max von Mises on a
body-fitted mesh (DPTO port, MATLAB-cross-checked).  Small/fast demo config.
"""
import os, sys, time, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lbracket import LBracketProblem            # noqa: E402
from framework import run_framework             # noqa: E402
from selection import fast_non_dominated_sort   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=40)
    ap.add_argument("--npop", type=int, default=48)
    ap.add_argument("--nxo", type=int, default=48)
    # seeding grid: the paper uses n_s1=4 (filter radius) x n_s2=25 (volume) = 100
    # LF designs (Table 1).  Defaults here are a tractable 4 x 12 = 48; pass
    # --n-s1 4 --n-s2 25 for the full paper scale (slower).
    ap.add_argument("--n-s1", type=int, default=4)
    ap.add_argument("--n-s2", type=int, default=12)
    ap.add_argument("--random-init", dest="random_init", action="store_true",
                    default=True, help="randomized LF multistart (more diversity)")
    ap.add_argument("--no-random-init", dest="random_init", action="store_false")
    ap.add_argument("--r-fillet", type=float, default=0.0)   # sharp re-entrant corner
    ap.add_argument("--hf-seeds", type=int, default=3)
    ap.add_argument("--lf-method", default="stress", choices=["stress", "compliance"])
    ap.add_argument("--lf-maxiter", type=int, default=60)
    ap.add_argument("--lf-move", type=float, default=0.1)
    ap.add_argument("--tag", default="stressLF")
    args = ap.parse_args()
    tag = args.tag

    prob = LBracketProblem(nelx_lf=75, hf_h=2.0, hf_minedge=3.0, hf_iter=80,
                           r_fillet=args.r_fillet, hf_seeds=args.hf_seeds,
                           lf_method=args.lf_method)
    cfg = dict(seed=0, n_s1=args.n_s1, n_s2=args.n_s2,
               R_min=2 * prob.mesh.h, R_max=5 * prob.mesh.h,
               V_min=0.30, V_max=0.60, lf_maxiter=args.lf_maxiter, lf_move=args.lf_move,
               N_pop=args.npop, N_xo=args.nxo, t_max=args.tmax,
               eps_min=5.0, eps_max=50.0, wc_iter=300, wc_tol=1e-8,
               sel_mode="diversity", random_init=args.random_init,
               checkpoint=os.path.join(OUT, f"lbr_ckpt_{tag}.npz"))

    # cache the LF population, keyed by a hash of the LF-relevant config so a
    # changed seeding / random-init / mesh never silently reloads a stale cache.
    import hashlib
    lf_key = dict(n_s1=cfg["n_s1"], n_s2=cfg["n_s2"], R_min=cfg["R_min"],
                  R_max=cfg["R_max"], V_min=cfg["V_min"], V_max=cfg["V_max"],
                  lf_maxiter=cfg["lf_maxiter"], lf_move=cfg["lf_move"],
                  random_init=cfg["random_init"], seed=cfg["seed"],
                  lf_method=args.lf_method, r_fillet=args.r_fillet, nelx=75)
    chash = hashlib.sha1(repr(sorted(lf_key.items())).encode()).hexdigest()[:8]
    cache = os.path.join(OUT, f"lbr_initpop_{tag}_{chash}.npz")
    partial = os.path.join(OUT, f"lbr_initpop_{tag}_{chash}.partial.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        designs, info = d["designs"], list(d["info"])
        print(f"[cache] loaded {len(designs)} LF designs ({chash})")
        cfg["init_designs"] = designs; cfg["init_info"] = info
    else:
        print(f"[lf] generating L-bracket population (LF={args.lf_method})...")
        t0 = time.time()
        designs, info = prob.generate_initial_population(
            cfg["n_s1"], cfg["n_s2"], cfg["R_min"], cfg["R_max"],
            cfg["V_min"], cfg["V_max"], maxiter=cfg["lf_maxiter"],
            move=cfg["lf_move"], verbose=True,
            random_init=cfg["random_init"], seed=cfg["seed"], cache_path=partial)
        np.savez(cache, designs=designs, info=np.array(info, dtype=object))
        if os.path.exists(partial):
            os.remove(partial)               # finished -> drop the partial
        cfg["init_designs"] = designs; cfg["init_info"] = info
        print(f"[lf] {len(designs)} designs in {time.time()-t0:.0f}s")

    initF = np.array([prob.hf_evaluate(g)[0] for g in designs])
    res = run_framework(prob, cfg, crossover="wasserstein")
    F = res["F"]
    print(f"\nwall={res['wall_time']:.0f}s  initial minJ1={initF[:,0].min():.4f}  "
          f"final minJ1={F[:,0].min():.4f}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    hv = res["hv_hist"]
    ax[0].plot(hv / hv[0], "-o", ms=3); ax[0].set_xlabel("iteration"); ax[0].grid(alpha=0.3)
    ax[0].set_ylabel("normalized hypervolume"); ax[0].set_title("HV convergence (body-fitted HF)")
    ax[1].scatter(initF[:, 1], initF[:, 0], s=24, c="0.6", label="initial (LF)")
    fr = fast_non_dominated_sort(F)[0]; P = F[fr]; P = P[np.argsort(P[:, 1])]
    ax[1].plot(P[:, 1], P[:, 0], "-o", ms=4, c="C3", label="optimized (Wasserstein)")
    ax[1].set_xlabel("$J_2$ volume fraction"); ax[1].set_ylabel("$J_1$ max von Mises")
    ax[1].set_title("Objective space"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"lbracket_results_{tag}.png"), dpi=130)
    print("saved results/lbracket_results.png")

    # representative initial vs optimized at similar volume
    import bodyfitted as bf
    target = np.median(F[fr, 1])
    io = int(np.argmin(np.abs(initF[:, 1] - target)))
    oo = fr[int(np.argmin(np.abs(F[fr, 1] - initF[io, 1])))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    for col, (g, lbl) in enumerate([(designs[io], "initial (LF)"),
                                    (res["population"][oo], "optimized (Wasserstein)")]):
        fld = prob._to_hf_field(g)
        J1, J2, mesh = bf.hf_lbracket_stress(fld, prob.xn, prob.yn, geom=prob.hf_geom,
                                             seed=0, n_iter=80, return_mesh=True)
        p, t = mesh["p"], mesh["t"]; solid = mesh["rho"] > 0.5
        vm = mesh["vm"].copy(); vm[~solid] = np.nan
        tp = ax[col].tripcolor(p[:, 0], p[:, 1], t,
                               facecolors=np.clip(vm, 0, np.nanpercentile(vm, 99)),
                               cmap="jet", edgecolors="none")
        ax[col].set_title(f"{lbl}\nJ1={J1:.3f}, J2={J2:.3f}"); ax[col].set_aspect("equal"); ax[col].axis("off")
    fig.suptitle("L-bracket: initial vs optimized (body-fitted HF stress)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"lbracket_compare_{tag}.png"), dpi=130)
    print("saved results/lbracket_compare.png")

    np.savez(os.path.join(OUT, f"lbr_result_{tag}.npz"), population=res["population"],
             F=F, hv_hist=hv, initF=initF, wall_time=res["wall_time"])


if __name__ == "__main__":
    main()
