"""
Reproduction of Section 5.1: the 2D cracked-plate stress-minimization example.

Runs the proposed framework with Wasserstein crossover and compares it against
the VAE-based crossover (conventional DDTD) and a simple linear-interpolation
crossover, reproducing the qualitative findings of Figs. 5-9:

  * Fig. 5  : LF-optimized initial population.
  * Fig. 6a : hypervolume convergence (normalized), per crossover operator.
  * Fig. 6b : objective space (max stress J1 vs volume fraction J2).
  * Fig. 7  : optimized population (Wasserstein crossover).
  * Fig. 8/9: representative initial vs. optimized design + stress field.

Resolution and population sizes are reduced relative to the paper (Table 1) so
the full pipeline runs on a laptop CPU in minutes; pass --paper for the exact
paper settings (slow).  The initial population is cached and reused across
crossover operators for a fair comparison.
"""
import os, sys, argparse, time, json, hashlib, platform, subprocess, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framework import StressPlateProblem, run_framework, linear_crossover  # noqa
from selection import fast_non_dominated_sort  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def get_config(args):
    if args.paper:
        cfg = dict(nelx=100, n_s1=4, n_s2=25, N_pop=100, N_xo=100, t_max=100,
                   lf_maxiter=120, lf_move=0.05,
                   R_min=0.03, R_max=0.12, V_min=0.30, V_max=0.60, R_h=0.01,
                   eps_min=1e-6, eps_max=1e-4, wc_iter=1000, wc_tol=1e-9,
                   vae_epochs=500)
    else:  # fast demo
        cfg = dict(nelx=60, n_s1=4, n_s2=10, N_pop=40, N_xo=40, t_max=30,
                   lf_maxiter=50, lf_move=0.08,
                   R_min=0.03, R_max=0.12, V_min=0.30, V_max=0.60, R_h=0.01,
                   eps_min=1e-5, eps_max=1.5e-3, wc_iter=400, wc_tol=1e-8,
                   vae_epochs=80)
    cfg.update(seed=0, sel_mode="diversity")
    return cfg


def get_initial_population(problem, cfg, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] loaded initial population from {cache}")
        return d["designs"], list(d["info"])
    print("[lf] generating initial population (this is the slow part)...")
    t0 = time.time()
    designs, info = problem.generate_initial_population(
        cfg["n_s1"], cfg["n_s2"], cfg["R_min"], cfg["R_max"],
        cfg["V_min"], cfg["V_max"], maxiter=cfg["lf_maxiter"],
        move=cfg["lf_move"], verbose=True)
    np.savez(cache, designs=designs, info=np.array(info, dtype=object))
    print(f"[lf] done in {time.time()-t0:.1f}s, cached to {cache}")
    return designs, info


def grid(ax, field, shape, title="", cmap="gray_r", vmax=None):
    ax.imshow(field.reshape(shape), origin="lower", cmap=cmap,
              extent=[0, 1, 0, 2], vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def plot_population(designs, shape, path, ncol=10, title="", maxn=40):
    n = min(len(designs), maxn)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.0, nrow * 2.0))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i < n:
            grid(ax, designs[i], shape)
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="exact paper settings (slow)")
    ap.add_argument("--methods", default="wasserstein,vae,linear")
    args = ap.parse_args()
    cfg = get_config(args)
    methods = args.methods.split(",")

    problem = StressPlateProblem(nelx=cfg["nelx"], R_h=cfg["R_h"])
    shape = problem.grid_shape
    print(f"problem: {cfg['nelx']}x{2*cfg['nelx']} grid, n={problem.n} elements")

    cache = os.path.join(OUT, f"init_pop_{cfg['nelx']}_{cfg['n_s1']}x{cfg['n_s2']}.npz")
    designs, info = get_initial_population(problem, cfg, cache)

    # ---- Fig 5: initial population ----
    initF = np.array([problem.hf_evaluate(g)[0] for g in designs])
    plot_population(designs, shape, os.path.join(OUT, "fig5_initial_population.png"),
                    title="Fig. 5  LF-optimized initial population (density)")

    results = {}
    base_cfg = dict(cfg); base_cfg["init_designs"] = designs; base_cfg["init_info"] = info
    tag = f"{cfg['nelx']}_t{cfg['t_max']}"

    for m in methods:
        rcache = os.path.join(OUT, f"res_{m}_{tag}.npz")
        if os.path.exists(rcache):
            d = np.load(rcache, allow_pickle=True)
            results[m] = dict(population=d["population"], F=d["F"],
                              hv_hist=d["hv_hist"], wall_time=float(d["wall_time"]),
                              _from_cache=True)
            print(f"[cache] loaded {m} results from {rcache}")
            continue
        print(f"\n===== running framework with {m.upper()} crossover =====")
        rcfg = dict(base_cfg)
        if m == "wasserstein":
            res = run_framework(problem, rcfg, crossover="wasserstein")
        elif m == "linear":
            res = run_framework(problem, rcfg, crossover=linear_crossover)
        elif m == "vae":
            from vae import VAECrossover
            vae = VAECrossover(n=problem.n, epochs=cfg["vae_epochs"], seed=0)
            res = run_framework(problem, rcfg, crossover=vae)
        else:
            print("unknown method", m); continue
        res["_from_cache"] = False
        results[m] = res
        np.savez(rcache, population=res["population"], F=res["F"],
                 hv_hist=res["hv_hist"], wall_time=res["wall_time"])
        print(f"  {m}: wall={res['wall_time']:.1f}s  final min J1={res['F'][:,0].min():.3f}")

    # ---- Fig 6a: hypervolume convergence (normalized) ----
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = dict(wasserstein="C3", vae="C0", linear="C2")
    labels = dict(wasserstein="Wasserstein (proposed)", vae="VAE (DDTD)", linear="Linear")
    for m, res in results.items():
        hv = res["hv_hist"]; hv = hv / hv[0]
        ax.plot(hv, color=colors.get(m, "k"), label=labels.get(m, m), lw=2)
    ax.set_xlabel("iteration $t$"); ax.set_ylabel("normalized hypervolume")
    ax.set_title("Fig. 6(a)  Hypervolume convergence"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6a_hypervolume.png"), dpi=130)
    plt.close(fig); print("saved fig6a")

    # ---- Fig 6b: objective space ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(initF[:, 1], initF[:, 0], s=22, c="0.6", label="initial", zorder=2)
    for m, res in results.items():
        F = res["F"]; fr = fast_non_dominated_sort(F)[0]
        P = F[fr]; P = P[np.argsort(P[:, 1])]
        ax.plot(P[:, 1], P[:, 0], "-o", ms=4, color=colors.get(m, "k"),
                label=labels.get(m, m), zorder=3)
    ax.set_xlabel("$J_2$  volume fraction"); ax.set_ylabel("$J_1$  max von Mises stress")
    ax.set_title("Fig. 6(b)  Objective space"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6b_objective_space.png"), dpi=130)
    plt.close(fig); print("saved fig6b")

    # ---- Fig 7: optimized population (Wasserstein) ----
    if "wasserstein" in results:
        plot_population(results["wasserstein"]["population"], shape,
                        os.path.join(OUT, "fig7_optimized_wasserstein.png"),
                        title="Fig. 7  Optimized population (Wasserstein crossover)")

    # ---- Fig 8/9: representative initial vs optimized stress comparison ----
    if "wasserstein" in results:
        make_stress_comparison(problem, designs, initF,
                               results["wasserstein"], shape)

    # ---- provenance manifest + text summary ----
    manifest_name = write_manifest(cfg, methods, results, problem, cache, tag)
    summarize(results, initF, manifest_name)


def make_stress_comparison(problem, designs, initF, res, shape):
    """Pick an initial and an optimized design at similar volume fraction."""
    F = res["F"]; pop = res["population"]
    # choose target volume ~ median of optimized front
    fr = fast_non_dominated_sort(F)[0]
    target_v = np.median(F[fr, 1])
    io = int(np.argmin(np.abs(initF[:, 1] - target_v)))
    oo = fr[int(np.argmin(np.abs(F[fr, 1] - initF[io, 1])))]

    fig, axes = plt.subplots(2, 2, figsize=(6, 7))
    for col, (g, tag) in enumerate([(designs[io], "initial"),
                                    (pop[oo], "optimized (Wasserstein)")]):
        obj, _, sigma = problem.hf_evaluate(g)
        rho = g.reshape(shape)
        axes[0, col].imshow(rho, origin="lower", cmap="gray_r", extent=[0, 1, 0, 2])
        axes[0, col].set_title(f"{tag}\n$J_1$={obj[0]:.1f}, $J_2$={obj[1]:.3f}", fontsize=9)
        im = axes[1, col].imshow(sigma.reshape(shape), origin="lower", cmap="jet",
                                 extent=[0, 1, 0, 2])
        plt.colorbar(im, ax=axes[1, col], fraction=0.08)
        for r in (0, 1):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    axes[0, 0].set_ylabel("density"); axes[1, 0].set_ylabel("von Mises stress")
    fig.suptitle("Fig. 8/9  Initial vs. optimized at similar volume fraction", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_stress_comparison.png"), dpi=130)
    plt.close(fig); print("saved fig8")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _git_dirty():
    """True if the working tree has uncommitted changes; None if not a repo."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode()
        return len(out.strip()) > 0
    except Exception:
        return None


def _sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_versions():
    vers = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "matplotlib", "jax"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = None
    return vers


def write_manifest(cfg, methods, results, problem, cache, tag):
    """Provenance record for one run -> results/run_manifest_<tag>.json."""
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "methods": methods,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "package_versions": _pkg_versions(),
        "config": {k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool, type(None)))},
        "seed": cfg.get("seed"),
        "problem": {
            "nelx": problem.mesh.nelx, "nely": problem.mesh.nely,
            "n_elements": problem.n, "grid_shape": list(problem.grid_shape),
            "hard_binarize": problem.hard_binarize,
            "sel_mode": cfg.get("sel_mode"),
        },
        "initial_population_cache": {
            "path": os.path.relpath(cache, OUT), "sha256": _sha256(cache)},
        "results": {},
    }
    for m, res in results.items():
        F = res["F"]; hv = res["hv_hist"]
        manifest["results"][m] = {
            "output_npz": f"res_{m}_{tag}.npz",
            "loaded_from_cache": bool(res.get("_from_cache", False)),
            "wall_time_s": round(float(res["wall_time"]), 2),
            "min_J1": round(float(F[:, 0].min()), 4),
            "min_J2": round(float(F[:, 1].min()), 4),
            "hv_first": round(float(hv[0]), 5),
            "hv_last": round(float(hv[-1]), 5),
            "hv_improvement_pct": round(100 * (hv[-1] / hv[0] - 1), 3),
        }
    path = os.path.join(OUT, f"run_manifest_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("saved", path)
    return os.path.basename(path)


def summarize(results, initF, manifest_name=None):
    lines = ["", "=" * 64, "SUMMARY  (2D cracked-plate stress minimization)", "=" * 64]
    fr0 = fast_non_dominated_sort(initF)[0]
    lines.append(f"initial      : min J1={initF[:,0].min():.3f}  |front0|={len(fr0)}")
    for m, res in results.items():
        F = res["F"]; hv = res["hv_hist"]
        imp = 100 * (hv[-1] / hv[0] - 1)
        lines.append(f"{m:12s}: min J1={F[:,0].min():.3f}  "
                     f"HV improvement={imp:5.1f}%  wall={res['wall_time']:.1f}s")
    lines.append("-" * 64)
    lines.append("NOTE: single-seed run; HV ranking is sensitive to the shared "
                 "extreme point and should not be over-interpreted (see README).")
    if manifest_name:
        lines.append(f"provenance: results/{manifest_name}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
