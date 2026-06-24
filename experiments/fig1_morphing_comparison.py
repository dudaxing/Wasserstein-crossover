"""
Reproduction of Fig. 1 (Section 2.3): comparison of morphing methods.

Takes two parent designs and interpolates between them with
  (1) linear (Euclidean) interpolation,
  (2) VAE latent interpolation (DDTD),
  (3) Wasserstein barycentric interpolation (proposed),
then plots (a) the morphing strips and (b) the resulting designs in objective
space.  As in the paper, the Wasserstein interpolation transports material
(solid/void stay distinct, topology changes smoothly) and can yield interpolants
that DOMINATE both parents, whereas linear interpolation merely cross-fades.

Uses the 2D cracked-plate stress problem for HF evaluation.  The initial
population produced by `run_2d_stress.py` is reused as the source of parents and
as the VAE training set.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from framework import StressPlateProblem  # noqa
from wasserstein import convolutional_barycenter, eps_to_sigma  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    cache = os.path.join(OUT, "init_pop_60_4x10.npz")
    if not os.path.exists(cache):
        print("run experiments/run_2d_stress.py first (need cached population)")
        return
    designs = np.load(cache, allow_pickle=True)["designs"]
    prob = StressPlateProblem(nelx=60, R_h=0.01, hard_binarize=True)
    shape = prob.grid_shape

    F = np.array([prob.hf_evaluate(g)[0] for g in designs])
    # pick two well-formed parents with distinct volume fractions / topologies
    # (avoid the degenerate near-disconnected min-volume outliers)
    iA = int(np.argmin(np.abs(F[:, 1] - 0.33)))             # lower volume
    iB = int(np.argmin(np.abs(F[:, 1] - 0.55)))             # higher volume
    A, B = designs[iA], designs[iB]
    print(f"parents: A (J1={F[iA,0]:.1f},J2={F[iA,1]:.2f})  "
          f"B (J1={F[iB,0]:.1f},J2={F[iB,1]:.2f})")

    lambdas = np.linspace(0, 1, 9)

    # (1) linear
    lin = [(1 - l) * A + l * B for l in lambdas]
    # (3) Wasserstein barycenter (fixed moderate eps)
    pA = A.reshape(shape); pA = pA / pA.sum()
    pB = B.reshape(shape); pB = pB / pB.sum()
    sigma = eps_to_sigma(5e-4, h=prob.mesh.h)
    wass = []
    for l in lambdas:
        bary = convolutional_barycenter(np.stack([pA, pB]), weights=[1 - l, l],
                                        sigma=sigma, n_iter=800, tol=1e-9).ravel()
        p = bary
        wass.append((p - p.min()) / (p.max() - p.min() + 1e-30))
    # (2) VAE latent interpolation
    vae_imgs = None
    try:
        from vae import VAECrossover
        vae = VAECrossover(n=prob.n, epochs=150, seed=0)
        vae.fit(designs)
        zA = vae.encode_mean(A[None, :])[0]
        zB = vae.encode_mean(B[None, :])[0]
        vae_imgs = [vae.decode(((1 - l) * zA + l * zB)[None, :])[0] for l in lambdas]
    except Exception as e:
        print("VAE unavailable:", e)

    # ---- morphing strips ----
    rows = [("Linear", lin), ("VAE (DDTD)", vae_imgs), ("Wasserstein", wass)]
    rows = [r for r in rows if r[1] is not None]
    fig, axes = plt.subplots(len(rows), len(lambdas),
                             figsize=(len(lambdas) * 1.0, len(rows) * 2.0))
    axes = np.atleast_2d(axes)
    for r, (name, imgs) in enumerate(rows):
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im.reshape(shape), origin="lower", cmap="gray_r",
                              extent=[0, 1, 0, 2], vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"$\\lambda$={lambdas[c]:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("Fig. 1(b)  Morphing between two structural designs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1b_morphing_strips.png"), dpi=120)
    plt.close(fig); print("saved fig1b")

    # ---- objective space of interpolants ----
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    def evalset(imgs):
        return np.array([prob.hf_evaluate(g)[0] for g in imgs])
    for name, imgs, col in [("Linear", lin, "C2"),
                            ("VAE (DDTD)", vae_imgs, "C0"),
                            ("Wasserstein", wass, "C3")]:
        if imgs is None:
            continue
        Fi = evalset(imgs)
        ax.plot(Fi[:, 1], Fi[:, 0], "-o", ms=4, color=col, label=name, alpha=0.85)
    ax.scatter([F[iA, 1], F[iB, 1]], [F[iA, 0], F[iB, 0]],
               s=90, marker="*", c="k", zorder=5, label="parents")
    ax.set_xlabel("$J_2$  volume fraction"); ax.set_ylabel("$J_1$  max von Mises stress")
    ax.set_title("Fig. 1(c)  Objective values of interpolants")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1c_objective_space.png"), dpi=130)
    plt.close(fig); print("saved fig1c")


if __name__ == "__main__":
    main()
