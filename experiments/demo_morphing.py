"""
Sanity check / Figure-1-style demo of Wasserstein barycentric morphing.

Morphs between two binary shapes by sweeping the barycentric weight lambda,
and contrasts it with simple linear (Euclidean) interpolation -- reproducing
the qualitative behaviour discussed in Section 2.3 of the paper (the
Wasserstein interpolation transports mass, keeping solid/void distinct, whereas
the linear one just cross-fades).
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wasserstein import convolutional_barycenter  # noqa: E402

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)


def disk(n, cx, cy, r):
    y, x = np.mgrid[0:n, 0:n]
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(float)


def ring(n, cx, cy, r_out, r_in):
    y, x = np.mgrid[0:n, 0:n]
    d2 = (x - cx) ** 2 + (y - cy) ** 2
    return ((d2 <= r_out ** 2) & (d2 >= r_in ** 2)).astype(float)


def main():
    n = 128
    # two distinct shapes: a disk on the left, a ring on the right
    A = disk(n, 42, 64, 18)
    B = ring(n, 86, 64, 24, 12)
    A /= A.sum()
    B /= B.sum()

    lambdas = np.linspace(0, 1, 7)
    sigma = 4.0

    fig, axes = plt.subplots(2, len(lambdas), figsize=(2.0 * len(lambdas), 4.2))
    for c, lam in enumerate(lambdas):
        # weight on B is lam, on A is (1-lam)
        w = np.array([1.0 - lam, lam])
        bary = convolutional_barycenter(np.stack([A, B]), weights=w,
                                        sigma=sigma, n_iter=2000, tol=1e-9)
        lin = (1.0 - lam) * A + lam * B

        axes[0, c].imshow(bary, cmap="viridis")
        axes[0, c].set_title(f"$\\lambda$={lam:.2f}", fontsize=9)
        axes[1, c].imshow(lin, cmap="viridis")
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("Wasserstein\nbarycenter", fontsize=10)
    axes[1, 0].set_ylabel("Linear\ninterpolation", fontsize=10)
    fig.suptitle("Wasserstein barycentric morphing vs. linear interpolation", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "demo_morphing.png")
    fig.savefig(path, dpi=130)
    print("saved", os.path.abspath(path))

    # quick numeric checks
    bary_mid = convolutional_barycenter(np.stack([A, B]), weights=[0.5, 0.5],
                                        sigma=sigma, n_iter=2000, tol=1e-9)
    print("barycenter sum (should be ~1):", bary_mid.sum())
    print("barycenter min/max:", bary_mid.min(), bary_mid.max())


if __name__ == "__main__":
    main()
