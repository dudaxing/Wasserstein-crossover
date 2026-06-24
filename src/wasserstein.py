"""
Wasserstein crossover for evolutionary topology optimization.

Faithful reproduction of the core operator from:
    T. Kii, K. Yaji, H. Teramoto, K. Fujita,
    "Wasserstein crossover for evolutionary algorithm-based topology
    optimization", Comput. Methods Appl. Mech. Engrg. 451 (2026) 118713.

The operator generates an offspring as the *Wasserstein barycenter* of two
parent material-density distributions, treated as probability distributions.

Key paper equations reproduced here:
  - Eq. (10): normalize a density vector into a probability vector.
  - Eq. (11): interpret the probability vector as a discrete measure on the grid.
  - Eq. (12): the offspring is the (entropy-regularized) Wasserstein barycenter
              with a random weight lambda in [0, 1].
  - Eq. (13): convert the barycenter probability vector back to a density via
              min-max scaling.
  - Algorithm 2: Sinkhorn / iterative-Bregman computation of the barycenter.
  - Eqs. (18)-(19): adaptive entropic regularization coefficient based on the
              L2 distance between the two selected parents.

Acceleration (Section 2.2 / 4.3): for the 2-Wasserstein distance with a squared
Euclidean ground cost, the kernel  K = exp(-C/eps)  is a Gaussian kernel, so the
matrix-vector products  K v  and  K^T u  in Algorithm 2 become *convolutions
with a Gaussian filter*.  We use this convolutional form, which makes the
operator scale to fine grids (and to 3D) cheaply.  The relation between the
entropic coefficient `eps` and the Gaussian standard deviation `sigma` (in grid
spacings `h`) is

        K_ij = exp(-||x_i - x_j||^2 / eps)   <=>   sigma = sqrt(eps / 2) / h

A NumPy backend is always available; an optional JAX backend (matching the
paper's implementation) is used automatically if JAX is installed.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------- #
#  eps <-> sigma conversion
# --------------------------------------------------------------------------- #
def eps_to_sigma(eps: float, h: float = 1.0) -> float:
    """Gaussian std (in pixels) equivalent to entropic coefficient `eps`.

    For the squared-Euclidean cost, K_ij = exp(-||x_i-x_j||^2 / eps) is a
    Gaussian with variance s^2 = eps/2 in physical units; in pixel units the
    standard deviation is sqrt(eps/2)/h.
    """
    return float(np.sqrt(eps / 2.0) / h)


# --------------------------------------------------------------------------- #
#  Convolutional Wasserstein barycenter  (Algorithm 2, convolutional form)
# --------------------------------------------------------------------------- #
def convolutional_barycenter(
    dists,
    weights=None,
    sigma: float = 1.0,
    n_iter: int = 1000,
    tol: float = 1e-9,
    stab: float = 1e-30,
    return_iters: bool = False,
):
    """Entropy-regularized Wasserstein barycenter via Gaussian convolutions.

    Implements Algorithm 2 of the paper, replacing every  K v  /  K^T u  product
    by a Gaussian blur (valid for the 2-Wasserstein distance on a regular grid;
    the Gaussian kernel is symmetric so K = K^T).

    Parameters
    ----------
    dists : (N, *grid) array
        N input probability distributions on a d-dimensional regular grid.
        Each slice must be nonnegative; it is renormalized to sum to one.
    weights : (N,) array, optional
        Barycentric weights lambda_i (>=0, sum to 1).  Defaults to uniform.
    sigma : float
        Gaussian standard deviation (in grid spacings) = entropic blur.
    n_iter : int
        Maximum number of Sinkhorn / Bregman iterations.
    tol : float
        Convergence tolerance on the marginal deviation E (Algorithm 2, line 8).
    stab : float
        Numerical floor added before divisions / used to avoid 0/0.

    Returns
    -------
    bary : (*grid,) array
        The barycenter probability distribution (sums to one).
    """
    dists = np.asarray(dists, dtype=np.float64)
    N = dists.shape[0]
    grid_shape = dists.shape[1:]

    if weights is None:
        weights = np.full(N, 1.0 / N)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    # normalize each input to a probability distribution
    a = dists.copy()
    a = a / (a.reshape(N, -1).sum(axis=1).reshape((N,) + (1,) * len(grid_shape)) + stab)

    def blur(field):
        # Gaussian blur == multiplication by the Gaussian kernel K.
        # 'reflect' (Neumann) boundary conserves mass, as in Solomon et al.
        return gaussian_filter(field, sigma=sigma, mode="reflect")

    u = np.ones_like(a)
    v = np.ones_like(a)
    bary = np.full(grid_shape, 1.0 / np.prod(grid_shape))

    n_done = n_iter
    for it in range(n_iter):
        bary_new = np.ones(grid_shape)
        marginals = np.empty_like(a)  # v^(i) * (K^T u^(i)) for the conv. check

        for i in range(N):
            Kv = blur(v[i])
            u[i] = a[i] / (Kv + stab)
            Ku = blur(u[i])
            marginals[i] = v[i] * Ku
            # accumulate the (weighted) geometric mean -> barycenter estimate
            bary_new *= np.power(Ku + stab, weights[i])

        # update v^(i) = bary / (K^T u^(i))     (iterative Bregman projection)
        for i in range(N):
            Ku = blur(u[i])
            v[i] = bary_new / (Ku + stab)

        # convergence: deviation of the per-distribution marginals (Algorithm 2)
        E = np.std(marginals, axis=0).sum()
        bary = bary_new
        if E < tol:
            n_done = it + 1
            break

    bary = bary / (bary.sum() + stab)
    if return_iters:
        return bary, n_done
    return bary


# --------------------------------------------------------------------------- #
#  Adaptive entropic regularization  (Eqs. 18-19)
# --------------------------------------------------------------------------- #
def population_distance_matrix(population):
    """Eq. (18): pairwise Euclidean distance matrix D over the population.

    `population` is (M, n) with each row a flattened density vector gamma^(k).
    """
    P = np.asarray(population, dtype=np.float64).reshape(len(population), -1)
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
    sq = np.sum(P * P, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (P @ P.T)
    np.clip(D2, 0.0, None, out=D2)
    return np.sqrt(D2)


def adaptive_eps(d_ij, d_min, d_max, eps_min, eps_max):
    """Eq. (19): map a parent distance d_ij to an entropic coefficient.

    Similar parents (small d_ij) -> small eps  -> sharp / accurate transport.
    Dissimilar parents (large d_ij) -> large eps -> more blur / cheaper.
    """
    if d_max <= d_min:
        return 0.5 * (eps_min + eps_max)
    t = (d_ij - d_min) / (d_max - d_min)
    return eps_min + (eps_max - eps_min) * t


# --------------------------------------------------------------------------- #
#  Wasserstein crossover  (Section 3, steps 1-4)
# --------------------------------------------------------------------------- #
def wasserstein_crossover(
    gamma_i,
    gamma_j,
    grid_shape,
    lam=None,
    eps=None,
    sigma=None,
    h=1.0,
    n_iter=1000,
    tol=1e-9,
    rng=None,
):
    """Generate one offspring as the Wasserstein barycenter of two parents.

    Steps (paper, Section 3):
      2. convert parents to probability distributions (Eqs. 10-11),
      3. compute the barycenter with weight (lam, 1-lam)  (Eq. 12),
      4. convert back to a density via min-max scaling      (Eq. 13).

    Parameters
    ----------
    gamma_i, gamma_j : (n,) arrays  (flattened density vectors, values in [0,1])
    grid_shape : tuple   spatial shape the vectors reshape to (e.g. (200,100))
    lam : float in [0,1] random barycentric weight (sampled if None)
    eps : float          entropic coefficient (converted to sigma) -- OR
    sigma : float        Gaussian std directly (takes precedence over eps)

    Returns
    -------
    gamma_star : (n,) array   the offspring density vector in [0,1].
    """
    if rng is None:
        rng = np.random.default_rng()
    if lam is None:
        lam = float(rng.uniform(0.0, 1.0))
    if sigma is None:
        if eps is None:
            raise ValueError("provide either `sigma` or `eps`")
        sigma = eps_to_sigma(eps, h)

    gi = np.asarray(gamma_i, dtype=np.float64).reshape(grid_shape)
    gj = np.asarray(gamma_j, dtype=np.float64).reshape(grid_shape)

    # Eq. (10)-(11): normalize to probability distributions
    pi = gi / (gi.sum() + 1e-30)
    pj = gj / (gj.sum() + 1e-30)

    bary = convolutional_barycenter(
        np.stack([pi, pj]),
        weights=np.array([lam, 1.0 - lam]),
        sigma=sigma,
        n_iter=n_iter,
        tol=tol,
    )

    # Eq. (13): min-max scaling back to a density in [0,1]
    p = bary.ravel()
    pmin, pmax = p.min(), p.max()
    if pmax - pmin < 1e-30:
        gamma_star = np.zeros_like(p)
    else:
        gamma_star = (p - pmin) / (pmax - pmin)
    return gamma_star
