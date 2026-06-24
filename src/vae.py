"""
VAE-based crossover (the DDTD baseline of the paper, Table 2).

Implements the variational autoencoder crossover used in conventional
data-driven topology design, against which the paper benchmarks the proposed
Wasserstein crossover (Section 5.1, Fig. 6).

Architecture (paper Table 2):
    encoder : [n, 512, 8]   (fully connected, ReLU; outputs mean & log-var)
    decoder : [8, 512, n]   (fully connected, ReLU hidden, sigmoid output)
    loss    : MSE reconstruction + 0.01 * KL divergence
    optimizer: Adam, lr 1e-3, batch size 10

Offspring are produced as in Section 2.3: encode two parents, linearly
interpolate the two latent vectors with a random weight, and decode.

A minimal JAX implementation with a hand-written Adam optimizer (no optax
dependency).  Falls back gracefully if JAX is unavailable.
"""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit, value_and_grad
    _HAVE_JAX = True
except Exception:                       # pragma: no cover
    _HAVE_JAX = False


def _init_params(key, n, hidden=512, latent=8):
    k = random.split(key, 6)
    def glorot(kk, shape):
        s = np.sqrt(2.0 / shape[0])
        return random.normal(kk, shape) * s
    return dict(
        We1=glorot(k[0], (n, hidden)), be1=jnp.zeros(hidden),
        Wmu=glorot(k[1], (hidden, latent)), bmu=jnp.zeros(latent),
        Wlv=glorot(k[2], (hidden, latent)), blv=jnp.zeros(latent),
        Wd1=glorot(k[3], (latent, hidden)), bd1=jnp.zeros(hidden),
        Wd2=glorot(k[4], (hidden, n)), bd2=jnp.zeros(n),
    )


if _HAVE_JAX:
    def _encode(p, x):
        h = jax.nn.relu(x @ p["We1"] + p["be1"])
        return h @ p["Wmu"] + p["bmu"], h @ p["Wlv"] + p["blv"]

    def _decode(p, z):
        h = jax.nn.relu(z @ p["Wd1"] + p["bd1"])
        return jax.nn.sigmoid(h @ p["Wd2"] + p["bd2"])

    def _loss(p, x, key, kl_w=0.01):
        mu, logvar = _encode(p, x)
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(key, mu.shape)
        z = mu + std * eps
        xr = _decode(p, z)
        recon = jnp.mean((xr - x) ** 2)
        kl = -0.5 * jnp.mean(1 + logvar - mu ** 2 - jnp.exp(logvar))
        return recon + kl_w * kl

    _loss_grad = jit(value_and_grad(_loss))


class VAECrossover:
    """Trainable VAE that performs crossover by latent interpolation."""

    def __init__(self, n, hidden=512, latent=8, lr=1e-3, epochs=300,
                 batch=10, kl_w=0.01, seed=0):
        if not _HAVE_JAX:
            raise RuntimeError("JAX is required for the VAE crossover.")
        self.n, self.hidden, self.latent = n, hidden, latent
        self.lr, self.epochs, self.batch, self.kl_w = lr, epochs, batch, kl_w
        self.key = random.PRNGKey(seed)
        self.params = None

    def fit(self, X):
        X = jnp.asarray(np.clip(X, 0.0, 1.0), dtype=jnp.float32)
        m = X.shape[0]
        self.key, sub = random.split(self.key)
        if self.params is None:
            self.params = _init_params(sub, self.n, self.hidden, self.latent)
        p = self.params
        # Adam state
        mom = {k: jnp.zeros_like(v) for k, v in p.items()}
        vel = {k: jnp.zeros_like(v) for k, v in p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        for ep in range(self.epochs):
            self.key, sub = random.split(self.key)
            perm = random.permutation(sub, m)
            for s in range(0, m, self.batch):
                idx = perm[s:s + self.batch]
                xb = X[idx]
                self.key, sub = random.split(self.key)
                loss, g = _loss_grad(p, xb, sub, self.kl_w)
                t += 1
                for k in p:
                    mom[k] = b1 * mom[k] + (1 - b1) * g[k]
                    vel[k] = b2 * vel[k] + (1 - b2) * g[k] ** 2
                    mhat = mom[k] / (1 - b1 ** t)
                    vhat = vel[k] / (1 - b2 ** t)
                    p[k] = p[k] - self.lr * mhat / (jnp.sqrt(vhat) + eps)
        self.params = p
        return float(loss)

    def encode_mean(self, x):
        mu, _ = _encode(self.params, jnp.asarray(x, dtype=jnp.float32))
        return np.asarray(mu)

    def decode(self, z):
        return np.asarray(_decode(self.params, jnp.asarray(z, dtype=jnp.float32)))

    def crossover(self, gi, gj, grid_shape, lam, rng):
        """Offspring = decode( lam*z_i + (1-lam)*z_j )."""
        zi = self.encode_mean(np.asarray(gi)[None, :])[0]
        zj = self.encode_mean(np.asarray(gj)[None, :])[0]
        z = lam * zi + (1 - lam) * zj
        child = self.decode(z[None, :])[0]
        return np.clip(child, 0.0, 1.0)
