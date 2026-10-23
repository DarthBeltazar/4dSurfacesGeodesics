"""Общие параметры тестов U2–U6."""
import numpy as np
import jax.numpy as jnp

from geo4d.integrators import make_integrator, Config
from geo4d.shapes import sample

TOL = 1e-10
NRAYS = 8


def path_length(shape):
    return min(10.0, 10.0 * shape.scale)


def run(shape, X, V, L, tol=TOL, **kw):
    cfg = Config(method="bs32", tol=tol, s_max=L, **kw)
    return make_integrator(shape, cfg)(jnp.asarray(X), jnp.asarray(V))


def rays(shape, n=NRAYS, seed=0):
    return sample(shape, n, seed=seed)


def maxnorm(a, b):
    return float(np.max(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)))
