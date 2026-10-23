"""U1: jet-числа (M4, M5) против jax.grad / jax.hessian во float64 в 1000 точках у поверхности."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d.jet import seed7, seed15, ops
from geo4d.shapes import SHAPES, sample, two_compact_wormholes

ALL = dict(SHAPES, two_compact_wh=two_compact_wormholes())
N = 1000
RTOL = 1e-10


def _near_surface(shape, seed=1):
    P, _ = sample(shape, N, seed=seed)
    rng = np.random.default_rng(seed + 100)
    P = P + 0.05 * shape.scale * rng.normal(size=P.shape)
    V = rng.normal(size=P.shape)
    return jnp.asarray(P), jnp.asarray(V)


def _close(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    err = np.abs(got - ref) / np.maximum(1.0, np.abs(ref))
    assert np.all(np.isfinite(got)), "нечисловые значения"
    assert err.max() < RTOL, f"max rel err {err.max():.2e}"


@pytest.mark.parametrize("name", sorted(ALL))
def test_jet7_vs_autodiff(name):
    sh = ALL[name]
    P, V = _near_surface(sh)
    J = jax.vmap(lambda x, v: sh.f(seed7(x, v)))(P, V)
    G = jax.vmap(jax.grad(sh.f_vec))(P)
    H = jax.vmap(jax.hessian(sh.f_vec))(P)
    _close(J.a, jax.vmap(sh.f_vec)(P))
    _close(J.d, G)
    _close(J.b, jnp.sum(G * V, -1))
    _close(2 * J.c, jnp.einsum("ni,nij,nj->n", V, H, V))


@pytest.mark.parametrize("name", sorted(ALL))
def test_jet15_vs_autodiff(name):
    sh = ALL[name]
    P, _ = _near_surface(sh)
    J = jax.vmap(lambda x: sh.f(seed15(x)))(P)
    _close(J.a, jax.vmap(sh.f_vec)(P))
    _close(J.g, jax.vmap(jax.grad(sh.f_vec))(P))
    _close(J.H, jax.vmap(jax.hessian(sh.f_vec))(P))


def test_smin_jets():
    def f(p):
        return ops.smin(ops.length(*p) - 1.0, p[3] - 0.3, 0.2)

    rng = np.random.default_rng(0)
    P, V = jnp.asarray(rng.normal(size=(N, 4))), jnp.asarray(rng.normal(size=(N, 4)))
    fv = lambda x: f(tuple(x[k] for k in range(4)))
    J = jax.vmap(lambda x, v: f(seed7(x, v)))(P, V)
    _close(J.d, jax.vmap(jax.grad(fv))(P))
    H = jax.vmap(jax.hessian(fv))(P)
    _close(2 * J.c, jnp.einsum("ni,nij,nj->n", V, H, V))
    _close(jax.vmap(lambda x: f(seed15(x)))(P).H, H)
