"""Правая часть геодезической (M2) и проекция (M3) через один проход SDF на Jet7."""
import jax.numpy as jnp

from .jet import seed7


def geodesic_rhs(f, alpha):
    """Возвращает R(x, v) -> (v̇, f, ∇f). Один вызов = один проход SDF на Jet7."""
    def R(x, v):
        J = f(seed7(x, v))                       # M4: f = a, ∇f·v = b, vᵀHv = 2c, ∇f = d
        g = J.d
        lam = -(2.0 * J.c + 2.0 * alpha * J.b + alpha * alpha * J.a) / jnp.sum(g * g, -1)
        return lam[..., None] * g, J.a, g       # M2
    return R


def project(x, v, fval, g):
    """M3: шаг Ньютона для x вдоль ∇f, проекция v на касательную плоскость и нормировка."""
    g2 = jnp.sum(g * g, -1, keepdims=True)
    x = x - fval[..., None] * g / g2
    n = g / jnp.sqrt(g2)
    v = v - jnp.sum(v * n, -1, keepdims=True) * n
    return x, v / jnp.linalg.norm(v, axis=-1, keepdims=True)
