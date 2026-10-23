"""Матрица переноса луча (M17): поля Якоби → производные точки и направления выхода."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .jet import seed15
from .jacobi import shape_operator
from .camera import jacobi_init, transverse_basis


def ambient_dv(f, x, v, J):
    """M17: ∂v = J′ − ⟨SJ, v⟩n — нормальная поправка (возвращается слагаемое −⟨SJ, v⟩n). x, v: (4,), J: (k, 4)."""
    Jt = f(seed15(x))
    S, n = shape_operator(Jt.g, Jt.H)
    return -(J @ (S @ v))[:, None] * n


def state_derivatives(f, x, v, E):
    """Производные (∂x, ∂v) по каждому полю в объемлющих координатах: (k, 4), (k, 4)."""
    return E[0], E[1] + ambient_dv(f, x, v, E[0])


def exit_dir_derivatives(exit_fn, f, x, v, E):
    """M17: ∂d∞ по каждому полю — jvp функции выхода вдоль (J, J′ − ⟨SJ,v⟩n). Возвращает d∞ (3,), ∂d∞ (k, 3)."""
    dx, dv = state_derivatives(f, x, v, E)
    d = lambda xx, vv: exit_fn(xx, vv)[0]
    d0 = d(x, v)
    dd = jax.vmap(lambda a, b: jax.jvp(d, (x, v), (a, b))[1])(dx, dv)
    return d0, dd


def transfer_matrix_4x4(f, x, v, E):
    """M17: 4×4 (∂x⊥ по b₁,b₂; ∂v⊥ по b₁,b₂) × (ap_r, ap_u, pix_r, pix_u) в точке x, v."""
    x, v = np.asarray(x), np.asarray(v)
    dx, dv = (np.asarray(t) for t in state_derivatives(f, jnp.asarray(x), jnp.asarray(v), jnp.asarray(E)))
    g = np.asarray(jax.grad(lambda p: f(tuple(p[k] for k in range(4))))(jnp.asarray(x)))
    B = transverse_basis(v, g / np.linalg.norm(g))
    return np.concatenate([B @ dx.T, B @ dv.T], 0)
