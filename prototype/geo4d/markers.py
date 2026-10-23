"""Светящиеся маркеры для компактных фигур (M21, M22): трассировка до попадания с шагом ≤ g(x)."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .integrators import make_integrator, Config, DONE_ESCAPE

MARKER_COLORS = np.array([[1.0, 0.25, 0.15], [0.2, 0.9, 0.3], [0.25, 0.45, 1.0], [1.0, 0.85, 0.2],
                          [0.9, 0.3, 0.9], [0.2, 0.9, 0.9]])
NO_HIT = np.array([0.08, 0.08, 0.08])      # луч не попал ни во что (потолок длины пути)
STEP_CAP = np.array([1.0, 0.0, 1.0])      # потолок числа шагов


def ball_markers(centers, chord_radii):
    """M21: 4D-шары g_i(p) = |p − m_i| − c_i. Возвращает g(x) -> (k,)."""
    C = jnp.asarray(np.asarray(centers, float))
    c = jnp.asarray(np.asarray(chord_radii, float))
    return lambda x: jnp.linalg.norm(x[None, :] - C, axis=-1) - c


def cylinder_intrinsic_markers(R, centers_szw, radii):
    """M22: маркеры на S¹×R² — евклидовы шары в развёртке (σ = Rθ, z, w), размноженные с периодом 2πR.
    g — внутреннее расстояние до шара (нижняя оценка длины пути до него): шаг ≤ g безопасен."""
    Cm = jnp.asarray(np.asarray(centers_szw, float))
    rad = jnp.asarray(np.asarray(radii, float))

    def g(x):
        sigma = R * jnp.arctan2(x[1], x[0])
        ds = jnp.mod(sigma - Cm[:, 0] + np.pi * R, 2 * np.pi * R) - np.pi * R
        return jnp.sqrt(ds ** 2 + (x[2] - Cm[:, 1]) ** 2 + (x[3] - Cm[:, 2]) ** 2) - rad
    return g


def trace_markers(shape, X, V, gfun, cfg: Config, eps_hit=1e-7):
    """Интегрирует лучи до g_min < eps_hit или s_max. Возвращает (id маркера или −1, s, статусы, попытки шага)."""
    gmin = lambda x: jnp.min(gfun(x))
    run = make_integrator(shape, cfg, escape=lambda x, v: gmin(x) < eps_hit, gcap=lambda x: jnp.maximum(gmin(x), 0.0))
    o = run(jnp.asarray(X), jnp.asarray(V))
    status = np.asarray(o["status"])
    ids = np.asarray(jnp.argmin(jax.vmap(gfun)(o["x"]), axis=1))
    ids = np.where(status == DONE_ESCAPE, ids, -1)
    return ids, np.asarray(o["s"]), status, np.asarray(o["nacc"] + o["nrej"])


def shade(ids, status):
    col = np.where((ids >= 0)[:, None], MARKER_COLORS[np.maximum(ids, 0) % len(MARKER_COLORS)], NO_HIT)
    return np.where((status == 3)[:, None], STEP_CAP, col)
