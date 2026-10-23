"""Червоточины: начальные состояния, условия выхода, асимптотическое направление (M9–M11)."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

# 8-узловая квадратура Гаусса–Лежандра на [0, 1] (M10)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W


def ellis_point(ell, b=1.0):
    """Точка на меридиане (ρ, 0, 0, w) и единичный касательный e_ℓ (M9)."""
    r = np.sqrt(ell ** 2 + b ** 2)
    x = np.array([r, 0.0, 0.0, b * np.arcsinh(ell / b)])
    e_ell = np.array([ell / r, 0.0, 0.0, b / r])
    return x, e_ell


def ellis_rays(ell_c, Ls, b=1.0):
    """Лучи в плоскости xy со старта ℓ = ℓ_c внутрь, с моментами L (sin β = L/r)."""
    x, e = ellis_point(ell_c, b)
    r = np.sqrt(ell_c ** 2 + b ** 2)
    sb = np.asarray(Ls) / r
    V = -np.sqrt(1 - sb ** 2)[:, None] * e + sb[:, None] * np.array([0.0, 1.0, 0.0, 0.0])
    return np.broadcast_to(x, V.shape).copy(), V


def phi_increment(x_old, x_new):
    """Приращение полярного угла в плоскости xy (для развёртки φ вдоль пути)."""
    return jnp.arctan2(x_old[0] * x_new[1] - x_old[1] * x_new[0], x_old[0] * x_new[0] + x_old[1] * x_new[1])[None]


def ellis_ell(x, b=1.0):
    rho = jnp.linalg.norm(x[..., :3], axis=-1)
    return jnp.sign(x[..., 3]) * jnp.sqrt(jnp.maximum(rho ** 2 - b ** 2, 0.0))


def ellis_escape(ell_max, b=1.0):
    r2 = ell_max ** 2 + b ** 2
    return lambda x, v: jnp.sum(x[:3] ** 2) >= r2


def ellis_tail_dphi(ell, L, b=1.0):
    """M10: Δφ(ℓ) = ∫_0^{1/|ℓ|} L du / (√(1 + b²u²) √(1 + (b² − L²)u²)), Гаусс–Лежандр 8 узлов."""
    U = (1.0 / jnp.abs(ell))[..., None]
    L = jnp.asarray(L)[..., None]
    u = U * _GL_X
    integrand = L / (jnp.sqrt(1 + b * b * u * u) * jnp.sqrt(1 + (b * b - L * L) * u * u))
    return U[..., 0] * jnp.sum(_GL_W * integrand, -1)


def plane_frame(x, v):
    """r̂ и e⊥ (в сторону движения) в плоскости движения в R³ листа, и момент L = |q × v_q|."""
    q, vq = x[..., :3], v[..., :3]
    rh = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    ep = vq - jnp.sum(vq * rh, -1, keepdims=True) * rh
    L = jnp.linalg.norm(q, axis=-1) * jnp.linalg.norm(ep, axis=-1)
    return rh, ep / jnp.maximum(jnp.linalg.norm(ep, axis=-1, keepdims=True), 1e-300), L


def ellis_exit(x, v, b=1.0):
    """Асимптотическое направление (R³ листа), лист (знак w) и Δφ для луча, покинувшего |ℓ| ≥ ℓ_max (M10)."""
    rh, ep, L = plane_frame(x, v)
    d = ellis_tail_dphi(ellis_ell(x, b), L, b)
    return jnp.cos(d)[..., None] * rh + jnp.sin(d)[..., None] * ep, jnp.sign(x[..., 3]), d


# ---- M11 ------------------------------------------------------------------------------------

def compact_escape(rho0):
    """В плоской зоне ρ ≥ ρ₀ и при удалении от центра луч больше не вернётся: прямая."""
    return lambda x, v: (jnp.sum(x[:3] ** 2) >= rho0 ** 2) & (jnp.sum(x[:3] * v[:3]) > 0)


def compact_exit(x, v):
    """Направление на бесконечности = направление v в R³ листа (кривизна ровно ноль)."""
    vq = v[..., :3]
    rh, ep, _ = plane_frame(x, v)
    d = jnp.arccos(jnp.clip(jnp.sum(rh * vq, -1) / jnp.linalg.norm(vq, axis=-1), -1, 1))
    return vq / jnp.linalg.norm(vq, axis=-1, keepdims=True), jnp.sign(x[..., 3]), d


def compact_rays(rho_c, Ls, b=1.0, rho0=3.0, W=1.5):
    """Лучи из точки (ρ_c, 0, 0, W) плоской зоны (ρ_c ≥ ρ₀) внутрь с моментами L."""
    assert rho_c >= rho0
    x = np.array([rho_c, 0.0, 0.0, W])
    sb = np.asarray(Ls) / rho_c
    V = np.c_[-np.sqrt(1 - sb ** 2), sb, np.zeros_like(sb), np.zeros_like(sb)]
    return np.broadcast_to(x, V.shape).copy(), V
