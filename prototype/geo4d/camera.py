"""Камера на M: касательный базис, pinhole, тонкая линза и начальные данные Якоби (M16)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

# Порядок полей Якоби в канале E: апертура вдоль right, up; пиксель вправо, вверх (M16, M17)
FIELDS = ("ap_r", "ap_u", "pix_r", "pix_u")


@dataclass
class Camera:
    x: np.ndarray        # точка на M
    f: np.ndarray        # forward
    r: np.ndarray        # right
    u: np.ndarray        # up
    fov: float = 90.0    # горизонтальный, градусы
    width: int = 64
    height: int = 64
    focus: float = 5.0   # F: расстояние до фокусной плоскости вдоль forward
    aperture: float = 0.0  # σ_ap гауссовой апертуры на ось (0 — pinhole)

    @property
    def tf(self):
        return np.tan(np.radians(self.fov) / 2)

    @property
    def theta_pix(self):
        return 2 * self.tf / self.width


def make_camera(shape, x, forward, up_hint, right_hint=None, **kw):
    """Касательный ортонормированный базис в x: forward, up — по подсказкам; right — по подсказке
    или дополнение с det(n, f, r, u) > 0."""
    g = np.asarray(jax.grad(shape.f_vec)(jnp.asarray(x, float)))
    n = g / np.linalg.norm(g)

    def orth(a, basis):
        """Двойной Грам–Шмидт; при вырожденной подсказке — первая координатная ось с устойчивым остатком."""
        for cand in [np.asarray(a, float), *np.eye(4)]:
            c = cand.copy()
            for _ in range(2):
                for b in basis:
                    c -= (c @ b) * b
            if np.linalg.norm(c) > 1e-6 * max(np.linalg.norm(cand), 1e-300):
                return c / np.linalg.norm(c)
        raise ValueError("не удалось построить касательный базис")

    f = orth(forward, [n])
    u = orth(up_hint, [n, f])
    if right_hint is not None:
        r = orth(right_hint, [n, f, u])
    else:
        r = orth(np.linalg.svd(np.stack([n, f, u]))[2][-1], [n, f, u])
        if np.linalg.det(np.stack([n, f, r, u])) < 0:
            r = -r
    return Camera(np.asarray(x, float), f, r, u, **kw)


def pixel_uv(cam: Camera):
    """Координаты U, V центров пикселей (M16), формы (H, W)."""
    tf = cam.tf
    U = ((np.arange(cam.width) + 0.5) / cam.width * 2 - 1) * tf
    V = (1 - (np.arange(cam.height) + 0.5) / cam.height * 2) * tf * cam.height / cam.width
    return np.meshgrid(U, V)


def pixel_dirs(cam: Camera, U, V):
    D = cam.f + U[..., None] * cam.r + V[..., None] * cam.u
    return D / np.linalg.norm(D, axis=-1, keepdims=True)


def jacobi_init(cam: Camera, U, V):
    """M16: E0 (..., 2, 4, 4) для полей (ap_r, ap_u, pix_r, pix_u) и центральные направления d₀."""
    D = cam.f + U[..., None] * cam.r + V[..., None] * cam.u
    nD = np.linalg.norm(D, axis=-1, keepdims=True)
    d0 = D / nD
    perp = lambda e: e - (d0 @ e)[..., None] * d0
    P_len = cam.focus / (d0 @ cam.f)[..., None]               # |P| = F/(d₀·f)
    dpix = 2 * cam.tf / cam.width
    Jx = np.stack([np.broadcast_to(cam.r, d0.shape), np.broadcast_to(cam.u, d0.shape),
                   np.zeros_like(d0), np.zeros_like(d0)], -2)
    Jp = np.stack([-perp(cam.r) / P_len, -perp(cam.u) / P_len,
                   dpix * perp(cam.r) / nD, dpix * perp(cam.u) / nD], -2)
    return np.stack([Jx, Jp], -3), d0


def project_to_M(shape, x, iters=3):
    g = jax.grad(shape.f_vec)
    for _ in range(iters):
        G = g(x)
        x = x - shape.f_vec(x) * G / (G @ G)
    return x


def lens_ray(shape, cam: Camera, U, V, alpha, beta, iters=3):
    """Эталонная конструкция тонкой линзы (M16) для одного луча: точка линзы a = α r + β u."""
    d0 = pixel_dirs(cam, np.asarray(U), np.asarray(V))
    a = alpha * cam.r + beta * cam.u
    target = cam.focus / (d0 @ cam.f) * d0 - a
    x = project_to_M(shape, jnp.asarray(cam.x + a), iters)
    g = jax.grad(shape.f_vec)(x)
    n = g / jnp.linalg.norm(g)
    v = jnp.asarray(target) - (jnp.asarray(target) @ n) * n
    return np.asarray(x), np.asarray(v / jnp.linalg.norm(v))


def transverse_basis(v, n):
    """M17: b₁, b₂ — Грам–Шмидт координатных осей, наименее совпадающих с v и n."""
    order = np.argsort(np.abs(v) + np.abs(n))
    out = []
    for k in order:
        e = np.eye(4)[k]
        for b in [v, n, *out]:
            e = e - (e @ b) * b
        if np.linalg.norm(e) > 1e-6:
            out.append(e / np.linalg.norm(e))
        if len(out) == 2:
            break
    return np.stack(out)
