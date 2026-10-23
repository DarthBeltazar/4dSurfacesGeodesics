"""Рендер этапа 1 (только фон): pinhole-камера в касательном пространстве, процедурные фоны по листам."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .integrators import make_integrator, Config, DONE_ESCAPE, DONE_NMAX

# Цветные «полюса»: направление, цвет (линейный RGB)
POLES = [((1, 0, 0), (1.0, 0.15, 0.1)), ((-1, 0, 0), (0.1, 0.9, 0.9)),
         ((0, 1, 0), (0.15, 1.0, 0.15)), ((0, -1, 0), (1.0, 0.2, 1.0)),
         ((0, 0, 1), (1.0, 0.95, 0.2)), ((0, 0, -1), (0.2, 0.3, 1.0))]
SHEET_BASE = {+1: ((0.05, 0.08, 0.18), (0.25, 0.35, 0.6)),     # верхний лист: холодный
              -1: ((0.2, 0.08, 0.03), (0.65, 0.4, 0.2))}       # нижний лист: тёплый
MISS = np.array([1.0, 0.0, 1.0])  # луч не завершился (потолок шагов)


def background(d, sheet, grid_deg=15.0, line_deg=0.5, pole_deg=8.0):
    """Процедурный HDR-фон: градиент по широте + сетка + цветные полюса. d: (..., 3), sheet: (...) ±1."""
    d = d / jnp.linalg.norm(d, axis=-1, keepdims=True)
    lat = jnp.degrees(jnp.arcsin(jnp.clip(d[..., 2], -1, 1)))
    lon = jnp.degrees(jnp.arctan2(d[..., 1], d[..., 0]))
    t = (0.5 + 0.5 * d[..., 2])[..., None]
    out = 0.0
    for s, (lo, hi) in SHEET_BASE.items():
        col = (1 - t) * jnp.asarray(lo) + t * jnp.asarray(hi)
        out = out + jnp.where((sheet == s)[..., None], col, 0.0)
    dl = jnp.abs((lat + grid_deg / 2) % grid_deg - grid_deg / 2)
    dn = jnp.abs((lon + grid_deg / 2) % grid_deg - grid_deg / 2) * jnp.cos(jnp.radians(lat))
    line = jnp.clip(1.0 - jnp.minimum(dl, dn) / line_deg, 0.0, 1.0)[..., None]
    out = out * (1 - line) + line * 0.9
    for p, c in POLES:
        ang = jnp.degrees(jnp.arccos(jnp.clip(d @ jnp.asarray(p, d.dtype), -1, 1)))
        w = jnp.clip((pole_deg - ang) / 1.0, 0.0, 1.0)[..., None]
        out = out * (1 - w) + w * 2.0 * jnp.asarray(c)
    return out


def pinhole_dirs(width, height, fov_deg, e_fwd, e_right, e_up):
    """Направления лучей в R⁴ (касательные к M в точке камеры), θ_pix — угловой размер пикселя в центре."""
    tf = np.tan(np.radians(fov_deg) / 2)
    u = (np.arange(width) + 0.5) / width * 2 - 1
    v = 1 - (np.arange(height) + 0.5) / height * 2
    U, Vv = np.meshgrid(u * tf, v * tf * height / width)
    D = e_fwd + U[..., None] * e_right + Vv[..., None] * e_up
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    return D.reshape(-1, 4), 2 * tf / width


def tonemap(rgb):
    x = np.asarray(rgb)
    x = x / (1 + x)                       # Рейнхард
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def render_wormhole(shape, x_start, dirs, cfg: Config, escape, exit_fn, chunk=32768, active=None):
    """Интегрирует лучи до выхода в плоскую/асимптотическую зону и красит фоном листа выхода.

    x_start: (4,) или (N, 4); active: (N,) bool — False для лучей, не входящих в область кривизны
    (их направление и лист берутся из exit_fn в стартовой точке без интегрирования, 0 вызовов SDF).
    Возвращает цвета, попытки шага, проходы SDF, статусы, направления выхода и листы.
    """
    run = make_integrator(shape, cfg, escape=escape)
    X_all = np.broadcast_to(np.asarray(x_start, float), dirs.shape)
    act = np.ones(len(dirs), bool) if active is None else np.asarray(active)

    @jax.jit
    def shade(x, v, status):
        d, sheet, _ = exit_fn(x, v)
        col = background(d, sheet)
        return jnp.where((status == DONE_ESCAPE)[:, None], col, jnp.asarray(MISS)), d, sheet

    N = len(dirs)
    steps, nsdf = np.zeros(N, int), np.zeros(N, int)
    status = np.full(N, DONE_ESCAPE)
    Xo, Vo = X_all.copy(), np.asarray(dirs, float).copy()
    idx = np.nonzero(act)[0]
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        o = run(jnp.asarray(X_all[j]), jnp.asarray(dirs[j]))
        Xo[j], Vo[j] = np.asarray(o["x"]), np.asarray(o["v"])
        steps[j] = np.asarray(o["nacc"] + o["nrej"])
        nsdf[j] = np.asarray(o["nsdf"])
        status[j] = np.asarray(o["status"])
    cols, d, sheet = (np.asarray(t) for t in shade(jnp.asarray(Xo), jnp.asarray(Vo), jnp.asarray(status)))
    return cols, steps, nsdf, status, d, sheet
