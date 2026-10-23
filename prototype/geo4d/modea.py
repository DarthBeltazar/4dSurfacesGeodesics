"""Режим A (DD-003): фон через матрицу переноса Якоби + фильтрованная выборка кубмапы; детектор нелинейности
(M20) и стохастический фолбэк; эталонный стохастический рендерер с теми же фильтрами."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from .integrators import make_integrator, Config, DONE_ESCAPE
from .camera import Camera, pixel_uv, jacobi_init, lens_rays_batch
from .transfer import exit_dir_derivatives
from .cubemap import sample_ellipse, sample_bilinear0, sample_gauss_exact
from .render import MISS

SIGMA_PIX = 0.5   # M18: гауссов фильтр пикселя, пиксели


@dataclass
class Detector:
    """M20: пороги детектора нелинейности."""
    det_lo: float = 0.25
    det_hi: float = 4.0
    tau: float = 0.25          # невязка второго порядка, в θ_pix


def _sheet_index(sheet):
    return jnp.where(sheet > 0, 0, 1)


def trace_central(shape, cam: Camera, cfg: Config, escape, exit_fn, chunk=8192):
    """Центральные лучи пикселей с 4 полями Якоби → d∞, ∂d∞ (4, 3), лист, статус, проходы Jet15."""
    U, V = pixel_uv(cam)
    E0, d0 = jacobi_init(cam, U, V)
    E0, d0 = E0.reshape(-1, 2, 4, 4), d0.reshape(-1, 4)
    run = make_integrator(shape, cfg.with_(jacobi=4), escape=escape)
    derivs = jax.jit(jax.vmap(lambda x, v, E: exit_dir_derivatives(exit_fn, shape.f, x, v, E)))
    sheet_fn = jax.jit(lambda x, v: exit_fn(x, v)[1])
    out = {k: [] for k in ("d", "dd", "sheet", "status", "nsdf15")}
    X0 = np.broadcast_to(cam.x, d0.shape)
    for i in range(0, len(d0), chunk):
        o = run(X0[i:i + chunk], d0[i:i + chunk], E0[i:i + chunk])
        d, dd = derivs(o["x"], o["v"], o["E"])
        out["d"].append(np.asarray(d))
        out["dd"].append(np.asarray(dd))
        out["sheet"].append(np.asarray(sheet_fn(o["x"], o["v"])))
        out["status"].append(np.asarray(o["status"]))
        out["nsdf15"].append(np.asarray(o["nsdf15"]))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["E0"] = E0
    return res


def footprint(dd, sigma_ap):
    """M18: Σ = σ_pix² A_pix A_pixᵀ + σ_ap² A_ap A_apᵀ → (ось₊, σ₊, ось₋, σ₋)."""
    A_ap, A_pix = dd[:, 0:2], dd[:, 2:4]                      # (N, 2, 3)
    Sig = SIGMA_PIX ** 2 * np.einsum("nki,nkj->nij", A_pix, A_pix) + \
        sigma_ap ** 2 * np.einsum("nki,nkj->nij", A_ap, A_ap)
    w, Q = np.linalg.eigh(Sig)                                 # по возрастанию; w[0] ≈ 0 (вдоль d)
    return Q[:, :, 2], np.sqrt(np.maximum(w[:, 2], 0)), Q[:, :, 1], np.sqrt(np.maximum(w[:, 1], 0))


def detector_flags(cam: Camera, tr, det: Detector):
    """M20: флаги фолбэка (H·W,) и диагностические величины."""
    H, W = cam.height, cam.width
    d = tr["d"].reshape(H, W, 3)
    A_pix = tr["dd"][:, 2:4]
    gram = np.einsum("nki,nli->nkl", A_pix, A_pix)
    Jp0 = tr["E0"][:, 1, 2:4]                                  # J′(0) пиксельных полей (плоский pinhole)
    gram0 = np.einsum("nki,nli->nkl", Jp0, Jp0)
    rho = np.sqrt(np.abs(np.linalg.det(gram))) / np.sqrt(np.abs(np.linalg.det(gram0)))
    bad = (rho < det.det_lo) | (rho > det.det_hi)
    ok = (tr["status"] == DONE_ESCAPE).reshape(H, W)
    sheet = tr["sheet"].reshape(H, W)
    A = A_pix.reshape(H, W, 2, 3)
    resid = np.zeros((H, W))
    flag_nb = np.zeros((H, W), bool)
    for di, dj, k, sg in [(1, 0, 0, +1), (-1, 0, 0, -1), (0, -1, 1, +1), (0, 1, 1, -1)]:   # вправо, влево, вверх, вниз
        jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        nj, ni = jj + dj, ii + di
        inside = (nj >= 0) & (nj < H) & (ni >= 0) & (ni < W)
        njc, nic = np.clip(nj, 0, H - 1), np.clip(ni, 0, W - 1)
        pred = d + sg * A[:, :, k]
        pred /= np.linalg.norm(pred, axis=-1, keepdims=True)
        r = np.linalg.norm(pred - d[njc, nic], axis=-1) / cam.theta_pix
        resid = np.maximum(resid, np.where(inside, r, 0))
        flag_nb |= inside & ((sheet[njc, nic] != sheet) | ~ok[njc, nic])
    flags = bad.reshape(H, W) | (resid > det.tau) | flag_nb | ~ok
    return flags.ravel(), dict(rho=rho, resid=resid.ravel())


def shade_mode_a(tr, mips, sigma_ap, exact_filter=False):
    """Цвет режима A. exact_filter — гауссов фильтр квадратурой по уровню 0 вместо мипов (для разложения ошибки)."""
    a1, s1, a2, s2 = footprint(tr["dd"], sigma_ap)
    args = (mips, _sheet_index(jnp.asarray(tr["sheet"])), jnp.asarray(tr["d"]),
            jnp.asarray(a1), jnp.asarray(s1), jnp.asarray(a2), jnp.asarray(s2))
    if exact_filter:
        col, L, n = sample_gauss_exact(*args), np.zeros(len(a1)), np.zeros(len(a1))
    else:
        col, L, n = jax.jit(sample_ellipse)(*args)
    col = np.where((tr["status"] == DONE_ESCAPE)[:, None], np.asarray(col), MISS)
    return col, np.asarray(L), np.asarray(n)


def stochastic_pixels(shape, cam: Camera, pix, nspp, cfg: Config, escape, exit_fn, mips, seed=0,
                      aperture="gauss", chunk=65536, return_samples=False, return_dirs=False):
    """Эталон/фолбэк: nspp лучей на пиксель (гаусс пикселя σ_pix; апертура — гаусс σ_ap на ось или равномерный
    диск радиуса 2σ_ap), мип 0 билинейно. pix — индексы пикселей (k,). Возвращает (k, 3) или (k, nspp, 3);
    при return_dirs — направления выхода (k, nspp, 3), листы и статусы (k, nspp) вместо цветов."""
    rng = np.random.default_rng(seed)
    U, V = (a.ravel() for a in pixel_uv(cam))
    dpix = 2 * cam.tf / cam.width
    run = make_integrator(shape, cfg, escape=escape)

    @jax.jit
    def shade(x, v, status):
        d, sheet, _ = exit_fn(x, v)
        c = sample_bilinear0(mips, _sheet_index(sheet), d)
        return jnp.where((status == DONE_ESCAPE)[:, None], c, jnp.asarray(MISS)), d, sheet

    P = np.repeat(np.asarray(pix), nspp)
    du, dv = rng.normal(0, SIGMA_PIX, (2, len(P)))
    if cam.aperture > 0 and aperture == "gauss":
        al, be = rng.normal(0, cam.aperture, (2, len(P)))
    elif cam.aperture > 0:
        r = 2 * cam.aperture * np.sqrt(rng.random(len(P)))
        ph = 2 * np.pi * rng.random(len(P))
        al, be = r * np.cos(ph), r * np.sin(ph)
    else:
        al = be = np.zeros(len(P))
    cols, dirs = np.zeros((len(P), 3)), np.zeros((len(P), 3))
    sheets, stats = np.zeros(len(P)), np.zeros(len(P), int)
    for i in range(0, len(P), chunk):
        sl = slice(i, i + chunk)
        x, v = lens_rays_batch(shape, cam, U[P[sl]] + du[sl] * dpix, V[P[sl]] + dv[sl] * dpix, al[sl], be[sl])
        o = run(x, v)
        c, d, sh = shade(o["x"], o["v"], o["status"])
        cols[sl], dirs[sl], sheets[sl], stats[sl] = np.asarray(c), np.asarray(d), np.asarray(sh), np.asarray(o["status"])
    k = len(pix)
    if return_dirs:
        return dirs.reshape(k, nspp, 3), sheets.reshape(k, nspp), stats.reshape(k, nspp)
    cols = cols.reshape(k, nspp, 3)
    return cols if return_samples else cols.mean(1)


def tonemap01(rgb):
    x = np.asarray(rgb)
    return np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def psnr(a, b):
    """PSNR на тонмапленных значениях [0, 1] (DD-003)."""
    mse = np.mean((tonemap01(a) - tonemap01(b)) ** 2)
    return 10 * np.log10(1.0 / mse)
