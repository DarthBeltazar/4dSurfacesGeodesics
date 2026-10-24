"""Режим B, прототип (DD-005): однонаправленный трассировщик путей по геодезическим M во float64.

Эталон для GPU. Плоские зоны и прямые соединения (M29), BSDF в кадре M28 (Ламберт, GGX-металл), NEE к
сферическим источникам и к фону только из плоской зоны с MIS (степенная эвристика), русская рулетка.
Продление луча — интегрирование везде (в плоской зоне интегратор вырождается в sphere tracing по hcap и g)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .integrators import Config, DONE_ESCAPE
from .objects import make_hit_tracer, shading_frame, g_all
from .camera import Camera, pixel_uv, lens_rays_batch
from .cubemap import sample_bilinear0, face_to_dir, dir_to_face
from .modea import SIGMA_PIX


def bucketed(fn):
    """Обёртка jit-функции батча: первая ось дополняется до степени двойки (меньше перекомпиляций)."""
    def call(*args):
        n = len(args[0])
        m = 1 << max(int(np.ceil(np.log2(max(n, 1)))), 4)
        pad = [jnp.concatenate([jnp.asarray(a), jnp.repeat(jnp.asarray(a)[:1], m - n, 0)]) if m > n else jnp.asarray(a)
               for a in args]
        out = fn(*pad)
        return jax.tree_util.tree_map(lambda t: np.asarray(t)[:n], out)
    return call


# ---- M29: плоские зоны ------------------------------------------------------------------------

@dataclass(eq=False)
class FlatZones:
    """Листы — гиперплоскости w = const; вне шаров |q − cᵢ| < ρ₀ кривизна ровно ноль (S ≡ 0)."""
    sheets: tuple = (0.0,)
    centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    rho0: float = 0.0

    def in_flat(self, x):
        """x (..., 4) -> bool."""
        if len(self.centers) == 0:
            return jnp.ones(x.shape[:-1], bool)
        d = jnp.linalg.norm(x[..., None, :3] - jnp.asarray(self.centers), axis=-1)
        return jnp.all(d >= self.rho0 * (1 - 1e-12), axis=-1)

    def seg_clear(self, x, y):
        """Отрезок [x, y] (в одном листе) не входит ни в один шар кривизны."""
        if len(self.centers) == 0:
            return jnp.ones(x.shape[:-1], bool)
        q0, q1 = x[..., None, :3], y[..., None, :3]
        c = jnp.asarray(self.centers)
        d = q1 - q0
        t = jnp.clip(jnp.sum((c - q0) * d, -1) / jnp.maximum(jnp.sum(d * d, -1), 1e-300), 0.0, 1.0)
        dist = jnp.linalg.norm(q0 + t[..., None] * d - c, axis=-1)
        return jnp.all(dist >= self.rho0 * (1 - 1e-12), axis=-1)

    def ray_clear(self, x, d):
        """Луч x + t d, t ≥ 0, не входит ни в один шар кривизны (d — 4D, касательный к листу)."""
        if len(self.centers) == 0:
            return jnp.ones(x.shape[:-1], bool)
        q0, dq = x[..., None, :3], d[..., None, :3]
        c = jnp.asarray(self.centers)
        t = jnp.maximum(jnp.sum((c - q0) * dq, -1) / jnp.sum(dq * dq, -1), 0.0)
        dist = jnp.linalg.norm(q0 + t[..., None] * dq - c, axis=-1)
        return jnp.all(dist >= self.rho0 * (1 - 1e-12), axis=-1)


def nee_lights(objs, flat: Optional[FlatZones], tol=1e-6):
    """M34: индексы излучающих 4D-шаров, допустимых для NEE — центр на листе, шар целиком в плоской зоне
    (|c_xyz − cᵢ| ≥ ρ₀ + r). Остальные излучатели светят только через сэмплирование BSDF. То же правило — в хосте."""
    if flat is None:
        return ()
    out = []
    for k, o in enumerate(objs):
        if o.mat.kind != "emit" or not o.spec or o.spec[0] != "ball4":
            continue
        c, r = np.asarray(o.spec[1], float), o.spec[2]
        if min(abs(c[3] - s) for s in flat.sheets) >= tol:
            continue
        if len(flat.centers) and np.min(np.linalg.norm(np.asarray(flat.centers) - c[:3], axis=-1)) < flat.rho0 + r:
            continue
        out.append(k)
    return tuple(out)


def sheet_index(w):
    """Индекс листа кубмапы: 0 — w ≥ 0 (верхний / единственный), 1 — w < 0."""
    return (w < 0).astype(jnp.int32)


# ---- сцена ------------------------------------------------------------------------------------

@dataclass(eq=False)
class Scene:
    shape: object
    objs: list
    mips: list                          # кубмапы фона (листы), уровни мипов
    exit_fn: Callable                   # (x, v) -> (d∞ (…, 3), лист, _) для ушедших лучей
    flat: Optional[FlatZones] = None    # None — плоских зон нет (Эллис, S³): NEE невозможна
    escape: Optional[Callable] = None   # условие ухода для сцен без плоских зон (x, v) -> bool
    lights: tuple = ()                  # индексы NEE-источников: 4D-шары с центром на листе в плоской зоне
    env_scale: float = 1.0
    meta: dict = field(default_factory=dict)   # для GPU: scene_id (scenes.slang), exit (1 — Эллис, 2 — плоский выход), ell_max


def ahead_clear(objs, x, v):
    """Луч x + t v (прямая) не может задеть ни один объект: мимо ограничивающих сфер, от полупространств."""
    ok = jnp.asarray(True)
    for o in objs:
        if o.plane is not None:
            ax, off, sg = o.plane
            ok = ok & (sg * v[ax] >= 0)
        else:
            oc = jnp.asarray(o.center) - x
            tca = oc @ v
            d2 = oc @ oc - tca * tca
            inside = oc @ oc < o.bound ** 2
            ok = ok & ~inside & ((tca <= 0) | (d2 >= o.bound ** 2))
    return ok


def make_escape(scene: Scene):
    if scene.flat is None:
        return scene.escape
    fz = scene.flat
    return lambda x, v: fz.in_flat(x[None])[0] & fz.ray_clear(x[None], v[None])[0] & ahead_clear(scene.objs, x, v)


def make_line_blocked(objs, eps=1e-9, iters=400):
    """-> blocked(x, d, tmax, skip): прямой отрезок x + t d, t ∈ (0, tmax) задевает объект (кроме skip)?
    Sphere tracing по ограниченным объектам, полупространства — аналитически. x (N, 4), d (N, 4), tmax, skip (N,)."""
    G = g_all(objs)
    bounded = np.array([o.plane is None for o in objs])

    def one(x, d, tmax, skip):
        def gb(p):
            g = G(p)
            g = jnp.where(jnp.asarray(bounded) & (jnp.arange(len(objs)) != skip), g, jnp.inf)
            return jnp.min(g)

        def body(_, st):
            t, blocked = st
            g = gb(x + t * d)
            blocked = blocked | ((g < eps) & (t < tmax))
            return t + jnp.maximum(g, 0.0), blocked
        _, blocked = lax.fori_loop(0, iters, body, (jnp.asarray(0.0), jnp.asarray(False)))
        for i, o in enumerate(objs):
            if o.plane is not None:
                ax, off, sg = o.plane
                tp = (off - x[ax]) / jnp.where(d[ax] == 0, 1e-300, d[ax])
                blocked = blocked | ((sg * d[ax] < 0) & (tp > 0) & (tp < tmax) & (skip != i))
        return blocked
    return bucketed(jax.jit(jax.vmap(one)))


# ---- BSDF в локальных координатах (t₁, t₂, n_g): z — нормаль --------------------------------------

def ggx_D(h, a):
    c2 = h[..., 2] ** 2
    return a * a / (np.pi * (c2 * (a * a - 1) + 1) ** 2)


def ggx_lambda(w, a):
    c2 = jnp.maximum(w[..., 2] ** 2, 1e-300)
    t2 = jnp.maximum(1 - c2, 0.0) / c2
    return 0.5 * (-1 + jnp.sqrt(1 + a * a * t2))


def bsdf_eval(kind, albedo, a, wo, wi):
    """f·cos θᵢ и pdf (по телесному углу) сэмплирования BSDF. kind: 0 — Ламберт, 1 — GGX-металл."""
    co, ci = wo[..., 2], wi[..., 2]
    ok = (co > 0) & (ci > 0)
    lam_f = albedo / np.pi * ci[..., None]
    lam_p = ci / np.pi
    h = wo + wi
    h = h / jnp.maximum(jnp.linalg.norm(h, axis=-1, keepdims=True), 1e-300)
    D = ggx_D(h, a)
    Lo, Li = ggx_lambda(wo, a), ggx_lambda(wi, a)
    F = albedo + (1 - albedo) * (1 - jnp.clip(jnp.sum(wi * h, -1), 0, 1))[..., None] ** 5
    met_f = F * (D / (1 + Lo + Li) / (4 * jnp.maximum(co, 1e-300)))[..., None]
    met_p = D / (1 + Lo) / (4 * jnp.maximum(co, 1e-300))
    f = jnp.where((kind == 0)[..., None], lam_f, met_f)
    p = jnp.where(kind == 0, lam_p, met_p)
    return jnp.where(ok[..., None], f, 0.0), jnp.where(ok, p, 0.0)


def bsdf_sample(kind, albedo, a, wo, u1, u2):
    """wi, вес f·cos/pdf, pdf. Ламберт — косинусное сэмплирование; GGX — VNDF (Heitz 2018)."""
    r, ph = jnp.sqrt(u1), 2 * np.pi * u2
    wl = jnp.stack([r * jnp.cos(ph), r * jnp.sin(ph), jnp.sqrt(jnp.maximum(1 - u1, 0.0))], -1)
    Vh = jnp.stack([a * wo[..., 0], a * wo[..., 1], wo[..., 2]], -1)
    Vh = Vh / jnp.linalg.norm(Vh, axis=-1, keepdims=True)
    lsq = Vh[..., 0] ** 2 + Vh[..., 1] ** 2
    T1 = jnp.where((lsq > 0)[..., None],
                   jnp.stack([-Vh[..., 1], Vh[..., 0], jnp.zeros_like(lsq)], -1) / jnp.sqrt(jnp.maximum(lsq, 1e-300))[..., None],
                   jnp.asarray([1.0, 0.0, 0.0]))
    T2 = jnp.cross(Vh, T1)
    t1, t2 = r * jnp.cos(ph), r * jnp.sin(ph)
    s = 0.5 * (1 + Vh[..., 2])
    t2 = (1 - s) * jnp.sqrt(jnp.maximum(1 - t1 * t1, 0.0)) + s * t2
    Nh = t1[..., None] * T1 + t2[..., None] * T2 + jnp.sqrt(jnp.maximum(1 - t1 * t1 - t2 * t2, 0.0))[..., None] * Vh
    Ne = jnp.stack([a * Nh[..., 0], a * Nh[..., 1], jnp.maximum(Nh[..., 2], 0.0)], -1)
    Ne = Ne / jnp.linalg.norm(Ne, axis=-1, keepdims=True)
    wm = 2 * jnp.sum(wo * Ne, -1, keepdims=True) * Ne - wo
    wi = jnp.where((kind == 0)[..., None], wl, wm)
    f, p = bsdf_eval(kind, albedo, a, wo, wi)
    w = jnp.where((p > 0)[..., None], f / jnp.maximum(p, 1e-300)[..., None], 0.0)
    return wi, w, p


# ---- фон как источник (M29): кусочно-постоянная по текселям грубого уровня + равномерная примесь -------

ENV_UNIFORM = 0.1


def env_table(mips, level=None):
    """Таблица выбора текселя ∝ яркость·телесный угол, на листе. Возвращает (cdf (листы, 6·n²), pmf, n)."""
    if level is None:
        level = max(0, len(mips) - 1 - 5)          # n = 32
    lev = np.asarray(mips[level])[:, :, 1:-1, 1:-1]   # без рамки: (листы, 6, n, n, 3)
    n = lev.shape[2]
    c = (np.arange(n) + 0.5) / n
    S, T = np.meshgrid(c, c)
    u, v = 2 * S - 1, 2 * T - 1
    omega = 4.0 / n ** 2 / (1 + u * u + v * v) ** 1.5
    lum = lev @ np.array([0.2126, 0.7152, 0.0722])
    w = np.maximum(lum, 0) * omega[None, None]
    pmf = w.reshape(len(lev), -1) + 1e-30          # чёрный фон (компактные сцены) — равномерная таблица
    pmf = pmf / pmf.sum(1, keepdims=True)
    return np.cumsum(pmf, 1), pmf, n


def env_sample(tab, sheet, rng):
    """Направления по таблице фона (с равномерной примесью ENV_UNIFORM). sheet (N,) — индексы. -> d (N, 3)."""
    cdf, pmf, n = tab
    N = len(sheet)
    uni = rng.random(N) < ENV_UNIFORM
    d_uni = rng.normal(size=(N, 3))
    d_uni /= np.linalg.norm(d_uni, axis=1, keepdims=True)
    r = rng.random(N)
    k = np.zeros(N, int)
    for si in np.unique(sheet):
        m = sheet == si
        k[m] = np.searchsorted(cdf[si], r[m])
    k = np.minimum(k, cdf.shape[1] - 1)
    face, rem = k // (n * n), k % (n * n)
    j, i = rem // n, rem % n
    s = (i + rng.random(N)) / n
    t = (j + rng.random(N)) / n
    d_tab = np.zeros((N, 3))
    for f in range(6):
        m = face == f
        if m.any():
            d_tab[m] = face_to_dir(f, s[m], t[m])
    return np.where(uni[:, None], d_uni, d_tab)


def env_pdf(tab, sheet, d):
    """pdf по телесному углу для env_sample."""
    cdf, pmf, n = tab
    face, s, t = (np.asarray(a) for a in dir_to_face(jnp.asarray(d)))
    i = np.clip((s * n).astype(int), 0, n - 1)
    j = np.clip((t * n).astype(int), 0, n - 1)
    p = pmf[sheet, face * n * n + j * n + i]
    u, v = 2 * s - 1, 2 * t - 1
    return (1 - ENV_UNIFORM) * p * n * n * (1 + u * u + v * v) ** 1.5 / 4.0 + ENV_UNIFORM / (4 * np.pi)


# ---- сферические источники (NEE) --------------------------------------------------------------

def sphere_cone(p, c, r):
    """cos θ_max конуса сферы (3D координаты листа)."""
    d2 = np.sum((c - p) ** 2, -1)
    return np.sqrt(np.maximum(1 - r * r / d2, 0.0))


def sample_sphere_light(p, c, r, rng):
    """Направление (3D) по телесному углу сферы, точка на сфере, pdf."""
    N = len(p)
    axis = c - p
    dist = np.linalg.norm(axis, axis=1)
    axis /= dist[:, None]
    cmax = sphere_cone(p, c, r)
    ct = 1 - rng.random(N) * (1 - cmax)
    st = np.sqrt(np.maximum(1 - ct * ct, 0))
    ph = 2 * np.pi * rng.random(N)
    a = np.where(np.abs(axis[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    b1 = np.cross(axis, a)
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
    b2 = np.cross(axis, b1)
    d = ct[:, None] * axis + (st * np.cos(ph))[:, None] * b1 + (st * np.sin(ph))[:, None] * b2
    oc = p - c
    bq = np.sum(oc * d, 1)
    disc = np.maximum(bq * bq - (np.sum(oc * oc, 1) - r * r), 0)
    t = -bq - np.sqrt(disc)
    return d, t, 1.0 / (2 * np.pi * (1 - cmax))


# ---- трассировщик -----------------------------------------------------------------------------

@dataclass
class PTConfig:
    spp: int = 16
    max_depth: int = 16
    rr_depth: int = 3
    nee: bool = True
    mis: str = "power"      # power — степенная эвристика; nee — веса 1/0 (вся прямая освещённость через NEE, для проверки)
    tol: float = 1e-8
    eps_hit: float = 1e-9
    eps_off: float = 1e-7
    s_max: float = 1000.0
    chunk: int = 4096


def render(scene: Scene, cam: Camera, pc: PTConfig, seed=0, pix=None):
    """Средний цвет пикселей (k, 3) и выборочная дисперсия среднего (k, 3); pix — индексы пикселей (по умолчанию все).
    Возвращает также статистику (доля путей, оборванных потолком)."""
    rng = np.random.default_rng(seed)
    sh = scene.shape
    objs = scene.objs
    nobj = len(objs)
    if pix is None:
        pix = np.arange(cam.width * cam.height)
    U, V = (a.ravel() for a in pixel_uv(cam))
    dpix = 2 * cam.tf / cam.width
    tracer = make_hit_tracer(sh, objs, Config(tol=pc.tol, h=0.05 * sh.scale, s_max=pc.s_max),
                             escape=make_escape(scene), eps_hit=pc.eps_hit, chunk=pc.chunk)
    frames = bucketed(jax.jit(jax.vmap(lambda x, k: lax.switch(k, [
        (lambda x, o=o: shading_frame(sh.f_vec, o.g_vec, x)) for o in objs], x))))
    gradf = bucketed(jax.jit(jax.vmap(jax.grad(sh.f_vec))))
    fvec = bucketed(jax.jit(jax.vmap(sh.f_vec)))
    exit_fn = bucketed(jax.jit(scene.exit_fn))
    env = bucketed(jax.jit(lambda si, d: sample_bilinear0(scene.mips, si, d) * scene.env_scale))
    tab = env_table(scene.mips) if (scene.flat is not None and pc.nee) else None
    fz = scene.flat
    kinds = np.array([{"diffuse": 0, "metal": 1, "emit": 2}[o.mat.kind] for o in objs])
    alb = np.array([o.mat.albedo for o in objs], float)
    rough = np.array([o.mat.rough for o in objs], float)
    emit = np.array([o.mat.emit for o in objs], float)
    lights = np.asarray(scene.lights, int)
    lc = np.array([objs[k].center[:3] for k in lights]) if len(lights) else np.zeros((0, 3))
    lr = np.array([objs[k].bound for k in lights]) if len(lights) else np.zeros(0)
    lw = np.array([objs[k].center[3] for k in lights]) if len(lights) else np.zeros(0)
    blocked = make_line_blocked(objs, eps=pc.eps_hit)
    is_light = np.isin(np.arange(nobj), lights)

    def proj(x):
        for _ in range(2):
            G = np.asarray(gradf(jnp.asarray(x)))
            x = x - (np.asarray(fvec(jnp.asarray(x))) / np.sum(G * G, 1))[:, None] * G
        return x

    def tangent(x, v):
        G = np.asarray(gradf(jnp.asarray(x)))
        n = G / np.linalg.norm(G, axis=1, keepdims=True)
        v = v - np.sum(v * n, 1, keepdims=True) * n
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def mis(pa, pb):
        """Вес стратегии a при конкуренции с b; при mis = nee стратегия NEE (вызывается первой) получает 1."""
        if pc.mis == "nee":
            return np.ones_like(pa) if mis.nee_first else np.zeros_like(pa)
        return pa * pa / np.maximum(pa * pa + pb * pb, 1e-300)

    P = np.repeat(np.asarray(pix), pc.spp)
    N = len(P)
    du, dv = rng.normal(0, SIGMA_PIX, (2, N))
    if cam.aperture > 0:
        al, be = rng.normal(0, cam.aperture, (2, N))
    else:
        al = be = np.zeros(N)
    x, v = (np.array(t) for t in lens_rays_batch(sh, cam, U[P] + du * dpix, V[P] + dv * dpix, al, be))
    T = np.ones((N, 3))
    L = np.zeros((N, 3))
    alive = np.ones(N, bool)
    prev_flat = np.zeros(N, bool)     # предыдущая вершина в плоской зоне (NEE была возможна)
    prev_pdf = np.zeros(N)
    capped = 0
    for depth in range(pc.max_depth):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        x0, v0 = x[idx], v[idx]
        o = tracer(x0, v0)
        hit = o["hit"]
        esc = (o["status"] == DONE_ESCAPE) & ~hit
        capped += int(np.sum(~hit & ~esc))
        # уход в фон
        e = np.nonzero(esc)[0]
        if len(e):
            d, sheet, _ = exit_fn(jnp.asarray(o["x"][e]), jnp.asarray(o["v"][e]))
            si = np.asarray(sheet_index(jnp.asarray(sheet)))
            Le = np.asarray(env(jnp.asarray(si), d))
            w = np.ones(len(e))
            if tab is not None:
                straight = prev_flat[idx[e]] & np.asarray(fz.ray_clear(jnp.asarray(x0[e]), jnp.asarray(v0[e])))
                s0 = np.asarray(sheet_index(jnp.asarray(x0[e, 3])))
                pe = env_pdf(tab, s0, v0[e, :3])
                mis.nee_first = False
                w = np.where(straight, mis(prev_pdf[idx[e]], pe), 1.0)
            L[idx[e]] += T[idx[e]] * w[:, None] * Le
        alive[idx[~hit]] = False
        h = np.nonzero(hit)[0]
        if len(h) == 0:
            continue
        ih = idx[h]
        xh, vh, k = o["x"][h], o["v"][h], o["obj"][h]
        # излучение
        em = emit[k]
        w = np.ones(len(h))
        if tab is not None and len(lights):
            straight = prev_flat[ih] & is_light[k] & (np.abs(x0[h, 3] - xh[:, 3]) < 1e-6) & np.asarray(
                fz.in_flat(jnp.asarray(xh)) & fz.seg_clear(jnp.asarray(x0[h]), jnp.asarray(xh)))
            li = np.searchsorted(lights, k).clip(0, max(len(lights) - 1, 0))
            pl = 1.0 / (2 * np.pi * (1 - sphere_cone(x0[h, :3], lc[li], lr[li]))) / len(lights)
            mis.nee_first = False
            w = np.where(straight, mis(prev_pdf[ih], pl), 1.0)
        L[ih] += T[ih] * w[:, None] * em
        refl = kinds[k] != 2
        alive[ih[~refl]] = False
        h, ih, xh, vh, k = h[refl], ih[refl], xh[refl], vh[refl], k[refl]
        if len(h) == 0:
            continue
        n_, ng, t1, t2, png = (np.asarray(a) for a in frames(jnp.asarray(xh), jnp.asarray(k)))
        B = np.stack([t1, t2, ng], 1)                          # (m, 3, 4): локальные оси
        wo = np.einsum("mij,mj->mi", B, -vh)
        front = wo[:, 2] > 0                                   # луч пришёл снаружи
        kd, ad, ra = jnp.asarray(kinds[k]), jnp.asarray(alb[k]), jnp.asarray(rough[k])
        flat_here = np.asarray(fz.in_flat(jnp.asarray(xh))) if tab is not None else np.zeros(len(h), bool)
        xoff = proj(xh + pc.eps_off * ng)
        # NEE: сферические источники
        if tab is not None and len(lights):
            m = np.nonzero(flat_here & front)[0]
            if len(m):
                li = rng.integers(0, len(lights), len(m))
                d3, t, pdf_l = sample_sphere_light(xoff[m, :3], lc[li], lr[li], rng)
                pdf_l = pdf_l / len(lights)
                wi4 = np.c_[d3, np.zeros(len(m))]
                wil = np.einsum("mij,mj->mi", B[m], wi4)
                f, pb = (np.asarray(a) for a in bsdf_eval(kd[m], ad[m], ra[m], jnp.asarray(wo[m]), jnp.asarray(wil)))
                yv = xoff[m] + t[:, None] * wi4
                same = np.abs(xh[m, 3] - lw[li]) < 1e-6           # источник на другом листе — вклад 0
                vis = same & np.asarray(fz.seg_clear(jnp.asarray(xoff[m]), jnp.asarray(yv))) & ~blocked(
                    xoff[m], wi4, t * (1 - 1e-9), lights[li])
                mis.nee_first = True
                c = f * emit[lights[li]] / pdf_l[:, None] * mis(pdf_l, pb)[:, None]
                L[ih[m]] += np.where(vis[:, None], T[ih[m]] * c, 0.0)
        # NEE: фон
        if tab is not None:
            m = np.nonzero(flat_here & front)[0]
            if len(m):
                s0 = np.asarray(sheet_index(jnp.asarray(xh[m, 3])))
                d3 = env_sample(tab, s0, rng)
                pe = env_pdf(tab, s0, d3)
                wi4 = np.c_[d3, np.zeros(len(m))]
                wil = np.einsum("mij,mj->mi", B[m], wi4)
                f, pb = (np.asarray(a) for a in bsdf_eval(kd[m], ad[m], ra[m], jnp.asarray(wo[m]), jnp.asarray(wil)))
                vis = np.asarray(fz.ray_clear(jnp.asarray(xoff[m]), jnp.asarray(wi4))) & ~blocked(
                    xoff[m], wi4, np.full(len(m), 1e30), np.full(len(m), -1))
                Le = np.asarray(env(jnp.asarray(s0), jnp.asarray(d3)))
                mis.nee_first = True
                c = f * Le / pe[:, None] * mis(pe, pb)[:, None]
                L[ih[m]] += np.where(vis[:, None], T[ih[m]] * c, 0.0)
        # сэмплирование BSDF
        u1, u2 = rng.random((2, len(h)))
        wil, wgt, pb = (np.asarray(a) for a in bsdf_sample(kd, ad, ra, jnp.asarray(wo), jnp.asarray(u1), jnp.asarray(u2)))
        ok = front & (pb > 0)
        wi4 = np.einsum("mij,mi->mj", B, wil)
        T[ih] *= np.where(ok[:, None], wgt, 0.0)
        x[ih] = xoff
        v[ih] = tangent(xoff, wi4)
        prev_flat[ih] = flat_here
        prev_pdf[ih] = pb
        alive[ih[~ok]] = False
        if depth + 1 >= pc.rr_depth:
            q = np.clip(np.max(T[ih], 1), 0.05, 0.95)
            surv = rng.random(len(ih)) < q
            T[ih] /= q[:, None]
            alive[ih[~surv]] = False
    k = len(pix)
    Ls = L.reshape(k, pc.spp, 3)
    return Ls.mean(1), Ls.var(1, ddof=1) / pc.spp, dict(capped=capped / N)


# ---- сцены режима B (DD-005 п. 1) ---------------------------------------------------------------

def constant_env(value=1.0, N=8):
    from .cubemap import bake, build_mips
    return build_mips(bake(lambda d, s: jnp.full(d.shape, value), N))


def procedural_env(N=64):
    from .cubemap import bake, build_mips
    from .render import background
    return build_mips(bake(background, N))


def scene_hyperplane(mips=None, furnace=False):
    """Гиперплоскость w = 0 (M — евклидово R³): пол, диффузный шар, металлический шар, сферический источник.
    furnace — печь: всё диффузное с альбедо 1, без источников, фон ≡ 1."""
    from . import shapes as S
    from .objects import ball4, halfspace, Material
    from .wormholes import compact_exit
    white = Material("diffuse", (1.0, 1.0, 1.0))
    objs = [halfspace(2, -1.0, white if furnace else Material("diffuse", (0.6, 0.6, 0.6))),
            ball4((0, 3, 0, 0), 1.0, white if furnace else Material("diffuse", (0.8, 0.3, 0.2))),
            ball4((2, 4, -0.4, 0), 0.6, white if furnace else Material("metal", (0.9, 0.8, 0.5), 0.25))]
    lights = ()
    if not furnace:
        objs.append(ball4((-1.5, 2, 2.5, 0), 0.4, Material("emit", emit=(30.0, 27.0, 22.0)), name="light"))
        lights = (3,)
    mips = constant_env() if furnace else (mips or procedural_env())
    return Scene(S.SHAPES["hyperplane"], objs, mips, compact_exit, FlatZones((0.0,)), lights=lights,
                 meta=dict(scene_id=0, exit=2))


def scene_two_wormholes(mips=None, furnace=False):
    """Сцена 3 (DD-005 п. 1): две компактные червоточины M12 (c = (∓4, 0, 0), b = 1, ρ₀ = 3, W = 1.5), пол z = −3.5
    на обоих листах; верхний лист — диффузный шар, металлический тор, источник; нижний — шар и источник другого
    цвета; малый металлический шар в зоне кривизны у горловины c₁."""
    from . import shapes as S
    from .objects import ball4, halfspace, torus_in_sheet, Material
    from .reference import compact_profile
    from .wormholes import compact_exit
    sh = S.two_compact_wormholes()
    p = sh.params
    W, rho0 = p["W"], p["rho0"]
    h, _ = compact_profile(p["b"], rho0, W)
    white = Material("diffuse", (1.0, 1.0, 1.0))
    m = (lambda mat: white if furnace else mat)
    objs = [halfspace(2, -3.5, m(Material("diffuse", (0.55, 0.55, 0.55)))),
            ball4((0, -4.5, -2.7, W), 0.8, m(Material("diffuse", (0.8, 0.3, 0.2)))),
            torus_in_sheet((0, 4.5, -3.2, W), 1.2, 0.3, m(Material("metal", (0.95, 0.75, 0.4), 0.2))),
            ball4((0, 4.5, -2.7, -W), 0.8, m(Material("diffuse", (0.2, 0.7, 0.3)))),
            ball4((-4, 0, 2.0, np.sqrt(h(2.0))), 0.35, m(Material("metal", (0.9, 0.9, 0.9), 0.1)))]
    lights = ()
    if not furnace:
        objs += [ball4((0, 0, 4, W), 0.5, Material("emit", emit=(40.0, 34.0, 26.0)), name="light_up"),
                 ball4((0, 0, 4, -W), 0.5, Material("emit", emit=(15.0, 25.0, 45.0)), name="light_down")]
        lights = (5, 6)
    mips = constant_env() if furnace else (mips or procedural_env())
    fz = FlatZones((W, -W), np.array([p["c1"], p["c2"]], float), rho0)
    return Scene(sh, objs, mips, compact_exit, fz, lights=lights, meta=dict(scene_id=10, exit=2))


def scene_ellis(mips=None, furnace=False, ell_max=12.0):
    """Эллис с объектами: плоских зон нет — только сэмплирование BSDF (MIS-веса ≡ 1)."""
    from . import shapes as S
    from .objects import ball4, Material
    from .wormholes import ellis_point, ellis_escape, ellis_exit
    white = Material("diffuse", (1.0, 1.0, 1.0))
    m = (lambda mat: white if furnace else mat)
    rot = lambda p, k: np.r_[np.roll(p[:3], k), p[3]]       # точка меридиана, повёрнутая на ось y (k = 1)
    c1, _ = ellis_point(-3.0)
    c2, _ = ellis_point(3.0)
    c3, _ = ellis_point(2.0)
    objs = [ball4(c1, 0.8, m(Material("diffuse", (0.8, 0.3, 0.2)))),
            ball4(rot(c2, 1), 0.6, m(Material("metal", (0.9, 0.8, 0.5), 0.2)))]
    if not furnace:
        objs.append(ball4(rot(c3, 1) * np.array([1, -1, 1, 1]), 0.5, Material("emit", emit=(30.0, 27.0, 22.0)),
                          name="light"))
    mips = constant_env() if furnace else (mips or procedural_env())
    return Scene(S.SHAPES["ellis"], objs, mips, ellis_exit, None, escape=ellis_escape(ell_max),
                 meta=dict(scene_id=8, exit=1, ell_max=ell_max))


def _on_surface(shape, guesses):
    from .shapes import project_points
    return np.asarray(project_points(shape, np.asarray(guesses, float)))


def scene_compact_space(name="S3", mips=None):
    """Компактные пространства этапа 5 (свободный полёт): S³, «сферитор», «ториосфера», «тигр» (T³). Фона нет (лучи замыкаются),
    плоских зон нет — свет только от излучающих шаров через сэмплирование BSDF. Объекты — 4D-шары с центрами на M."""
    from . import shapes as S
    from .objects import ball4, Material
    from .wormholes import compact_exit
    sh = S.SHAPES[name]
    if name == "S3":
        pts = _on_surface(sh, [[0, 0, 0, 1], [0, 0, 0, -1], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0],
                               [0.6, 0.6, -0.5, 0.2]])
        rad = 0.22
    elif name == "tiger":      # T³: (|(x,y)| − R₁)² + (|(z,w)| − R₂)² = r², R₁ = 1.2, R₂ = 0.8, r = 0.3
        a = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        g = [[(1.2 + 0.3 * np.cos(2 * t)) * np.cos(t), (1.2 + 0.3 * np.cos(2 * t)) * np.sin(t),
              (0.8 + 0.3 * np.sin(2 * t)) * np.cos(3 * t), (0.8 + 0.3 * np.sin(2 * t)) * np.sin(3 * t)] for t in a]
        pts = _on_surface(sh, g)
        rad = 0.12
    else:
        a = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        g = [[1.4 * np.cos(t), 1.4 * np.sin(t), 0.25 * np.cos(3 * t), 0.25 * np.sin(2 * t)] for t in a]
        pts = _on_surface(sh, g)
        rad = 0.15
    mats = [Material("emit", emit=(12.0, 10.0, 7.0)), Material("diffuse", (0.85, 0.25, 0.2)),
            Material("diffuse", (0.2, 0.8, 0.3)), Material("metal", (0.9, 0.9, 0.95), 0.15),
            Material("emit", emit=(4.0, 7.0, 14.0)), Material("diffuse", (0.85, 0.85, 0.85)),
            Material("metal", (0.95, 0.7, 0.35), 0.3)]
    objs = [ball4(p, rad * (1.4 if m.kind == "emit" else 1.0), m) for p, m in zip(pts, mats)]
    sid = {"S3": 1, "spheritorus": 5, "torisphere": 6, "tiger": 7}[name]
    return Scene(sh, objs, mips or constant_env(0.0), compact_exit, None, escape=None,
                 meta=dict(scene_id=sid, exit=2, ell_max=1e9, s_max=8 * np.pi))     # непопавшие лучи — чёрные


def scene_s3(mips=None):
    return scene_compact_space("S3", mips)
