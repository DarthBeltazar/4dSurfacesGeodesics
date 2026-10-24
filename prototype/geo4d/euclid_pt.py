"""Независимый евклидов трассировщик путей в R³ (DD-005, план проверки п. 5) — эталон для сцены «гиперплоскость».

Без кода M: аналитические пересечения со сферами и полупространствами, нормали — аналитически, кадр — любой
ортонормированный (BSDF изотропны). Общими с `pathtrace` остаются только проверенные отдельно BSDF и выборка фона."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .pathtrace import bsdf_eval, bsdf_sample, env_table, env_sample, env_pdf, sample_sphere_light, sphere_cone, PTConfig
from .cubemap import sample_bilinear0
from .camera import pixel_uv
from .modea import SIGMA_PIX

KIND = {"diffuse": 0, "metal": 1, "emit": 2}


def from_scene(scene):
    """Сфера/полупространство из объектов сцены на гиперплоскости w = 0 (4D-шары с центром при w = 0)."""
    sph, pln = [], []
    for i, o in enumerate(scene.objs):
        if o.plane is not None:
            pln.append((i, *o.plane))
        else:
            assert abs(o.center[3]) < 1e-12
            sph.append((i, np.asarray(o.center[:3], float), float(o.bound)))
    return sph, pln


def intersect(sph, pln, o, d, eps=1e-9):
    """Ближайшее пересечение лучей o + t d (t > eps). -> t (N,), индекс объекта (N, −1 — нет), нормаль (N, 3)."""
    N = len(o)
    tbest = np.full(N, np.inf)
    ibest = np.full(N, -1)
    nrm = np.zeros((N, 3))
    for i, c, r in sph:
        oc = o - c
        b = np.sum(oc * d, 1)
        disc = b * b - (np.sum(oc * oc, 1) - r * r)
        sq = np.sqrt(np.maximum(disc, 0))
        t = np.where(-b - sq > eps, -b - sq, -b + sq)
        ok = (disc > 0) & (t > eps) & (t < tbest)
        tbest = np.where(ok, t, tbest)
        ibest = np.where(ok, i, ibest)
        p = o + t[:, None] * d
        nrm = np.where(ok[:, None], (p - c) / r, nrm)
    for i, ax, off, sg in pln:
        dd = d[:, ax]
        t = (off - o[:, ax]) / np.where(dd == 0, 1e-300, dd)
        ok = (t > eps) & (t < tbest) & (sg * dd < 0)
        tbest = np.where(ok, t, tbest)
        ibest = np.where(ok, i, ibest)
        e = np.zeros(3)
        e[ax] = sg
        nrm = np.where(ok[:, None], e, nrm)
    return tbest, ibest, nrm


def frame(n):
    a = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return np.stack([t1, np.cross(n, t1), n], 1)          # строки: t₁, t₂, n


def render(scene, cam, pc: PTConfig, seed=0):
    rng = np.random.default_rng(seed)
    sph, pln = from_scene(scene)
    objs = scene.objs
    kinds = np.array([KIND[o.mat.kind] for o in objs])
    alb = np.array([o.mat.albedo for o in objs], float)
    rough = np.array([o.mat.rough for o in objs], float)
    emit = np.array([o.mat.emit for o in objs], float)
    lights = np.asarray(scene.lights, int)
    lc = np.array([objs[k].center[:3] for k in lights]).reshape(-1, 3)
    lr = np.array([objs[k].bound for k in lights])
    tab = env_table(scene.mips)
    env = lambda d: np.asarray(sample_bilinear0(scene.mips, jnp.zeros(len(d), jnp.int32), jnp.asarray(d))) * scene.env_scale
    mis = lambda a, b: a * a / np.maximum(a * a + b * b, 1e-300)

    U, V = (a.ravel() for a in pixel_uv(cam))
    dpix = 2 * cam.tf / cam.width
    P = np.repeat(np.arange(cam.width * cam.height), pc.spp)
    N = len(P)
    du, dv = rng.normal(0, SIGMA_PIX, (2, N))
    f3, r3, u3 = cam.f[:3], cam.r[:3], cam.u[:3]
    d = f3 + (U[P] + du * dpix)[:, None] * r3 + (V[P] + dv * dpix)[:, None] * u3
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.tile(cam.x[:3], (N, 1))
    T, L = np.ones((N, 3)), np.zeros((N, 3))
    alive = np.ones(N, bool)
    prev_pdf = np.zeros(N)
    first = np.ones(N, bool)
    for depth in range(pc.max_depth):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        oo, dd = o[idx], d[idx]
        t, k, n = intersect(sph, pln, oo, dd)
        miss = k < 0
        if miss.any():
            e = np.nonzero(miss)[0]
            w = np.where(first[idx[e]], 1.0, mis(prev_pdf[idx[e]], env_pdf(tab, np.zeros(len(e), int), dd[e])))
            L[idx[e]] += T[idx[e]] * w[:, None] * env(dd[e])
            alive[idx[e]] = False
        h = np.nonzero(~miss)[0]
        if len(h) == 0:
            continue
        ih, kh, p = idx[h], k[h], oo[h] + t[h, None] * dd[h]
        isl = np.isin(kh, lights)
        if len(lights):
            li = np.searchsorted(lights, kh).clip(0, len(lights) - 1)
            pl = 1 / (2 * np.pi * (1 - sphere_cone(oo[h], lc[li], lr[li]))) / len(lights)
            w = np.where(first[ih] | ~isl, 1.0, mis(prev_pdf[ih], pl))
        else:
            w = np.ones(len(h))
        L[ih] += T[ih] * w[:, None] * emit[kh]
        refl = kinds[kh] != 2
        alive[ih[~refl]] = False
        h, ih, kh, p = h[refl], ih[refl], kh[refl], p[refl]
        if len(h) == 0:
            continue
        B = frame(n[h])
        wo = np.einsum("mij,mj->mi", B, -dd[h])
        front = wo[:, 2] > 0
        kd, ad, ra = jnp.asarray(kinds[kh]), jnp.asarray(alb[kh]), jnp.asarray(rough[kh])
        po = p + pc.eps_off * n[h]
        m = np.nonzero(front)[0]
        if len(lights) and len(m):
            li = rng.integers(0, len(lights), len(m))
            ws, ts, pdf_l = sample_sphere_light(po[m], lc[li], lr[li], rng)
            pdf_l = pdf_l / len(lights)
            f, pb = (np.asarray(a) for a in bsdf_eval(kd[m], ad[m], ra[m], jnp.asarray(wo[m]),
                                                     jnp.asarray(np.einsum("mij,mj->mi", B[m], ws))))
            tt, kk, _ = intersect(sph, pln, po[m], ws)
            vis = kk == lights[li]
            L[ih[m]] += np.where(vis[:, None], T[ih[m]] * f * emit[lights[li]] / pdf_l[:, None] * mis(pdf_l, pb)[:, None], 0)
        if len(m):
            ws = env_sample(tab, np.zeros(len(m), int), rng)
            pe = env_pdf(tab, np.zeros(len(m), int), ws)
            f, pb = (np.asarray(a) for a in bsdf_eval(kd[m], ad[m], ra[m], jnp.asarray(wo[m]),
                                                     jnp.asarray(np.einsum("mij,mj->mi", B[m], ws))))
            _, kk, _ = intersect(sph, pln, po[m], ws)
            L[ih[m]] += np.where((kk < 0)[:, None], T[ih[m]] * f * env(ws) / pe[:, None] * mis(pe, pb)[:, None], 0)
        u1, u2 = rng.random((2, len(h)))
        wil, wgt, pb = (np.asarray(a) for a in bsdf_sample(kd, ad, ra, jnp.asarray(wo), jnp.asarray(u1), jnp.asarray(u2)))
        ok = front & (pb > 0)
        T[ih] *= np.where(ok[:, None], wgt, 0)
        o[ih] = po
        d[ih] = np.einsum("mij,mi->mj", B, wil)
        prev_pdf[ih] = pb
        first[ih] = False
        alive[ih[~ok]] = False
        if depth + 1 >= pc.rr_depth:
            q = np.clip(np.max(T[ih], 1), 0.05, 0.95)
            surv = rng.random(len(ih)) < q
            T[ih] /= q[:, None]
            alive[ih[~surv]] = False
    Ls = L.reshape(-1, pc.spp, 3)
    return Ls.mean(1), Ls.var(1, ddof=1) / pc.spp
