"""Интеграторы лучей (DD-001): вложенный RK с FSAL (BS32, DOPRI5), RK4 и наивная схема с фиксированным шагом.

Один луч — lax.while_loop, батч — vmap. Ведётся счётчик проходов SDF (Jet7) на луч.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .rhs import geodesic_rhs, project

# Статусы завершения луча
RUNNING, DONE_SMAX, DONE_ESCAPE, DONE_NMAX = 0, 1, 2, 3


@dataclass(frozen=True)
class Tableau:
    A: tuple
    b: tuple
    bhat: Optional[tuple]   # вложенная схема низшего порядка (None — без контроля ошибки)
    q: int                  # порядок вложенной схемы: показатель 1/(q+1) в M7
    fsal: bool


# M7
BS32 = Tableau(A=((), (1 / 2,), (0.0, 3 / 4), (2 / 9, 1 / 3, 4 / 9)),
               b=(2 / 9, 1 / 3, 4 / 9, 0.0), bhat=(7 / 24, 1 / 4, 1 / 3, 1 / 8), q=2, fsal=True)

DOPRI5 = Tableau(
    A=((),
       (1 / 5,),
       (3 / 40, 9 / 40),
       (44 / 45, -56 / 15, 32 / 9),
       (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
       (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
       (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)),
    b=(35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0),
    bhat=(5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40),
    q=4, fsal=True)

RK4 = Tableau(A=((), (1 / 2,), (0.0, 1 / 2), (0.0, 0.0, 1.0)), b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
              bhat=None, q=0, fsal=False)

TABLEAUS = dict(bs32=BS32, dopri5=DOPRI5, rk4=RK4)


@dataclass(frozen=True)
class Config:
    method: str = "bs32"        # bs32 | dopri5 | rk4 | naive
    tol: float = 1e-8           # допуск M8 (адаптивные методы)
    h: float = 0.05             # начальный шаг (адаптивные) или фиксированный шаг
    h_max: Optional[float] = None  # None — shape.hcap(x) или ℓ_s (иначе шаг «туннелирует» сквозь кривизну)
    h_min: float = 1e-12
    s_max: float = 10.0         # потолок длины пути
    n_max: int = 100_000        # потолок числа попыток шага
    fsal: str = "stale"         # stale | reeval (DD-001)
    xscale: Optional[float] = None  # ℓ_s в M8; None — масштаб фигуры
    alpha: Optional[float] = None   # α в M2; None — по фигуре (ADR-0001)
    record: int = 0             # >0: сохранять первые `record` принятых состояний
    fixed: bool = False         # вложенная схема с фиксированным шагом h (для графика сходимости)

    def with_(self, **kw):
        return replace(self, **kw)


def make_integrator(shape, cfg: Config, escape: Callable = None, aux: Callable = None, naux: int = 0,
                    gcap: Callable = None, f: Callable = None):
    """Возвращает jit-функцию run(x0, v0) для батча (N, 4) лучей.

    escape(x, v) -> bool    — луч покинул область интегрирования (проверяется после принятого шага);
    aux(x_old, x_new) -> (naux,) — накапливаемая величина вдоль пути (например, развёрнутый угол);
    gcap(x) -> float        — ограничение шага SDF объектов (sphere tracing);
    f — переопределение f фигуры (для обёрток U5/U6).
    """
    f = f or shape.f
    alpha = shape.alpha if cfg.alpha is None else cfg.alpha
    xs = shape.scale if cfg.xscale is None else cfg.xscale
    if cfg.h_max is not None:
        hcap = lambda x: cfg.h_max
    elif shape.hcap is not None:
        hcap = shape.hcap
    else:
        hcap = lambda x: shape.scale
    R = geodesic_rhs(f, alpha)
    naive = cfg.method == "naive"
    T = None if naive else TABLEAUS[cfg.method]
    adaptive = (not naive) and T.bhat is not None and not cfg.fixed
    S = 0 if naive else len(T.A)
    extra_eval = 0 if naive or T.fsal else 1
    reeval = cfg.fsal == "reeval" and not naive

    def tangential(e, n):
        return e - jnp.sum(e * n) * n

    def step(x, v, acc, h):
        """Одна попытка шага. Возвращает (x⁺, v⁺, acc⁺, err, вызовы SDF)."""
        if naive:
            x1 = x + h * v
            _, fv, g = R(x1, v)
            xp, vp = project(x1, v, fv, g)
            return xp, vp, acc, jnp.asarray(0.0), 1          # ускорение схеме не нужно
        kx, kv = [v], [acc]
        fv = g = None
        for i in range(1, S):
            xi = x + h * sum(a * k for a, k in zip(T.A[i], kx) if a != 0.0)
            vi = v + h * sum(a * k for a, k in zip(T.A[i], kv) if a != 0.0)
            ai, fv, g = R(xi, vi)
            kx.append(vi)
            kv.append(ai)
        if T.fsal:
            xn, vn, an = xi, vi, kv[-1]                      # последняя стадия — в y⁺
        else:
            xn = x + h * sum(bj * k for bj, k in zip(T.b, kx))
            vn = v + h * sum(bj * k for bj, k in zip(T.b, kv))
            an, fv, g = R(xn, vn)
        if adaptive:
            db = [bj - bh for bj, bh in zip(T.b, T.bhat)]
            ex = h * sum(d * k for d, k in zip(db, kx) if d != 0.0)
            ev = h * sum(d * k for d, k in zip(db, kv) if d != 0.0)
            n = g / jnp.linalg.norm(g)
            err = jnp.maximum(jnp.linalg.norm(tangential(ex, n)) / xs,
                              jnp.linalg.norm(tangential(ev, n)))       # M8
        else:
            err = jnp.asarray(0.0)
        xp, vp = project(xn, vn, fv, g)                                  # M3
        if reeval:
            ap, _, _ = R(xp, vp)
        else:
            ap = an                                                      # stale FSAL (DD-001)
        return xp, vp, ap, err, (S - 1) + extra_eval + (1 if reeval else 0)

    def run_one(x0, v0):
        acc0, _, _ = R(x0, v0)
        h0 = jnp.minimum(jnp.minimum(cfg.h, hcap(x0)), cfg.s_max)
        if gcap is not None:
            h0 = jnp.minimum(h0, gcap(x0))
        rec = jnp.zeros((max(cfg.record, 1), 9)).at[0].set(jnp.concatenate([x0, v0, jnp.zeros(1)]))
        st = dict(x=x0, v=v0, acc=acc0, h=h0, s=jnp.asarray(0.0), nacc=jnp.asarray(0), nrej=jnp.asarray(0),
                  nsdf=jnp.asarray(1), status=jnp.asarray(RUNNING), aux=jnp.zeros(naux), rec=rec)

        def cond(st):
            return st["status"] == RUNNING

        def body(st):
            x, v, h = st["x"], st["v"], st["h"]
            xp, vp, ap, err, ncalls = step(x, v, st["acc"], h)
            accept = (err <= cfg.tol) | (h <= cfg.h_min) if adaptive else jnp.asarray(True)
            s_new = jnp.where(accept, st["s"] + h, st["s"])
            x_new = jnp.where(accept, xp, x)
            v_new = jnp.where(accept, vp, v)
            a_new = jnp.where(accept, ap, st["acc"])
            if adaptive:
                fac = jnp.clip(0.9 * (cfg.tol / jnp.maximum(err, 1e-300)) ** (1.0 / (T.q + 1)), 0.2, 5.0)  # M7
                h_new = h * fac
            else:
                h_new = jnp.asarray(cfg.h)
            h_new = jnp.minimum(h_new, hcap(x_new))
            if gcap is not None:
                h_new = jnp.minimum(h_new, gcap(x_new))
            remaining = cfg.s_max - s_new
            h_new = jnp.maximum(jnp.minimum(h_new, remaining), 0.0)
            nacc = st["nacc"] + accept.astype(int)
            nrej = st["nrej"] + (~accept).astype(int)
            status = jnp.where(remaining <= 1e-13 * cfg.s_max, DONE_SMAX, RUNNING)
            if escape is not None:
                status = jnp.where(accept & escape(x_new, v_new), DONE_ESCAPE, status)
            status = jnp.where((status == RUNNING) & (nacc + nrej >= cfg.n_max), DONE_NMAX, status)
            new_aux = st["aux"] + (jnp.where(accept, aux(x, x_new), 0.0) if aux is not None else 0.0)
            rec = st["rec"]
            if cfg.record:
                row = jnp.concatenate([x_new, v_new, s_new[None]])
                idx = jnp.where(accept & (nacc < cfg.record), nacc, cfg.record)  # вне диапазона — отбрасывается
                rec = rec.at[idx].set(row, mode="drop")
            return dict(x=x_new, v=v_new, acc=a_new, h=h_new, s=s_new, nacc=nacc, nrej=nrej,
                        nsdf=st["nsdf"] + ncalls, status=status, aux=new_aux, rec=rec)

        out = lax.while_loop(cond, body, st)
        out.pop("acc")
        if not cfg.record:
            out.pop("rec")
        return out

    return jax.jit(jax.vmap(run_one))
