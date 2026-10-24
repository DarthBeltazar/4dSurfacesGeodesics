"""Объекты режима B (DD-005): 4D SDF g, пересечённые с M. Предел шага (M26), эрмитово уточнение попадания (M27),
кадр оттенения в T_xM (M28). SDF объектов пишутся через `ops` и работают на массивах и jet/дуальных числах."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .jet import ops
from .integrators import make_integrator, Config, DONE_ESCAPE


@dataclass(frozen=True)
class Material:
    kind: str = "diffuse"                 # diffuse | metal | emit
    albedo: tuple = (0.8, 0.8, 0.8)
    rough: float = 0.3                    # α GGX (metal)
    emit: tuple = (0.0, 0.0, 0.0)         # радиантность излучателя


@dataclass(eq=False)
class Obj:
    name: str
    g: Callable                           # g(p), p = (x, y, z, w); 1-липшицева, ≤ расстояния до {g = 0} (M26)
    mat: Material = field(default_factory=Material)
    center: np.ndarray = None             # ограничивающая 4D-сфера {g ≤ 0} (для аналитического пропуска)
    bound: float = np.inf
    plane: tuple = None                   # (ось, смещение, знак) для полупространства — аналитика на прямых

    def g_vec(self, P):
        return self.g(tuple(P[..., k] for k in range(4)))


def ball4(c, r, mat=Material(), name="ball"):
    """4D-шар |p − c| − r; при c на листе плоской зоны режет лист по 3D-шару радиуса r (M26)."""
    c = [float(a) for a in c]
    return Obj(name, lambda p: ops.length(*(p[i] - c[i] for i in range(4))) - r, mat, np.array(c), float(r))


def torus_in_sheet(c, R, r, mat=Material(), name="torus"):
    """√((√(dx² + dy²) − R)² + dz² + dw²) − r: в листе w = c_w — 3D-тор с осью z."""
    c = [float(a) for a in c]

    def g(p):
        d = [p[i] - c[i] for i in range(4)]
        return ops.length(ops.length(d[0], d[1]) - R, d[2], d[3]) - r
    return Obj(name, g, mat, np.array(c), float(R + r))


def halfspace(axis, offset, mat=Material(), sign=1.0, name="floor"):
    """sign·(p[axis] − offset): не зависит от w, поэтому есть на обоих листах."""
    return Obj(name, lambda p: sign * (p[axis] - offset), mat, None, np.inf, (axis, float(offset), float(sign)))


def g_all(objs):
    """x (4,) -> (k,) значения SDF всех объектов."""
    return lambda x: jnp.stack([jnp.asarray(o.g(tuple(x[i] for i in range(4)))) for o in objs])


# ---- M27: эрмитово уточнение ------------------------------------------------------------------

def hermite(x0, v0, x1, v1, h, s):
    """Кубический эрмитов сплайн по концам шага (dx/ds = v) и его производная по длине дуги."""
    s2, s3 = s * s, s * s * s
    x = (2 * s3 - 3 * s2 + 1) * x0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * x1 + (s3 - s2) * h * v1
    dx = ((6 * s2 - 6 * s) * x0 + (3 * s2 - 4 * s + 1) * h * v0 + (-6 * s2 + 6 * s) * x1 + (3 * s2 - 2 * s) * h * v1) / h
    return x, dx


def make_refiner(f_vec, gk, n_illinois=30, n_newton=2, h_lin=1e-4):
    """M27: корень g(H(σ)) методом Иллинойса, проекция на M (M3), поправки Ньютона вдоль v.
    gk(x, k) — SDF объекта k; h_lin — порог шага, ниже которого v интерполируется линейно. Возвращает refine(x0, v0, x1, v1, h, k) -> (x, v, σ)."""
    gradf = jax.grad(f_vec)
    gradg = jax.grad(gk, argnums=0)

    def proj_M(x):
        for _ in range(2):
            gf = gradf(x)
            x = x - f_vec(x) * gf / (gf @ gf)
        return x

    def refine(x0, v0, x1, v1, h, k):
        G = lambda s: gk(hermite(x0, v0, x1, v1, h, s)[0], k)
        fa, fb = G(0.0), G(1.0)

        def body(_, st):
            a, fa, b, fb, side = st
            c = (a * fb - b * fa) / (fb - fa)
            fc = G(c)
            left = fc * fb < 0          # корень в [c, b]: сдвигаем a; иначе — b
            a2, fa2 = jnp.where(left, b, a), jnp.where(left, fb, fa)
            b2, fb2 = c, fc
            # модификация Иллинойса: если один конец держится два раза подряд — делим его значение пополам
            fa2 = jnp.where(side == jnp.where(left, 1, -1), 0.5 * fa2, fa2)
            return a2, fa2, b2, fb2, jnp.where(left, 1, -1)

        crossed = fb < 0
        a, fa_, b, fb_, _ = lax.fori_loop(0, n_illinois, body, (0.0, fa, 1.0, fb, 0))
        s = jnp.where(crossed, jnp.where(jnp.abs(fa_) < jnp.abs(fb_), a, b), 1.0)
        x, dx = hermite(x0, v0, x1, v1, h, s)
        # при малом h производная эрмита теряет знаки на (x₁ − x₀)/h; линейная интерполяция v там точна до O(h²κ)
        dx = jnp.where(h > h_lin, dx, (1 - s) * v0 + s * v1)
        n = gradf(x)
        n = n / jnp.linalg.norm(n)
        v = dx - (dx @ n) * n
        v = v / jnp.linalg.norm(v)
        for _ in range(n_newton):
            x = proj_M(x)
            gg = gradg(x, k)
            x = x - gk(x, k) / (gg @ v) * v
        x = proj_M(x)
        n = gradf(x)
        n = n / jnp.linalg.norm(n)
        v = v - (v @ n) * n
        return x, v / jnp.linalg.norm(v), s

    return refine


def make_hit_tracer(shape, objs, cfg: Config, escape: Callable = None, eps_hit=1e-9, chunk=None):
    """Лучи до попадания в объект (M26: h ≤ min(hcap, g_min + ε_hit), остановка при g_min < ε_hit) с уточнением M27.

    Возвращает run(X, V) -> dict: hit (bool), obj (индекс, −1 — нет), x, v (точка и направление попадания или
    конечное состояние), s (длина пути), status, nsdf. chunk — фиксированный размер батча (дополнение до него
    избавляет от перекомпиляции при разных N)."""
    G = g_all(objs)
    gmin = lambda x: jnp.min(G(x))
    esc = (lambda x, v: gmin(x) < eps_hit) if escape is None else (lambda x, v: (gmin(x) < eps_hit) | escape(x, v))
    integ = make_integrator(shape, cfg, escape=esc, gcap=lambda x: jnp.maximum(gmin(x), 0.0) + eps_hit, prev=True)
    gk = lambda x, k: G(x)[k]
    refine = jax.vmap(make_refiner(shape.f_vec, gk))

    @jax.jit
    def post(o):
        Gx = jax.vmap(G)(o["x"])
        hit = (o["status"] == DONE_ESCAPE) & (jnp.min(Gx, axis=1) < eps_hit)
        k = jnp.argmin(Gx, axis=1)
        h = jnp.maximum(o["s"] - o["sprev"], 1e-300)
        xr, vr, sig = refine(o["xprev"], o["vprev"], o["x"], o["v"], h, k)
        return dict(hit=hit, obj=jnp.where(hit, k, -1), x=jnp.where(hit[:, None], xr, o["x"]),
                    v=jnp.where(hit[:, None], vr, o["v"]), s=jnp.where(hit, o["sprev"] + sig * h, o["s"]),
                    status=o["status"], nsdf=o["nsdf"])

    def run(X, V):
        X, V = np.asarray(X, float), np.asarray(V, float)
        n = len(X)
        if n == 0:
            return dict(hit=np.zeros(0, bool), obj=np.zeros(0, int), x=X, v=V, s=np.zeros(0), status=np.zeros(0, int),
                        nsdf=np.zeros(0, int))
        c = chunk or n
        outs = []
        for i in range(0, n, c):
            xs, vs = X[i:i + c], V[i:i + c]
            m = len(xs)
            if m < c:      # дополнение копиями первого луча
                xs = np.concatenate([xs, np.repeat(xs[:1], c - m, 0)])
                vs = np.concatenate([vs, np.repeat(vs[:1], c - m, 0)])
            o = post(integ(jnp.asarray(xs), jnp.asarray(vs)))
            outs.append({k: np.asarray(val)[:m] for k, val in o.items()})
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}

    return run


def trace_hits(shape, objs, X, V, cfg: Config, escape: Callable = None, eps_hit=1e-9):
    """Однократный вызов make_hit_tracer (тесты и скрипты)."""
    return make_hit_tracer(shape, objs, cfg, escape, eps_hit)(X, V)


# ---- M28: кадр оттенения ----------------------------------------------------------------------

def cross4(a, b, c):
    """Обобщённое векторное произведение в R⁴: вектор ⊥ a, b, c (кофакторы матрицы 3×4)."""
    M = jnp.stack([a, b, c])
    cols = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    return jnp.stack([(-1) ** i * jnp.linalg.det(M[:, cols[i]]) for i in range(4)])


def shading_frame(f_vec, g_vec, x):
    """n — нормаль M, n_g = P∇g/|P∇g| — нормаль объекта в T_xM (M28), (t₁, t₂) — дополнение до ортонормированного
    базиса T_xM: t₁ — остаток оси с наименьшей проекцией на span(n, n_g) (|остаток|² ≥ 1/2, так как сумма квадратов
    проекций осей на 2D-подпространство равна 2), t₂ = n × n_g × t₁ (R⁴).
    Возвращает (n, n_g, t₁, t₂, |P∇g|)."""
    n = jax.grad(f_vec)(x)
    n = n / jnp.linalg.norm(n)
    gg = jax.grad(g_vec)(x)
    pg = gg - (gg @ n) * n
    png = jnp.linalg.norm(pg)
    ng = pg / png
    e = jnp.eye(4)[jnp.argmin(n * n + ng * ng)]
    t1 = e - (e @ n) * n - (e @ ng) * ng
    t1 = t1 / jnp.linalg.norm(t1)
    t2 = cross4(n, ng, t1)
    return n, ng, t1, t2 / jnp.linalg.norm(t2), png
