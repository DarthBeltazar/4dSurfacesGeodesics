"""Библиотека фигур и червоточин (DD-002). Все f написаны через `ops` и работают на массивах и jet-числах."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .jet import ops


@dataclass(eq=False)
class Shape:
    name: str
    f: Callable                       # f(p), p = (x, y, z, w)
    scale: float                      # ℓ_s: масштаб геометрии (M8, α)
    sample_raw: Callable              # rng, n -> (n, 4) точки около M (уточняются Ньютоном)
    planes: tuple = ()                # плоскости (i, j) с сохраняющимся Lᵢⱼ (M13)
    exact: Optional[Callable] = None  # (x0, v0, s) -> (x, v), s: (...,), M13
    invariants: Optional[Callable] = None  # (x, v) -> массив дополнительных инвариантов
    params: dict = field(default_factory=dict)
    hcap: Optional[Callable] = None   # x -> предельный шаг (иначе ℓ_s); защита от «туннелирования», DD-001

    @property
    def alpha(self):
        # ADR-0001: α = 1/ℓ_s
        return 1.0 / self.scale

    def f_vec(self, P):
        """f на массиве (..., 4)."""
        return self.f(tuple(P[..., k] for k in range(4)))


def _unit(rng, n, d):
    u = rng.normal(size=(n, d))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def project_points(shape: Shape, P, iters=30):
    """Ньютон вдоль ∇f до f = 0 (эталонные производные через jax.grad)."""
    g = jax.vmap(jax.grad(shape.f_vec))
    fv = jax.vmap(shape.f_vec)
    P = jnp.asarray(P)
    for _ in range(iters):
        G = g(P)
        P = P - (fv(P) / jnp.sum(G * G, -1))[:, None] * G
    return P


def random_tangent(shape: Shape, P, rng):
    G = np.asarray(jax.vmap(jax.grad(shape.f_vec))(jnp.asarray(P)))
    n = G / np.linalg.norm(G, axis=1, keepdims=True)
    v = rng.normal(size=P.shape)
    v -= np.sum(v * n, 1, keepdims=True) * n
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sample(shape: Shape, n, seed=0):
    """n точек на M и единичных касательных векторов."""
    rng = np.random.default_rng(seed)
    P = np.asarray(project_points(shape, shape.sample_raw(rng, n)))
    return P, random_tangent(shape, P, rng)


def moments(x, v, planes):
    """Lᵢⱼ = xᵢvⱼ − xⱼvᵢ для списка плоскостей."""
    return jnp.stack([x[..., i] * v[..., j] - x[..., j] * v[..., i] for i, j in planes], -1)


# ----------------------------------------------------------------------------------------------
# Тестовые фигуры (TASK.md, таблица; точные решения M13)

def hyperplane():
    def exact(x0, v0, s):
        s = jnp.asarray(s)[..., None]
        return x0 + s * v0, v0

    return Shape("hyperplane", lambda p: p[3], 1.0,
                 lambda rng, n: np.c_[rng.uniform(-2, 2, (n, 3)), np.zeros(n)],
                 planes=((0, 1), (1, 2), (0, 2)), exact=exact)


def sphere3(R=1.0):
    def exact(x0, v0, s):
        s = jnp.asarray(s)[..., None]
        c, si = jnp.cos(s / R), jnp.sin(s / R)
        return c * x0 + R * si * v0, -si / R * x0 + c * v0

    return Shape("S3", lambda p: ops.length(*p) - R, R, lambda rng, n: R * _unit(rng, n, 4),
                 planes=tuple((i, j) for i in range(4) for j in range(i + 1, 4)), exact=exact,
                 params=dict(R=R))


def sphere2_x_r(R=1.0):
    def exact(x0, v0, s):
        s = jnp.asarray(s)[..., None]
        q0, u = x0[..., :3], v0[..., :3]
        c = jnp.linalg.norm(u, axis=-1, keepdims=True)
        om = c / R
        q = jnp.cos(om * s) * q0 + jnp.sin(om * s) / om * u
        dq = -om * jnp.sin(om * s) * q0 + jnp.cos(om * s) * u
        return (jnp.concatenate([q, x0[..., 3:] + s * v0[..., 3:]], -1),
                jnp.concatenate([dq, v0[..., 3:]], -1))

    return Shape("S2xR", lambda p: ops.length(p[0], p[1], p[2]) - R, R,
                 lambda rng, n: np.c_[R * _unit(rng, n, 3), rng.uniform(-1, 1, n)],
                 planes=((0, 1), (1, 2), (0, 2)), exact=exact, params=dict(R=R))


def sphere1_x_r2(R=1.0):
    def exact(x0, v0, s):
        th0 = jnp.arctan2(x0[..., 1], x0[..., 0])
        vth = (x0[..., 0] * v0[..., 1] - x0[..., 1] * v0[..., 0]) / R  # R·dθ/ds
        th = th0 + vth / R * s
        s = s[..., None]
        er = jnp.stack([jnp.cos(th), jnp.sin(th)], -1)
        et = jnp.stack([-jnp.sin(th), jnp.cos(th)], -1)
        x = jnp.concatenate([R * er, x0[..., 2:] + s * v0[..., 2:]], -1)
        v = jnp.concatenate([vth[..., None] * et, v0[..., 2:]], -1)
        return x, v

    def raw(rng, n):
        th = rng.uniform(0, 2 * np.pi, n)
        return np.c_[R * np.cos(th), R * np.sin(th), rng.uniform(-1, 1, (n, 2))]

    return Shape("S1xR2", lambda p: ops.length(p[0], p[1]) - R, R, raw, planes=((0, 1),),
                 exact=exact, params=dict(R=R))


def ellipsoid(a=(0.8, 1.0, 1.3, 1.6)):
    a = tuple(float(t) for t in a)
    A = np.array(a)

    def joachimsthal(x, v):
        return (jnp.sum(x ** 2 / A ** 4, -1) * jnp.sum(v ** 2 / A ** 2, -1))[..., None]

    return Shape("ellipsoid", lambda p: ops.length(*(p[k] * (1.0 / a[k]) for k in range(4))) - 1.0, 1.0,
                 lambda rng, n: _unit(rng, n, 4) * A, invariants=joachimsthal, params=dict(a=a))


def spheritorus(R=1.0, r=0.4):
    def raw(rng, n):
        th = rng.uniform(0, 2 * np.pi, n)
        return np.c_[(R + r * np.cos(th))[:, None] * _unit(rng, n, 3), r * np.sin(th)]

    return Shape("spheritorus", lambda p: ops.length(ops.length(p[0], p[1], p[2]) - R, p[3]) - r, r, raw,
                 planes=((0, 1), (1, 2), (0, 2)), params=dict(R=R, r=r))


def torisphere(R=1.0, r=0.4):
    def raw(rng, n):
        u = _unit(rng, n, 3)
        ph = rng.uniform(0, 2 * np.pi, n)
        rad = R + r * u[:, 0]
        return np.c_[rad * np.cos(ph), rad * np.sin(ph), r * u[:, 1], r * u[:, 2]]

    return Shape("torisphere", lambda p: ops.length(ops.length(p[0], p[1]) - R, p[2], p[3]) - r, r, raw,
                 planes=((0, 1), (2, 3)), params=dict(R=R, r=r))


def tiger(R1=1.2, R2=0.8, r=0.3):
    def raw(rng, n):
        th, p1, p2 = (rng.uniform(0, 2 * np.pi, n) for _ in range(3))
        A, B = R1 + r * np.cos(th), R2 + r * np.sin(th)
        return np.c_[A * np.cos(p1), A * np.sin(p1), B * np.cos(p2), B * np.sin(p2)]

    return Shape("tiger", lambda p: ops.length(ops.length(p[0], p[1]) - R1, ops.length(p[2], p[3]) - R2) - r,
                 r, raw, planes=((0, 1), (2, 3)), params=dict(R1=R1, R2=R2, r=r))


# ----------------------------------------------------------------------------------------------
# Червоточины

def ellis(b=1.0):
    """M9: f = ρ − b cosh(w/b)."""
    def raw(rng, n):
        ell = rng.uniform(-4 * b, 4 * b, n)
        return np.c_[np.sqrt(ell ** 2 + b ** 2)[:, None] * _unit(rng, n, 3), b * np.arcsinh(ell / b)]

    # Радиус кривизны растёт как r²/b: шаг ограничен долей r = ρ.
    return Shape("ellis", lambda p: ops.length(p[0], p[1], p[2]) - b * ops.cosh(p[3] * (1.0 / b)), b, raw,
                 planes=((0, 1), (1, 2), (0, 2)), params=dict(b=b),
                 hcap=lambda x: 0.5 * jnp.linalg.norm(x[:3]))


def compact_psi(u):
    """M11: ψ(u) = 1 − exp(1 − 1/(1 − u)) при u < 1, иначе 1 (C∞). Ветвь — по значению (DD-002)."""
    q = 1.0 - u
    inside = ops.value(q) > 1e-2      # при q ≤ 1e-2 отброшенный хвост ≤ e^{-99}
    q_safe = ops.where(inside, q, 1.0)
    return ops.where(inside, 1.0 - ops.exp(1.0 - 1.0 / q_safe), 1.0)


def compact_h(rho, b, rho0, W):
    return (W * W) * compact_psi((rho - b) * (1.0 / (rho0 - b)))


def compact_wormhole(b=1.0, rho0=3.0, W=1.5):
    """M11: f = w² − h(ρ). Вне ρ ≥ ρ₀ — гиперплоскости w = ±W."""
    def f(p):
        return p[3] * p[3] - compact_h(ops.length(p[0], p[1], p[2]), b, rho0, W)

    def raw(rng, n):
        rho = rng.uniform(b, rho0 + 1.0, n)
        h = np.asarray(compact_h(jnp.asarray(rho), b, rho0, W))
        w = np.sqrt(np.maximum(h, 0.0)) * rng.choice([-1.0, 1.0], n)
        return np.c_[rho[:, None] * _unit(rng, n, 3), w]

    # Вне шара ρ₀ оценка ошибки ≡ 0: шаг ограничен расстоянием до шара (sphere tracing), внутри — b/4.
    return Shape("compact_wh", f, b, raw, planes=((0, 1), (1, 2), (0, 2)),
                 params=dict(b=b, rho0=rho0, W=W),
                 hcap=lambda x: jnp.maximum(jnp.linalg.norm(x[:3]) - rho0, 0.25 * b))


def two_compact_wormholes(c1=(-4.0, 0.0, 0.0), c2=(4.0, 0.0, 0.0), b=1.0, rho0=3.0, W=1.5):
    """M12: две компактные червоточины с непересекающимися шарами."""
    assert np.linalg.norm(np.subtract(c1, c2)) > 2 * rho0

    def f(p):
        r1 = ops.length(p[0] - c1[0], p[1] - c1[1], p[2] - c1[2])
        r2 = ops.length(p[0] - c2[0], p[1] - c2[1], p[2] - c2[2])
        return p[3] * p[3] - (compact_h(r1, b, rho0, W) + compact_h(r2, b, rho0, W) - W * W)

    def raw(rng, n):
        P = compact_wormhole(b, rho0, W).sample_raw(rng, n)
        P[:, :3] += np.where(rng.random(n) < 0.5, 1.0, 0.0)[:, None] * np.subtract(c1, c2) + np.asarray(c2)
        return P

    def hcap(x):
        d1 = jnp.linalg.norm(x[:3] - jnp.asarray(c1)) - rho0
        d2 = jnp.linalg.norm(x[:3] - jnp.asarray(c2)) - rho0
        return jnp.maximum(jnp.minimum(d1, d2), 0.25 * b)

    return Shape("two_compact_wh", f, b, raw, params=dict(c1=c1, c2=c2, b=b, rho0=rho0, W=W), hcap=hcap)


# ----------------------------------------------------------------------------------------------

def all_shapes():
    return [hyperplane(), sphere3(), sphere2_x_r(), sphere1_x_r2(), ellipsoid(), spheritorus(),
            torisphere(), tiger(), ellis(), compact_wormhole()]


SHAPES = {s.name: s for s in all_shapes()}


# ----------------------------------------------------------------------------------------------
# Обёртки для U5 и U6

def reparam(shape: Shape, kind: str) -> Shape:
    """U5: та же M, другая f: φ(f) с φ(0) = 0, φ′ > 0, или f, умноженная на положительную функцию."""
    f0 = shape.f
    fs = {
        "cubic": lambda p: f0(p) + f0(p) * f0(p) * f0(p),
        "fexp": lambda p: f0(p) * ops.exp(f0(p)),
        "mult": lambda p: f0(p) * (1.0 + 0.5 * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3])),
    }
    return Shape(f"{shape.name}/{kind}", fs[kind], shape.scale, shape.sample_raw, shape.planes,
                 shape.exact, shape.invariants, shape.params, shape.hcap)


def random_rotation(seed):
    """Случайное вращение из SO(4) (QR гауссовой матрицы с исправлением знаков)."""
    rng = np.random.default_rng(seed)
    Q, Rm = np.linalg.qr(rng.normal(size=(4, 4)))
    Q = Q * np.sign(np.diag(Rm))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def transformed(shape: Shape, Q, t) -> Shape:
    """U6: f′(p) = f(Qᵀ(p − t)), M′ = Q M + t."""
    Q, t = np.asarray(Q, float), np.asarray(t, float)

    def back(p):
        d = [p[j] - t[j] for j in range(4)]
        return tuple(sum(float(Q[j, k]) * d[j] for j in range(4)) for k in range(4))

    def raw(rng, n):
        return shape.sample_raw(rng, n) @ Q.T + t

    hcap = None if shape.hcap is None else (lambda x: shape.hcap(jnp.asarray(Q.T) @ (x - t)))
    return Shape(f"{shape.name}/moved", lambda p: shape.f(back(p)), shape.scale, raw, (), None, None,
                 shape.params, hcap)
