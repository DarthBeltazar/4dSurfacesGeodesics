"""Jet-числа (M4, M5) и пространство имён ops (M6), см. docs/math.md.

SDF пишутся один раз как f(p), p = (x, y, z, w), через арифметику и функции из `ops`.
Один и тот же код работает на jnp-массивах (эталон через jax.grad/jax.hessian), Jet7 и Jet15.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util


class _JetBase:
    __array_priority__ = 1000

    # --- арифметика, общая для обоих типов -----------------------------------
    def __add__(self, o):
        if isinstance(o, _JetBase):
            return type(self)(*(p + q for p, q in zip(self._fields(), o._fields())))
        return type(self)(self.a + o, *self._fields()[1:])

    __radd__ = __add__

    def __neg__(self):
        return type(self)(*(-p for p in self._fields()))

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, _JetBase):
            return self._mul(o)
        return type(self)(*(p * _bcast(o, p) for p in self._fields()))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, _JetBase):
            return self * _recip(o)
        return self * (1.0 / o)

    def __rtruediv__(self, o):
        return _recip(self) * o

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise TypeError("Jet ** поддерживает только неотрицательные целые степени")
        if n == 0:
            return self * 0.0 + 1.0
        r = self
        for _ in range(n - 1):
            r = r * self
        return r


def _bcast(k, p):
    """Скаляр/массив k формы батча, домноженный на поле p (у p могут быть хвостовые оси)."""
    k = jnp.asarray(k)
    return jnp.reshape(k, k.shape + (1,) * (jnp.ndim(p) - jnp.ndim(k))) if jnp.ndim(k) else k


@tree_util.register_pytree_node_class
class Jet7(_JetBase):
    """M4: J = a + b·t + c·t² + Σ dᵢsᵢ  (t³ = 0, t·sᵢ = 0, sᵢsⱼ = 0)."""

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d

    def _fields(self):
        return (self.a, self.b, self.c, self.d)

    def _mul(self, o):
        a1, b1, c1, d1 = self._fields()
        a2, b2, c2, d2 = o._fields()
        return Jet7(a1 * a2, a1 * b2 + b1 * a2, a1 * c2 + b1 * b2 + c1 * a2,
                    a1[..., None] * d2 + d1 * a2[..., None])

    def _unary(self, g0, g1, g2):
        return Jet7(g0, g1 * self.b, g1 * self.c + 0.5 * g2 * self.b ** 2, g1[..., None] * self.d)

    def tree_flatten(self):
        return self._fields(), None

    @classmethod
    def tree_unflatten(cls, _, ch):
        return cls(*ch)




@tree_util.register_pytree_node_class
class Jet15(_JetBase):
    """M5: J = a + gᵀε + ½ εᵀHε, ε ∈ R⁴ (H симметрична: 15 независимых компонент)."""

    def __init__(self, a, g, H):
        self.a, self.g, self.H = a, g, H

    def _fields(self):
        return (self.a, self.g, self.H)

    def _mul(self, o):
        a1, g1, H1 = self._fields()
        a2, g2, H2 = o._fields()
        gg = g1[..., :, None] * g2[..., None, :]
        return Jet15(a1 * a2, a1[..., None] * g2 + a2[..., None] * g1,
                     a1[..., None, None] * H2 + a2[..., None, None] * H1 + gg + jnp.swapaxes(gg, -1, -2))

    def _unary(self, f0, f1, f2):
        g = self.g
        return Jet15(f0, f1[..., None] * g,
                     f1[..., None, None] * self.H + f2[..., None, None] * (g[..., :, None] * g[..., None, :]))

    def tree_flatten(self):
        return self._fields(), None

    @classmethod
    def tree_unflatten(cls, _, ch):
        return cls(*ch)




@tree_util.register_pytree_node_class
class JetHVP(_JetBase):
    """M23: J = a + b·t + Σ dᵢsᵢ + Σ mᵢ t sᵢ  (t² = 0, sᵢsⱼ = 0). Засев (x_k, w_k, e_k, 0) даёт
    f = a, ∇f·w = b, ∇f = d, H·w = m за один проход."""

    def __init__(self, a, b, d, m):
        self.a, self.b, self.d, self.m = a, b, d, m

    def _fields(self):
        return (self.a, self.b, self.d, self.m)

    def _mul(self, o):
        a1, b1, d1, m1 = self._fields()
        a2, b2, d2, m2 = o._fields()
        A1, A2, B1, B2 = a1[..., None], a2[..., None], b1[..., None], b2[..., None]
        return JetHVP(a1 * a2, a1 * b2 + b1 * a2, A1 * d2 + d1 * A2, A1 * m2 + m1 * A2 + B1 * d2 + d1 * B2)

    def _unary(self, g0, g1, g2):
        G1, G2 = g1[..., None], g2[..., None]
        return JetHVP(g0, g1 * self.b, G1 * self.d, G1 * self.m + G2 * self.b[..., None] * self.d)

    def tree_flatten(self):
        return self._fields(), None

    @classmethod
    def tree_unflatten(cls, _, ch):
        return cls(*ch)


@tree_util.register_pytree_node_class
class Dual(_JetBase):
    """M24: дуальные числа первого порядка по k направлениям: a + Σ bᵢεᵢ (εᵢεⱼ = 0). Прямое AD «руками»
    для производной направления выхода (то же, что будет в Slang)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def _fields(self):
        return (self.a, self.b)

    def _mul(self, o):
        return Dual(self.a * o.a, self.a[..., None] * o.b + self.b * o.a[..., None])

    def _unary(self, g0, g1, g2):
        return Dual(g0, g1[..., None] * self.b)

    def tree_flatten(self):
        return self._fields(), None

    @classmethod
    def tree_unflatten(cls, _, ch):
        return cls(*ch)


def _recip(J):
    a = J.a
    return J._unary(1.0 / a, -1.0 / a ** 2, 2.0 / a ** 3)


def _sqrt_family(y, thr=1e-3):
    """(cos√y, s = sin√y/√y, s′, s″) — замкнутые формы при y ≥ thr, ряды при y < thr (одна функция)."""
    small = y < thr
    ys = jnp.where(small, 1.0, y)
    r = jnp.sqrt(ys)
    c, s = jnp.cos(r), jnp.sin(r) / r
    s1 = (c - s) / (2 * ys)
    s2 = (-0.5 * s - s1) / (2 * ys) - s1 / ys
    c_ser = 1 - y / 2 + y * y / 24 - y ** 3 / 720
    s_ser = 1 - y / 6 + y * y / 120 - y ** 3 / 5040
    s1_ser = -1 / 6 + y / 60 - y * y / 1680
    s2_ser = 1 / 60 - y / 840 + y * y / 30240
    return (jnp.where(small, c_ser, c), jnp.where(small, s_ser, s), jnp.where(small, s1_ser, s1),
            jnp.where(small, s2_ser, s2))


# --- засев -------------------------------------------------------------------

def seed7(x, v):
    """M4: координата k ↦ (x_k, v_k, 0, e_k). x, v: (..., 4). Возвращает кортеж из 4 Jet7."""
    eye = jnp.eye(4, dtype=x.dtype)
    z = jnp.zeros_like(x[..., 0])
    return tuple(Jet7(x[..., k], v[..., k], z, jnp.broadcast_to(eye[k], x.shape)) for k in range(4))


def seed_hvp(x, w):
    """M23: координата k ↦ (x_k, w_k, e_k, 0). x, w: (..., 4)."""
    eye = jnp.eye(4, dtype=x.dtype)
    Z = jnp.zeros_like(x)
    return tuple(JetHVP(x[..., k], w[..., k], jnp.broadcast_to(eye[k], x.shape), Z) for k in range(4))


def dual(a, b):
    """M24: компонента a (...) с производными b (..., k)."""
    return Dual(jnp.asarray(a), jnp.asarray(b))


def seed15(x):
    """M5: координата k ↦ (x_k, e_k, 0)."""
    eye = jnp.eye(4, dtype=x.dtype)
    Z = jnp.zeros(x.shape + (4,), x.dtype)
    return tuple(Jet15(x[..., k], jnp.broadcast_to(eye[k], x.shape), Z) for k in range(4))


# --- ops: функции, диспетчеризуемые по типу -----------------------------------

class ops:
    @staticmethod
    def value(u):
        return u.a if isinstance(u, _JetBase) else u

    @staticmethod
    def _apply(u, f0, f1, f2):
        if isinstance(u, _JetBase):
            a = u.a
            return u._unary(f0(a), f1(a), f2(a))
        return f0(u)

    @staticmethod
    def sqrt(u):
        return ops._apply(u, jnp.sqrt, lambda a: 0.5 / jnp.sqrt(a), lambda a: -0.25 / (a * jnp.sqrt(a)))

    @staticmethod
    def exp(u):
        return ops._apply(u, jnp.exp, jnp.exp, jnp.exp)

    @staticmethod
    def log(u):
        return ops._apply(u, jnp.log, lambda a: 1.0 / a, lambda a: -1.0 / a ** 2)

    @staticmethod
    def cosh(u):
        return ops._apply(u, jnp.cosh, jnp.sinh, jnp.cosh)

    @staticmethod
    def sinh(u):
        return ops._apply(u, jnp.sinh, jnp.cosh, jnp.sinh)

    @staticmethod
    def cos(u):
        return ops._apply(u, jnp.cos, lambda a: -jnp.sin(a), lambda a: -jnp.cos(a))

    @staticmethod
    def sin(u):
        return ops._apply(u, jnp.sin, jnp.cos, lambda a: -jnp.sin(a))

    @staticmethod
    def sinc(u):
        """sin(u)/u, гладко при u → 0 (ряд при |u| < 1e-3: обе ветви — одна функция)."""
        def f0(a):
            sa = jnp.where(jnp.abs(a) < 1e-3, 1.0, a)
            return jnp.where(jnp.abs(a) < 1e-3, 1 - a * a / 6 + a ** 4 / 120, jnp.sin(sa) / sa)

        def f1(a):
            sa = jnp.where(jnp.abs(a) < 1e-3, 1.0, a)
            return jnp.where(jnp.abs(a) < 1e-3, -a / 3 + a ** 3 / 30, (sa * jnp.cos(sa) - jnp.sin(sa)) / sa ** 2)

        def f2(a):
            sa = jnp.where(jnp.abs(a) < 1e-3, 1.0, a)
            return jnp.where(jnp.abs(a) < 1e-3, -1 / 3 + a * a / 10,
                             ((2 - sa * sa) * jnp.sin(sa) - 2 * sa * jnp.cos(sa)) / sa ** 3)
        return ops._apply(u, f0, f1, f2)

    @staticmethod
    def cos_sqrt(u):
        """cos(√y) — гладкая функция y ≥ 0 (M24). Производные: c′ = −s/2, c″ = −s′/2, s = sinc(√y)."""
        f = _sqrt_family
        return ops._apply(u, lambda a: f(a)[0], lambda a: -0.5 * f(a)[1], lambda a: -0.5 * f(a)[2])

    @staticmethod
    def sinc_sqrt(u):
        """sin(√y)/√y — гладкая функция y ≥ 0 (M24)."""
        f = _sqrt_family
        return ops._apply(u, lambda a: f(a)[1], lambda a: f(a)[2], lambda a: f(a)[3])

    @staticmethod
    def where(cond, x, y):
        """Выбор ветви по значению. Обе ветви должны быть записью одной гладкой функции (M6)."""
        if not isinstance(x, _JetBase) and not isinstance(y, _JetBase):
            return jnp.where(cond, x, y)
        T = type(x) if isinstance(x, _JetBase) else type(y)
        ref = x if isinstance(x, _JetBase) else y
        X = x if isinstance(x, _JetBase) else ref * 0.0 + x
        Y = y if isinstance(y, _JetBase) else ref * 0.0 + y
        return T(*(jnp.where(_bcast(cond, p), p, q) for p, q in zip(X._fields(), Y._fields())))

    @staticmethod
    def length(*comps, eps=None):
        """M6: защищённая длина √(max(|u|², ε²)); точна при |u| ≥ ε. ε = 1e-150 (float64) / 1e-18 (float32)."""
        if eps is None:
            eps = 1e-150 if jax.config.read("jax_enable_x64") else 1e-18
        q = comps[0] * comps[0]
        for c in comps[1:]:
            q = q + c * c
        small = ops.value(q) < eps * eps
        q_safe = ops.where(small, eps * eps, q)
        return ops.sqrt(q_safe)

    @staticmethod
    def smin(a, b, k):
        """M6: C∞ гладкий минимум −k ln(e^{−a/k} + e^{−b/k}), устойчивая форма."""
        va, vb = ops.value(a), ops.value(b)
        lo = ops.where(va < vb, a, b)
        hi = ops.where(va < vb, b, a)
        return lo - k * ops.log(1.0 + ops.exp((lo - hi) / k))
