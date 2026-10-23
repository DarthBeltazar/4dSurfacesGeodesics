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




def _recip(J):
    a = J.a
    return J._unary(1.0 / a, -1.0 / a ** 2, 2.0 / a ** 3)


# --- засев -------------------------------------------------------------------

def seed7(x, v):
    """M4: координата k ↦ (x_k, v_k, 0, e_k). x, v: (..., 4). Возвращает кортеж из 4 Jet7."""
    eye = jnp.eye(4, dtype=x.dtype)
    z = jnp.zeros_like(x[..., 0])
    return tuple(Jet7(x[..., k], v[..., k], z, jnp.broadcast_to(eye[k], x.shape)) for k in range(4))


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
    def length(*comps, eps=1e-150):
        """M6: защищённая длина √(max(|u|², ε²)); точна при |u| ≥ ε."""
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
