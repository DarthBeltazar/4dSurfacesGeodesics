"""Поля Якоби (M14, M15): правая часть луча + 4 полей за один проход Jet15, секционная кривизна."""
import jax
import jax.numpy as jnp

from .jet import seed15, seed_hvp


def shape_operator(g, H):
    """M14: S = P H P / |∇f|, n = ∇f/|∇f|."""
    gn = jnp.linalg.norm(g)
    n = g / gn
    P = jnp.eye(4, dtype=g.dtype) - jnp.outer(n, n)
    return P @ H @ P / gn, n


def jacobi_rhs(f, alpha):
    """R(x, v, E) -> (v̇, Ė, f, ∇f). E: (2, k, 4) — E[0] = J, E[1] = J′ = DJ/dt для k полей. Один проход Jet15."""
    def R(x, v, E):
        J = f(seed15(x))
        g, H, fa = J.g, J.H, J.a
        lam = -(v @ H @ v + 2.0 * alpha * (g @ v) + alpha * alpha * fa) / (g @ g)   # M2 (vᵀHv из H)
        S, n = shape_operator(g, H)
        Sv = S @ v
        vSv = v @ Sv
        Jm, Jp = E[0], E[1]
        SJ = Jm @ S                                                                 # S симметрична
        dJ = Jp - (Jm @ Sv)[:, None] * n                                            # M15
        dJp = -vSv * SJ + (SJ @ v)[:, None] * Sv - (Jp @ Sv)[:, None] * n
        return lam * g, jnp.stack([dJ, dJp]), fa, g
    return R


def project_fields(E, g):
    """M15: J ← PJ, J′ ← PJ′."""
    n = g / jnp.linalg.norm(g)
    return E - (E @ n)[..., None] * n


def sectional_curvature(S, X, Y):
    """M15: K(X, Y) = (⟨SX,X⟩⟨SY,Y⟩ − ⟨SX,Y⟩²) / (|X|²|Y|² − ⟨X,Y⟩²)."""
    return ((X @ S @ X) * (Y @ S @ Y) - (X @ S @ Y) ** 2) / ((X @ X) * (Y @ Y) - (X @ Y) ** 2)


def jacobi_rhs_hvp(f, alpha):
    """Как jacobi_rhs, но на HVP-jet (M23): проход с w = v даёт f, ∇f, H·v; по проходу на поле — H·(PJ_k).
    Sv := P(H v)/|∇f| (без правой проекции v: на M совпадает с M14; на стадиях отличается на O(n·v)).
    Проходов на вычисление: 1 + k."""
    def R(x, v, E):
        J0 = f(seed_hvp(x, v))
        g, fa, Hv = J0.d, J0.a, J0.m
        gn = jnp.linalg.norm(g)
        n = g / gn
        lam = -(v @ Hv + 2.0 * alpha * (g @ v) + alpha * alpha * fa) / (g @ g)   # M2
        P = lambda u: u - (u @ n)[..., None] * n if u.ndim > 1 else u - (u @ n) * n
        Sv = P(Hv) / gn
        vSv = v @ Sv
        Jm, Jp = E[0], E[1]
        PJ = P(Jm)
        HJ = jax.vmap(lambda w: f(seed_hvp(x, w)).m)(PJ)
        SJ = P(HJ) / gn
        dJ = Jp - (Jm @ Sv)[:, None] * n                                            # M15
        dJp = -vSv * SJ + (SJ @ v)[:, None] * Sv - (Jp @ Sv)[:, None] * n
        return lam * g, jnp.stack([dJ, dJp]), fa, g
    return R
