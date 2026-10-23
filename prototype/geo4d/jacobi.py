"""Поля Якоби (M14, M15): правая часть луча + 4 полей за один проход Jet15, секционная кривизна."""
import jax.numpy as jnp

from .jet import seed15


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
