"""Эталоны (DD-006): DOP853 во float64 по M2 с производными из jax.grad/jax.hessian (независимо от jet-кода),
квадратуры углов отклонения (M9, M11)."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp, quad


def reference_rhs(f_vec, alpha):
    """M2 на автодифференцировании JAX; f_vec: (4,) -> скаляр."""
    grad, hess = jax.grad(f_vec), jax.hessian(f_vec)

    @jax.jit
    def F(y):
        x, v = y[:4], y[4:]
        g, H = grad(x), hess(x)
        lam = -(v @ H @ v + 2 * alpha * (g @ v) + alpha ** 2 * f_vec(x)) / (g @ g)
        return jnp.concatenate([v, lam * g])

    return F


def dop853(shape, x0, v0, L, alpha=None, f_vec=None, rtol=1e-12, atol=1e-12, t_eval=None):
    """Траектории батча лучей до длины пути L. Возвращает (x, v) формы (N, 4) или, при t_eval, (N, T, 8)."""
    F = reference_rhs(f_vec or shape.f_vec, shape.alpha if alpha is None else alpha)
    fun = lambda t, y: np.asarray(F(jnp.asarray(y)))
    out = []
    for x, v in zip(np.atleast_2d(x0), np.atleast_2d(v0)):
        sol = solve_ivp(fun, (0.0, L), np.r_[x, v], method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
        assert sol.success, sol.message
        out.append(sol.y.T if t_eval is not None else sol.y[:, -1])
    out = np.array(out)
    return out if t_eval is not None else (out[:, :4], out[:, 4:])


_Q = dict(epsabs=1e-14, epsrel=1e-13, limit=500)


# ---- M9: Эллис -------------------------------------------------------------------------------

def ellis_phi_total(L, ell_c, b=1.0):
    """Полный угол φ, набранный лучом с моментом L, стартующим внутрь с ℓ = ℓ_c > 0, до бесконечности.
    Возвращает (φ, лист выхода: +1 — тот же, −1 — противоположный)."""
    r = lambda l: np.sqrt(l * l + b * b)
    if L < b:
        F = lambda l: L / (r(l) * np.sqrt(r(l) ** 2 - L * L))
        return quad(F, -np.inf, 0.0, **_Q)[0] + quad(F, 0.0, ell_c, **_Q)[0], -1
    lt = np.sqrt(L * L - b * b)
    G = lambda s: 2 * L / (r(lt + s * s) * np.sqrt(2 * lt + s * s))      # ℓ = ℓ_t + σ²
    return quad(G, 0.0, np.sqrt(ell_c - lt), **_Q)[0] + quad(G, 0.0, np.inf, **_Q)[0], +1


# ---- M11: компактная червоточина -------------------------------------------------------------

def compact_profile(b, rho0, W):
    """h(ρ) и h′(ρ) для M11 (numpy, аналитически)."""
    def h(rho):
        q = 1.0 - (rho - b) / (rho0 - b)
        return np.where(q > 1e-2, W * W * (1.0 - np.exp(1.0 - 1.0 / np.where(q > 1e-2, q, 1.0))), W * W)

    def dh(rho):
        q = 1.0 - (rho - b) / (rho0 - b)
        qs = np.where(q > 1e-2, q, 1.0)
        return np.where(q > 1e-2, W * W * np.exp(1.0 - 1.0 / qs) / qs ** 2 / (rho0 - b), 0.0)

    return h, dh


def compact_phi_total(L, rho_c, b=1.0, rho0=3.0, W=1.5):
    """Как ellis_phi_total, для M11; старт на листе w > 0 при ρ = ρ_c, луч идёт внутрь."""
    h, dh = compact_profile(b, rho0, W)

    def dphi_drho(rho):  # dφ/dρ без множителя 1/√(ρ² − L²), ρ ≤ ρ₀
        hv = h(rho)
        stretch = np.sqrt(1.0 + dh(rho) ** 2 / (4.0 * hv)) if hv > 0 else np.inf
        return L * stretch / rho

    def seg(r_lo, r_hi):
        """∫ от поворотной/горловинной точки r_lo до r_hi (≤ ρ₀), замена ρ = r_lo + σ²."""
        if r_hi <= r_lo:
            return 0.0
        G = lambda s: 2 * s * dphi_drho(r_lo + s * s) / np.sqrt((r_lo + s * s) ** 2 - L * L) \
            if (r_lo + s * s) > L else 0.0
        if L > b:   # поворотная точка ρ = L: (ρ² − L²) = σ²(2L + σ²)
            G = lambda s: 2 * dphi_drho(L + s * s) / np.sqrt(2 * L + s * s)
        return quad(G, 0.0, np.sqrt(r_hi - r_lo), **_Q)[0]

    tail = np.arcsin(L / rho0)  # ∫_{ρ₀}^{∞} L dρ / (ρ √(ρ² − L²))  (плоская зона)
    if L >= rho0:
        raise ValueError("луч не входит в область кривизны")
    head = np.arcsin(L / rho0) - np.arcsin(L / rho_c) if rho_c > rho0 else 0.0  # от ρ_c до ρ₀
    r_in = min(rho_c, rho0)
    if L > b:
        return head + seg(L, r_in) + seg(L, rho0) + tail, +1
    # L < b: проход сквозь горловину; около ρ = b √(1 + w′²) ~ 1/√(ρ − b), замена ρ = b + σ²
    return head + seg(b, r_in) + seg(b, rho0) + tail, -1


def dop853_hit(shape, g_vec, x0, v0, L, alpha=None, rtol=1e-12, atol=1e-12, max_step=0.05):
    """Эталон попадания в объект (DD-005, M27): DOP853 с событием g(x) = 0 (корень по плотному выводу).
    max_step — иначе на прямых шаг больше объекта и событие (смена знака на концах шага) пропускается.
    Возвращает (hit (N,), x (N, 4), v (N, 4), s (N,))."""
    F = reference_rhs(shape.f_vec, shape.alpha if alpha is None else alpha)
    fun = lambda t, y: np.asarray(F(jnp.asarray(y)))
    gj = jax.jit(g_vec)
    ev = lambda t, y: float(gj(jnp.asarray(y[:4])))
    ev.terminal, ev.direction = True, -1
    H, X, Vv, S = [], [], [], []
    for x, v in zip(np.atleast_2d(x0), np.atleast_2d(v0)):
        sol = solve_ivp(fun, (0.0, L), np.r_[x, v], method="DOP853", rtol=rtol, atol=atol, events=ev,
                        max_step=max_step)
        assert sol.success, sol.message
        hit = len(sol.t_events[0]) > 0
        y = sol.y_events[0][0] if hit else sol.y[:, -1]
        H.append(hit), X.append(y[:4]), Vv.append(y[4:]), S.append(sol.t_events[0][0] if hit else L)
    return np.array(H), np.array(X), np.array(Vv), np.array(S)
