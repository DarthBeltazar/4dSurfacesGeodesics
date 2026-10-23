"""Червоточины: угол отклонения против квадратуры (M9–M11), поправка хвоста M10, профиль M11."""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import quad

from geo4d.shapes import SHAPES, compact_h
from geo4d.integrators import make_integrator, Config
from geo4d import wormholes as wh, reference as ref

TOL = 1e-8
ERR = 1e-6   # допуск на полный угол φ (рад) при tol = 1e-8


def test_ellis_deflection_50():
    sh = SHAPES["ellis"]
    ell_c, ell_max = 10.0, 30.0
    Ls = np.r_[np.linspace(0.05, 0.9, 25), np.linspace(1.1, 4.0, 25)]   # без L ≈ b (неустойчивая орбита)
    X, V = wh.ellis_rays(ell_c, Ls)
    run = make_integrator(sh, Config(tol=TOL, s_max=500.0), escape=wh.ellis_escape(ell_max),
                          aux=wh.phi_increment, naux=1)
    o = run(jnp.asarray(X), jnp.asarray(V))
    assert np.all(np.asarray(o["status"]) == 2)
    _, sheet, dphi = wh.ellis_exit(o["x"], o["v"])
    Q = [ref.ellis_phi_total(L, ell_c) for L in Ls]
    assert np.max(np.abs(np.asarray(o["aux"][:, 0] + dphi) - [q for q, _ in Q])) < ERR
    assert np.all(np.asarray(sheet) == [s for _, s in Q])


def test_compact_deflection():
    sh = SHAPES["compact_wh"]
    p = sh.params
    Ls = np.r_[np.linspace(0.05, 0.9, 10), np.linspace(1.1, 2.9, 10)]
    X, V = wh.compact_rays(5.0, Ls, **p)
    run = make_integrator(sh, Config(tol=TOL, s_max=100.0), escape=wh.compact_escape(p["rho0"]),
                          aux=wh.phi_increment, naux=1)
    o = run(jnp.asarray(X), jnp.asarray(V))
    assert np.all(np.asarray(o["status"]) == 2)
    _, sheet, dphi = wh.compact_exit(o["x"], o["v"])
    Q = [ref.compact_phi_total(L, 5.0, **p) for L in Ls]
    assert np.max(np.abs(np.asarray(o["aux"][:, 0] + dphi) - [q for q, _ in Q])) < ERR
    assert np.all(np.asarray(sheet) == [s for _, s in Q])


def test_ellis_tail_correction():
    """M10 против прямой квадратуры хвоста."""
    for L in [0.3, 2.0, 5.0]:
        for ell in [10.0, 30.0]:
            q = quad(lambda l: L / (np.sqrt(l * l + 1) * np.sqrt(l * l + 1 - L * L)), ell, np.inf,
                     epsabs=1e-14, epsrel=1e-13)[0]
            assert abs(float(wh.ellis_tail_dphi(ell, L)) - q) < 1e-13


def test_compact_profile():
    """M11: r(0) = b, h < 0 при ρ < b, w ≡ W при ρ ≥ ρ₀, r′(ℓ) монотонно растёт до 1."""
    p = SHAPES["compact_wh"].params
    b, rho0, W = p["b"], p["rho0"], p["W"]
    h = lambda r: np.asarray(compact_h(jnp.asarray(r), b, rho0, W))
    assert abs(h(b)) < 1e-15
    assert np.all(h(np.linspace(0.01, b - 1e-6, 100)) < 0)
    assert np.all(h(np.linspace(rho0, rho0 + 5, 100)) == W * W)
    hp, dhp = ref.compact_profile(b, rho0, W)
    r = np.linspace(b + 1e-3, b + 0.99 * (rho0 - b), 2000)   # q > 1e-2 (дальше хвост ≤ e^{-99} отброшен)
    assert np.all(dhp(r) > 0)
    drho_dl = 1.0 / np.sqrt(1.0 + dhp(r) ** 2 / (4 * hp(r)))  # r′(ℓ) = dρ/dℓ
    assert np.all(np.diff(drho_dl) > -1e-12) and drho_dl[-1] > 1 - 1e-12
