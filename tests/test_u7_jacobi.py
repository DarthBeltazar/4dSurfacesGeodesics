"""U7: поля Якоби (M15) против вариаций точных решений и секционные кривизны, восстановленные из полей."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d.shapes import SHAPES, sample
from geo4d.integrators import make_integrator, Config
from geo4d.jacobi import shape_operator, sectional_curvature
from geo4d.jet import seed15

EXACT = ["hyperplane", "S3", "S2xR", "S1xR2"]


def normal(sh, x):
    g = jax.grad(sh.f_vec)(jnp.asarray(x))
    return np.asarray(g / jnp.linalg.norm(g))


def tangent_perp(sh, x, v, rng, k=1):
    """k ортонормированных касательных векторов ⊥ v."""
    n = normal(sh, x)
    out = []
    for _ in range(k):
        e = rng.normal(size=4)
        for b in [n, v, *out]:
            e = e - (e @ b) * b
        out.append(e / np.linalg.norm(e))
    return out


def fields_and_exact(sh, X, V, L, rng):
    """Для каждого луча 2 поля: (J=0, J′=e₁) и (J=e₂, J′=0). Интеграл против jvp точного решения."""
    E0, T0 = [], []
    for x, v in zip(X, V):
        e1, e2 = tangent_perp(sh, x, v, rng, 2)
        E0.append(np.stack([[np.zeros(4), e2], [e1, np.zeros(4)]]))     # E[0]=J, E[1]=J′
        T0.append((e1, e2))
    run = make_integrator(sh, Config(tol=1e-11, s_max=L, jacobi=2))
    o = run(X, V, np.array(E0))
    ex = lambda x, v: sh.exact(x, v, jnp.asarray(L))
    Jex = []
    for x, v, (e1, e2) in zip(X, V, T0):
        # поле 1: δx = 0, δv = e₁;  поле 2: δx = e₂, δv = −⟨S e₂, v⟩ n (M17; J′ = 0)
        H = jax.hessian(sh.f_vec)(jnp.asarray(x))
        S, n = shape_operator(jax.grad(sh.f_vec)(jnp.asarray(x)), H)
        dv2 = -float(e2 @ np.asarray(S) @ v) * np.asarray(n)
        j1 = jax.jvp(ex, (jnp.asarray(x), jnp.asarray(v)), (jnp.asarray(np.zeros(4)), jnp.asarray(e1)))[1][0]
        j2 = jax.jvp(ex, (jnp.asarray(x), jnp.asarray(v)), (jnp.asarray(e2), jnp.asarray(dv2)))[1][0]
        Jex.append(np.stack([np.asarray(j1), np.asarray(j2)]))
    return np.asarray(o["E"][:, 0]), np.array(Jex), o


@pytest.mark.parametrize("name", EXACT)
def test_jacobi_vs_exact_variation(name):
    sh = SHAPES[name]
    X, V = sample(sh, 6, seed=21)
    rng = np.random.default_rng(5)
    L = 2 * np.pi * sh.scale            # S³: через сопряжённую точку πR
    J, Jex, o = fields_and_exact(sh, X, V, L, rng)
    assert np.all(np.asarray(o["status"]) == 1)
    err = np.max(np.linalg.norm(J - Jex, axis=-1))
    assert err < 1e-7, err


def test_s3_closed_form():
    """S³: J(0)=0, J′(0)=e ⇒ J(t) = R sin(t/R) e (e ⊥ x₀, v₀ переносится параллельно и постоянен)."""
    sh = SHAPES["S3"]
    R = sh.params["R"]
    X, V = sample(sh, 4, seed=2)
    rng = np.random.default_rng(1)
    E0, es = [], []
    for x, v in zip(X, V):
        (e,) = tangent_perp(sh, x, v, rng)
        es.append(e)
        E0.append(np.stack([[np.zeros(4)], [e]]))
    for t in [0.5, np.pi * R, 4.0]:
        o = make_integrator(sh, Config(tol=1e-11, s_max=t, jacobi=1))(X, V, np.array(E0))
        J = np.asarray(o["E"][:, 0, 0])
        assert np.max(np.abs(J - R * np.sin(t / R) * np.array(es))) < 1e-8


def _k_from_fields(sh, x, v, e, ts=(0.02, 0.04)):
    """K(v, e) из проинтегрированного поля J(0)=0, J′(0)=e: |J(t)| = t − K t³/6 + O(t⁵) (на геодезической,
    где K постоянна, |J| = sin(√K t)/√K), экстраполяция Ричардсона по двум t."""
    ks = []
    for t in ts:
        o = make_integrator(sh, Config(tol=1e-13, s_max=t, jacobi=1, h_max=t / 8))(
            x[None], v[None], np.stack([[np.zeros(4)], [e]])[None])
        j = np.linalg.norm(np.asarray(o["E"][0, 0, 0]))
        ks.append(6 * (t - j) / t ** 3)
    # ошибка O(t²): K(t) = K + c t²
    return (ks[1] * ts[0] ** 2 - ks[0] * ts[1] ** 2) / (ts[0] ** 2 - ts[1] ** 2)


def analytic_k(name, sh, x, X, Y):
    if name == "S3":
        return 1.0 / sh.params["R"] ** 2
    if name == "S1xR2":
        return 0.0
    if name == "S2xR":
        R = sh.params["R"]
        q = x[:3] / np.linalg.norm(x[:3])
        Pq = lambda a: a[:3] - (a[:3] @ q) * q          # проекция на касательную S²
        Xq, Yq = Pq(X), Pq(Y)
        return ((Xq @ Xq) * (Yq @ Yq) - (Xq @ Yq) ** 2) / R ** 2
    raise KeyError(name)


@pytest.mark.parametrize("name", ["S3", "S2xR", "S1xR2"])
def test_sectional_curvature_from_fields(name):
    """U7: K из интегрированных полей против аналитической; на S¹×R² — ровно 0 (S ранга 1)."""
    sh = SHAPES[name]
    X, V = sample(sh, 4, seed=8)
    rng = np.random.default_rng(3)
    for x, v in zip(X, V):
        for e in tangent_perp(sh, x, v, rng, 2):
            K_true = analytic_k(name, sh, x, v, e)
            K_jac = _k_from_fields(sh, x, v, e)
            JJ = sh.f(seed15(jnp.asarray(x)))
            K_gauss = float(sectional_curvature(shape_operator(JJ.g, JJ.H)[0], jnp.asarray(v), jnp.asarray(e)))
            assert abs(K_gauss - K_true) < 1e-12
            assert abs(K_jac - K_true) < 1e-3 * max(1.0, abs(K_true)), (K_jac, K_true)


def test_s2xr_anisotropy():
    """S²×R: v с компонентой w; поле вдоль S² и вдоль w дают разные K = cos²β/R² и 0."""
    sh = SHAPES["S2xR"]
    R = sh.params["R"]
    x = np.array([R, 0, 0, 0.3])
    beta = 0.6
    v = np.array([0, np.cos(beta), 0, np.sin(beta)])
    e_s2 = np.array([0, 0, 1.0, 0])                        # касательная к S², ⊥ v
    e_mix = np.array([0, -np.sin(beta), 0, np.cos(beta)])  # в плоскости (S², w), ⊥ v
    assert abs(_k_from_fields(sh, x, v, e_s2) - np.cos(beta) ** 2 / R ** 2) < 1e-3
    assert abs(_k_from_fields(sh, x, v, e_mix)) < 1e-6
