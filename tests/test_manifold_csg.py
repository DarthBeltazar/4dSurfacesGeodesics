"""M38 (черн.): CSG многообразий гладким минимумом с компактным носителем (docs/TODO.md, п. 1).

- s(t) (csabs): чётная, выпуклая, |s′| ≤ 1, s = |t| при |t| ≥ k, s(0) = k/2;
- csmin/csmax вне полосы |a − b| < k — ровно min/max;
- U1: Jet7/Jet15 против jax.grad/jax.hessian на «плоскость ∪ S³» и «слой − шар»;
- носитель: вне шара |q| < r_c на M кривизна ровно ноль (плоские зоны M29 остаются точными);
- шейка открыта при k > 2d (ось внутри), закрыта при k < 2d;
- геодезические через шейку с выведенными ℓ_s, hcap — против DOP853."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d.integrators import Config, make_integrator
from geo4d.jet import seed7, seed15
from geo4d.reference import dop853
from geo4d.shapes import csabs, csmax, csmin, plane_sphere_union, sample, slab_minus_ball

R, D, K = 3.0, 0.5, 2.0
CASES = {"plane_sphere": (plane_sphere_union(R, D, K), R + 1.25 * K),
         "slab_ball": (slab_minus_ball(1.5, 2.0, 1.0), 2.0 + 1.0)}


def test_csabs_shape():
    k = 0.7
    t = jnp.linspace(-1.5 * k, 1.5 * k, 20001)
    s = np.asarray(csabs(t, k))
    ds = np.asarray(jax.vmap(jax.grad(lambda u: csabs(u, k)))(t))
    d2 = np.asarray(jax.vmap(jax.grad(jax.grad(lambda u: csabs(u, k))))(t))
    assert np.allclose(s, s[::-1], rtol=0, atol=1e-15)
    assert np.all(s >= np.abs(np.asarray(t)) - 1e-15)
    assert np.all(np.abs(ds) <= 1 + 1e-12)
    assert d2.min() > -1e-9, d2.min()
    out = np.abs(np.asarray(t)) >= k
    assert np.array_equal(s[out], np.abs(np.asarray(t))[out])
    assert abs(s[len(s) // 2] - 0.5 * k) < 1e-15


def test_csmin_exact_outside_band():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=1000) * 3, rng.normal(size=1000) * 3
    k = 0.5
    out = np.abs(a - b) >= k
    assert np.array_equal(np.asarray(csmin(a, b, k))[out], np.minimum(a, b)[out])
    assert np.array_equal(np.asarray(csmax(a, b, k))[out], np.maximum(a, b)[out])
    m = np.asarray(csmin(a, b, k))
    assert np.all(m <= np.minimum(a, b)) and np.all(m >= np.minimum(a, b) - 0.25 * k - 1e-15)


@pytest.mark.parametrize("name", sorted(CASES))
def test_jets_vs_autodiff(name):
    sh, _ = CASES[name]
    P, _ = sample(sh, 600, seed=1)
    rng = np.random.default_rng(101)
    P = jnp.asarray(P + 0.05 * rng.normal(size=P.shape))
    V = jnp.asarray(rng.normal(size=P.shape))
    J7 = jax.vmap(lambda x, v: sh.f(seed7(x, v)))(P, V)
    J15 = jax.vmap(lambda x: sh.f(seed15(x)))(P)
    G = jax.vmap(jax.grad(sh.f_vec))(P)
    H = jax.vmap(jax.hessian(sh.f_vec))(P)

    def close(got, ref):
        err = np.abs(np.asarray(got) - np.asarray(ref)) / np.maximum(1.0, np.abs(np.asarray(ref)))
        assert np.all(np.isfinite(np.asarray(got))) and err.max() < 1e-10, err.max()
    close(J7.a, jax.vmap(sh.f_vec)(P))
    close(J7.d, G)
    close(2 * J7.c, jnp.einsum("ni,nij,nj->n", V, H, V))
    close(J15.g, G)
    close(J15.H, H)


@pytest.mark.parametrize("name", sorted(CASES))
def test_curvature_inside_carrier(name):
    """На M вне шара |q| < r_c — vᵀHv = 0 ровно (кривизна ноль, плоские зоны M29), внутри — регулярно."""
    sh, r_c = CASES[name]
    P, V = sample(sh, 3000, seed=2)
    P, V = jnp.asarray(P), jnp.asarray(V)
    J = jax.vmap(lambda x, v: sh.f(seed7(x, v)))(P, V)
    q = np.linalg.norm(np.asarray(P[:, :3]), axis=1)
    assert np.sum(q >= r_c) > 100
    assert np.all(np.asarray(J.c)[q >= r_c] == 0.0)
    assert np.linalg.norm(np.asarray(J.d), axis=1).min() > 0.1


def test_neck_open_iff_k_gt_2d():
    """На оси q = 0 посередине зазора f = d/2 − k/4: при k > 2d внутри (шейка), при k < 2d — снаружи."""
    for k, inside in [(1.2, True), (0.8, False)]:
        sh = plane_sphere_union(R, D, k)
        assert (float(sh.f_vec(jnp.array([0.0, 0.0, 0.0, 0.5 * D]))) < 0) == inside


@pytest.mark.parametrize("name", sorted(CASES))
def test_geodesics_vs_dop853(name):
    sh, r_c = CASES[name]
    n, L = 12, 30.0
    X = np.c_[np.full(n, -(r_c + 2.0)), np.linspace(0.0, 2.0, n), np.zeros(n),
              np.full(n, 0.0 if name == "plane_sphere" else 1.5)]
    V = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    xr, _ = dop853(sh, X, V, L)
    p = make_integrator(sh, Config(tol=1e-8, s_max=L))(jnp.asarray(X), jnp.asarray(V))
    assert np.all(np.asarray(p["status"]) == 1)
    e = np.linalg.norm(np.asarray(p["x"]) - xr, axis=1)
    assert e.max() < 1e-4, e.max()
