"""M26–M28: попадание в объекты {g = 0} ∩ M (предел шага, эрмитово уточнение) против DOP853 с событием; кадр оттенения."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d import shapes as S, objects as O, reference as R, wormholes as wh
from geo4d.integrators import Config

TOL = 1e-8


def _aim(shape, X, T, rng, noise):
    G = np.asarray(jax.vmap(jax.grad(shape.f_vec))(jnp.asarray(X)))
    n = G / np.linalg.norm(G, axis=1, keepdims=True)
    d = T - X + noise * rng.normal(size=X.shape)
    d -= np.sum(d * n, 1, keepdims=True) * n
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _case(name):
    rng = np.random.default_rng(1)
    if name == "hyperplane":
        sh = S.SHAPES["hyperplane"]
        objs = [O.ball4((3, 0.3, 0, 0), 1.0)]
        X, T, L, noise = np.c_[rng.uniform(-1, 1, (16, 3)), np.zeros(16)], np.array([3, 0.3, 0, 0]), 10, 0.8
    elif name == "S3":
        sh = S.SHAPES["S3"]
        objs = [O.ball4((0, 0, 0, -1), 0.3)]
        X, T, L, noise = np.asarray(S.sample(sh, 16, seed=5)[0]), np.array([0, 0, 0, -1.0]), 7, 0.3
    elif name == "ellipsoid":
        sh = S.SHAPES["ellipsoid"]
        P = np.asarray(S.sample(sh, 17, seed=7)[0])
        objs = [O.ball4(P[0], 0.3)]
        X, T, L, noise = P[1:], P[0], 10, 0.2
    elif name == "ellis":      # шар на другом листе; радиальный луч выходит на ту же сторону по xyz
        sh = S.SHAPES["ellis"]
        c, _ = wh.ellis_point(-3.0)
        objs = [O.ball4(c, 0.8)]
        x, _ = wh.ellis_point(5.0)
        X, T, L, noise = np.tile(x, (16, 1)), np.zeros(4), 20, 0.08
    else:                      # компактная: шар в зоне кривизны и тор в плоской зоне нижнего листа
        sh = S.SHAPES["compact_wh"]
        p = sh.params
        h, _ = R.compact_profile(p["b"], p["rho0"], p["W"])
        c = np.array([0, 2.0, 0, np.sqrt(h(2.0))])
        objs = [O.ball4(c, 0.4), O.torus_in_sheet((6, 0, 0, -1.5), 1.0, 0.3)]
        X = np.c_[np.full(16, 5.0), rng.uniform(-1, 1, (16, 2)), np.full(16, 1.5)]
        T, L, noise = np.r_[np.tile(c, (8, 1)), np.zeros((8, 4))], 20, 0.3
    return sh, objs, X, _aim(sh, X, T, rng, noise), L


@pytest.mark.parametrize("name", ["hyperplane", "S3", "ellipsoid", "ellis", "compact_wh"])
def test_hit_vs_dop853(name):
    sh, objs, X, V, L = _case(name)
    gmin = lambda x: jnp.min(jnp.stack([o.g_vec(x) for o in objs]))
    rh, rx, rv, rs = R.dop853_hit(sh, gmin, X, V, L)
    o = O.trace_hits(sh, objs, X, V, Config(tol=TOL, h=0.05 * sh.scale, s_max=L), eps_hit=1e-12)
    assert rh.sum() >= 4
    assert np.array_equal(o["hit"], rh)
    b = rh
    assert np.max(np.linalg.norm(o["x"][b] - rx[b], axis=1)) < 50 * TOL
    assert np.max(np.linalg.norm(o["v"][b] - rv[b], axis=1)) < 50 * TOL
    assert np.max(np.abs(o["s"][b] - rs[b])) < 50 * TOL
    assert max(abs(float(gmin(jnp.asarray(x)))) for x in o["x"][b]) < 1e-12
    assert max(abs(float(sh.f_vec(jnp.asarray(x)))) for x in o["x"][b]) < 1e-12


def test_grazing_hyperplane_exact():
    """Скользящие лучи: прицельный параметр → радиус шара; точное попадание (−√(1 − p²), p, 0, 0)."""
    sh = S.SHAPES["hyperplane"]
    p = np.array([0.0, 0.9, 0.99, 0.999, 0.99999])
    X = np.c_[np.full(5, -5.0), p, np.zeros(5), np.zeros(5)]
    V = np.tile([1.0, 0, 0, 0], (5, 1))
    o = O.trace_hits(sh, [O.ball4((0, 0, 0, 0), 1.0)], X, V, Config(tol=TOL, s_max=10), eps_hit=1e-12)
    assert o["hit"].all()
    exact = np.c_[-np.sqrt(1 - p ** 2), p, np.zeros(5), np.zeros(5)]
    assert np.max(np.linalg.norm(o["x"] - exact, axis=1)) < 1e-9


@pytest.mark.parametrize("name", ["ellipsoid", "compact_wh", "S3"])
def test_shading_frame_orthonormal(name):
    sh = S.SHAPES[name]
    P = np.asarray(S.sample(sh, 20, seed=3)[0])
    obj = O.ball4(P[0] + 0.1, 0.5)
    frame = jax.jit(lambda x: O.shading_frame(sh.f_vec, obj.g_vec, x))
    for x in P[1:]:
        n, ng, t1, t2, png = frame(jnp.asarray(x))
        B = np.stack([n, ng, t1, t2])
        assert np.max(np.abs(B @ B.T - np.eye(4))) < 1e-12
        gg = np.asarray(jax.grad(obj.g_vec)(jnp.asarray(x)))
        assert np.isclose(np.asarray(ng) @ gg, png)            # n_g — направление проекции ∇g
