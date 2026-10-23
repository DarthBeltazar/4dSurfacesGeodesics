"""Подготовка к GPU (DD-004 п. 4, 7): HVP-jet (M23), Якоби на HVP, производная выхода дуальными числами (M24)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d.jet import seed_hvp
from geo4d.shapes import SHAPES, sample, two_compact_wormholes
from geo4d.integrators import make_integrator, Config
from geo4d.camera import make_camera, jacobi_init
from geo4d.transfer import exit_dir_derivatives, state_derivatives
from geo4d import wormholes as wh

ALL = dict(SHAPES, two_compact_wh=two_compact_wormholes())


@pytest.mark.parametrize("name", sorted(ALL))
def test_u1_hvp(name):
    """U1 для M23: f, ∇f, ∇f·w, H·w против jax.grad / jax.hessian."""
    sh = ALL[name]
    P, _ = sample(sh, 500, seed=1)
    rng = np.random.default_rng(9)
    P = jnp.asarray(P + 0.05 * sh.scale * rng.normal(size=P.shape))
    W = jnp.asarray(rng.normal(size=P.shape))
    J = jax.vmap(lambda x, w: sh.f(seed_hvp(x, w)))(P, W)
    G = jax.vmap(jax.grad(sh.f_vec))(P)
    H = jax.vmap(jax.hessian(sh.f_vec))(P)
    for got, ref in [(J.a, jax.vmap(sh.f_vec)(P)), (J.d, G), (J.b, jnp.sum(G * W, -1)),
                     (J.m, jnp.einsum("nij,nj->ni", H, W))]:
        err = np.abs(np.asarray(got) - np.asarray(ref)) / np.maximum(1.0, np.abs(np.asarray(ref)))
        assert err.max() < 1e-10


@pytest.mark.parametrize("name", ["S3", "S2xR", "tiger", "ellis", "compact_wh"])
def test_jacobi_hvp_matches_jet15(name):
    """Якоби на HVP (1 + k проходов) против Jet15: на M формулы совпадают; на стадиях — O(локальной ошибки)."""
    sh = SHAPES[name]
    X, V = sample(sh, 4, seed=3)
    rng = np.random.default_rng(4)
    cams = [make_camera(sh, x, v, rng.normal(size=4), fov=60, width=64, height=64, focus=3 * sh.scale) for x, v in zip(X, V)]
    E0 = np.stack([jacobi_init(c, np.asarray(0.1), np.asarray(-0.2))[0] for c in cams])
    D0 = np.stack([jacobi_init(c, np.asarray(0.1), np.asarray(-0.2))[1] for c in cams])
    L = min(10.0, 10.0 * sh.scale)
    a = make_integrator(sh, Config(tol=1e-10, s_max=L, jacobi=4))(X, D0, E0)
    b = make_integrator(sh, Config(tol=1e-10, s_max=L, jacobi=4, jacobi_jet="hvp"))(X, D0, E0)
    scale = np.max(np.abs(np.asarray(a["E"])))
    assert np.max(np.abs(np.asarray(a["E"]) - np.asarray(b["E"]))) < 1e-6 * scale
    assert np.max(np.abs(np.asarray(a["x"]) - np.asarray(b["x"]))) < 1e-9


def _exit_states(scene, n=12):
    sh = SHAPES[scene]
    if scene == "ellis":
        x_cam, e_ell = wh.ellis_point(6.0)
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=64, height=64, focus=4.0)
        esc = wh.ellis_escape(20.0)
    else:
        cam = make_camera(sh, np.array([3.5, 0, 0, sh.params["W"]]), [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                          fov=90, width=64, height=64, focus=4.0)
        esc = wh.compact_escape(sh.params["rho0"])
    rng = np.random.default_rng(0)
    UV = np.r_[rng.uniform(-0.9, 0.9, (n - 2, 2)), [[1e-7, 2e-7], [0.0, 0.0]]]   # + почти/точно радиальные
    E0, D0 = jacobi_init(cam, UV[:, 0], UV[:, 1])
    o = make_integrator(sh, Config(tol=1e-10, s_max=1e4, jacobi=4), escape=esc)(np.broadcast_to(cam.x, D0.shape), D0, E0)
    ok = np.asarray(o["status"]) == 2
    return sh, np.asarray(o["x"])[ok], np.asarray(o["v"])[ok], np.asarray(o["E"])[ok]


@pytest.mark.parametrize("scene", ["ellis", "compact_wh"])
def test_exit_derivative_dual_vs_jvp(scene):
    sh, X, V, E = _exit_states(scene)
    assert len(X) >= 8
    gen = wh.ellis_dir_generic if scene == "ellis" else wh.compact_dir_generic
    ref_fn = wh.ellis_exit if scene == "ellis" else wh.compact_exit
    for x, v, Ei in zip(X, V, E):
        x, v, Ei = jnp.asarray(x), jnp.asarray(v), jnp.asarray(Ei)
        d_ref, dd_ref = exit_dir_derivatives(ref_fn, sh.f, x, v, Ei)
        dx, dv = state_derivatives(sh.f, x, v, Ei)
        d, dd = wh.exit_derivatives_dual(gen, x, v, dx, dv)
        assert np.all(np.isfinite(np.asarray(dd)))
        assert np.max(np.abs(np.asarray(d) - np.asarray(d_ref))) < 1e-12
        assert np.max(np.abs(np.asarray(dd) - np.asarray(dd_ref))) < 1e-9 * max(1.0, float(np.max(np.abs(dd_ref))))


def test_exact_radial_ray_finite():
    """Точно радиальный луч (L = 0): значения и производные конечны (jvp старой формы даёт NaN/inf не всегда)."""
    x = jnp.array([20.0, 0.0, 0.0, 3.0])
    v = jnp.array([0.999, 0.0, 0.0, 0.0447])
    dx = jnp.eye(4)[:2]
    dv = jnp.eye(4)[1:3]
    d, dd = wh.exit_derivatives_dual(wh.ellis_dir_generic, x, v, dx, dv)
    assert np.all(np.isfinite(np.asarray(d))) and np.all(np.isfinite(np.asarray(dd)))


@pytest.mark.parametrize("op", ["cos", "sin", "sinc", "cos_sqrt", "sinc_sqrt"])
def test_new_ops_second_order(op):
    """M4 для новых унарных функций: Jet7 (g, g′, g″) против jax.grad/jax.hessian, включая окрестность нуля."""
    from geo4d.jet import ops, seed7
    fn = getattr(ops, op)
    f = lambda p: fn(p[0] * p[0] * 0.3 + p[1] * 0.1 + p[2] * p[3])     # аргумент около 0 и вдали от него
    fv = lambda x: f(tuple(x[k] for k in range(4)))
    rng = np.random.default_rng(1)
    P = rng.normal(size=(400, 4)) * np.r_[np.full(200, 1e-2), np.full(200, 1.5)][:, None]
    if op.endswith("sqrt"):
        P = np.abs(P)
    P, V = jnp.asarray(P), jnp.asarray(rng.normal(size=(400, 4)))
    J = jax.vmap(lambda x, v: f(seed7(x, v)))(P, V)
    G = jax.vmap(jax.grad(fv))(P)
    H = jax.vmap(jax.hessian(fv))(P)
    assert np.max(np.abs(np.asarray(J.d) - np.asarray(G))) < 1e-9
    assert np.max(np.abs(np.asarray(2 * J.c) - np.asarray(jnp.einsum("ni,nij,nj->n", V, H, V)))) < 1e-8


def test_flat_flight_jacobi_carry():
    """M25: камера компактной червоточины в плоской зоне — поля, перенесённые через аналитический пролёт,
    против разностей соседних лучей, проинтегрированных от самой камеры."""
    from geo4d import modea as MA
    from geo4d.transfer import neighbours, central_fd
    from geo4d.camera import pixel_uv
    sh = SHAPES["compact_wh"]
    p = sh.params
    cam = make_camera(sh, np.array([8.0, 0, 0, p["W"]]), [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
                      fov=60, width=16, height=16, focus=5.0)
    esc = wh.compact_escape(p["rho0"])
    tr = MA.trace_central(sh, cam, Config(tol=1e-11, s_max=1e4), esc, wh.compact_exit, flight=(np.zeros(3), p["rho0"]))
    assert 0 < tr["active"].sum() < len(tr["active"])
    U, V = (a.ravel() for a in pixel_uv(cam))
    rng = np.random.default_rng(0)
    for pix in list(rng.choice(np.nonzero(tr["active"])[0], 5, replace=False)) + [int(np.nonzero(~tr["active"])[0][0])]:
        Xn, Vn, eps = neighbours(sh, cam, U[pix], V[pix])
        on = make_integrator(sh, Config(method="dopri5", tol=1e-13, s_max=1e4), escape=esc)(Xn, Vn)
        dn, sheet, _ = wh.compact_exit(on["x"], on["v"])
        if len(set(np.asarray(sheet).tolist())) > 1:
            continue
        fd = central_fd(np.asarray(dn), eps)
        rel = np.max(np.linalg.norm(tr["dd"][pix] - fd, axis=1)) / np.max(np.linalg.norm(fd, axis=1))
        assert rel < 1e-4, (pix, rel)
