"""Матрица переноса (M16, M17) против конечных разностей соседних лучей на всех фигурах.

1) Согласованность: с одним и тем же фиксированным шагом расхождение полей и разностей сходится к нулю
   с порядком метода (≈ 3): M15 — линеаризация только на M, а стадии RK чуть сходят с M.
2) Точность: поля при адаптивном tol против соседей, посчитанных DOPRI5 с tol 1e-13; критерий < 1e-4.
Ошибка столбца нормируется на полную норму (∂x, ∂v), которая не обращается в ноль и у каустики, поэтому
случаи не исключаются; ratio (det позиционного блока пиксельных полей к плоскому L²·det J′(0)) печатается
для диагностики.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from geo4d.shapes import SHAPES, sample, reparam
from geo4d.integrators import make_integrator, Config
from geo4d.camera import make_camera, jacobi_init, pixel_dirs, lens_ray
from geo4d.transfer import state_derivatives, exit_dir_derivatives, transfer_matrix_4x4
from geo4d import wormholes as wh

EPS_PIX = 1e-3        # сдвиг пикселя для разностей, пиксели
EPS_AP = 1e-4         # сдвиг апертуры, в единицах ℓ_s
NCAM = 3

CASES = sorted(SHAPES) + ["S3/mult", "ellipsoid/mult", "tiger/fexp", "ellis/mult"]


def shape_of(name):
    if "/" in name:
        base, kind = name.split("/")
        return reparam(SHAPES[base], kind)
    return SHAPES[name]


def cameras(sh, n, seed):
    X, V = sample(sh, n, seed=seed)
    rng = np.random.default_rng(seed + 1000)          # отдельный поток: не совпадать с сэмплером точек
    cams = []
    for x, v in zip(X, V):
        cam = make_camera(sh, x, v, rng.normal(size=4), fov=60.0, width=64, height=64, focus=3 * sh.scale)
        U, Vv = rng.uniform(-0.4, 0.4, 2)
        cams.append((cam, U, Vv))
    return cams


def neighbours(sh, cam, U, V):
    """Начальные состояния 8 соседей (±ε по каждому из 4 входов) и шаги ε (в единицах входа)."""
    dpix = 2 * cam.tf / cam.width
    xs, vs = [], []
    for k in range(4):
        for sgn in (+1, -1):
            if k < 2:
                a = sgn * EPS_AP * sh.scale
                x, v = lens_ray(sh, cam, U, V, a if k == 0 else 0.0, a if k == 1 else 0.0)
            else:
                dU, dV = (sgn * EPS_PIX * dpix, 0.0) if k == 2 else (0.0, sgn * EPS_PIX * dpix)
                x, v = cam.x, pixel_dirs(cam, np.asarray(U + dU), np.asarray(V + dV))
            xs.append(x)
            vs.append(v)
    eps = np.array([EPS_AP * sh.scale] * 2 + [EPS_PIX] * 2)
    return np.array(xs), np.array(vs), eps


def central_fd(Y, eps):
    Y = Y.reshape(4, 2, -1)
    return (Y[:, 0] - Y[:, 1]) / (2 * eps[:, None])


def run_case(sh, cam, U, V, L, cfg_c, cfg_n):
    E0, d0 = jacobi_init(cam, np.asarray(U), np.asarray(V))
    oc = make_integrator(sh, cfg_c.with_(s_max=L, jacobi=4))(cam.x[None], d0[None], E0[None])
    xc, vc, E = (np.asarray(oc[k][0]) for k in ("x", "v", "E"))
    Xn, Vn, eps = neighbours(sh, cam, U, V)
    on = make_integrator(sh, cfg_n.with_(s_max=L))(Xn, Vn)
    fd_x = central_fd(np.asarray(on["x"]), eps)
    fd_v = central_fd(np.asarray(on["v"]), eps)
    dx, dv = (np.asarray(t) for t in state_derivatives(sh.f, jnp.asarray(xc), jnp.asarray(vc), jnp.asarray(E)))
    M = transfer_matrix_4x4(sh.f, xc, vc, E)
    # «вне каустик»: позиционный блок пиксельных полей против плоского L·J′(0)
    det = abs(np.linalg.det(M[:2, 2:]))
    flat = L ** 2 * abs(np.linalg.det(transfer_matrix_4x4(sh.f, cam.x, d0, E0)[2:, 2:]))
    rel = max(np.max(np.linalg.norm(dx - fd_x, axis=1) / np.linalg.norm(np.c_[fd_x, fd_v], axis=1)),
              np.max(np.linalg.norm(dv - fd_v, axis=1) / np.linalg.norm(np.c_[fd_x, fd_v], axis=1)))
    return rel, det / flat


@pytest.mark.parametrize("name", CASES)
def test_consistency_fixed_steps(name):
    sh = shape_of(name)
    L = min(10.0, 10.0 * sh.scale)
    for cam, U, V in cameras(sh, NCAM, seed=1):
        errs = []
        for div in (40, 80):
            cfg = Config(method="bs32", fixed=True, h=sh.scale / div, h_max=np.inf)
            errs.append(run_case(sh, cam, U, V, L, cfg, cfg)[0])
        assert errs[1] < 1e-4 and (errs[1] < 1e-8 or np.log2(errs[0] / errs[1]) > 2.3), errs


@pytest.mark.parametrize("name", CASES)
def test_accuracy_vs_fd(name):
    sh = shape_of(name)
    L = min(10.0, 10.0 * sh.scale)
    for cam, U, V in cameras(sh, NCAM, seed=2):
        rel, ratio = run_case(sh, cam, U, V, L, Config(tol=1e-10), Config(method="dopri5", tol=1e-13))
        assert rel < 1e-4, (rel, ratio)


@pytest.mark.parametrize("scene", ["ellis", "compact_wh"])
def test_exit_direction_derivatives(scene):
    """∂d∞ (jvp выхода вдоль полей) против разностей асимптотических направлений соседних лучей."""
    sh = SHAPES[scene]
    if scene == "ellis":
        x_cam, e_ell = wh.ellis_point(6.0)
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=64, height=64, focus=4.0)
        esc, ex = wh.ellis_escape(20.0), wh.ellis_exit
    else:
        p = sh.params
        # камера внутри шара кривизны на верхнем листе, у края плоской зоны
        x0 = np.array([3.5, 0.0, 0.0, p["W"]])
        cam = make_camera(sh, x0, [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=64, height=64, focus=4.0)
        esc, ex = wh.compact_escape(p["rho0"]), wh.compact_exit
    rng = np.random.default_rng(0)
    tested = 0
    for U, V in list(rng.uniform(-0.9, 0.9, (8, 2))) + [(1e-7, 2e-7)]:   # последний — почти радиальный луч
        E0, d0 = jacobi_init(cam, np.asarray(U), np.asarray(V))
        oc = make_integrator(sh, Config(tol=1e-11, s_max=1e4, jacobi=4), escape=esc)(cam.x[None], d0[None], E0[None])
        if int(oc["status"][0]) != 2:
            continue
        xc, vc, E = (jnp.asarray(oc[k][0]) for k in ("x", "v", "E"))
        dinf, dd = exit_dir_derivatives(ex, sh.f, xc, vc, E)
        assert np.all(np.isfinite(np.asarray(dd)))
        Xn, Vn, eps = neighbours(sh, cam, U, V)
        on = make_integrator(sh, Config(method="dopri5", tol=1e-13, s_max=1e4), escape=esc)(Xn, Vn)
        if not np.all(np.asarray(on["status"]) == 2):
            continue
        dn, sheet, _ = ex(on["x"], on["v"])
        if len(set(np.asarray(sheet).tolist())) > 1:
            continue
        fd = central_fd(np.asarray(dn), eps)
        rel = np.max(np.linalg.norm(np.asarray(dd) - fd, axis=1)) / np.max(np.linalg.norm(fd, axis=1))
        assert rel < 1e-4, (U, V, rel)
        tested += 1
    assert tested >= 6
