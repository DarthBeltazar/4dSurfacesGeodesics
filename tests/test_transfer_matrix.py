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

from geo4d.shapes import SHAPES
from geo4d.integrators import make_integrator, Config
from geo4d.camera import make_camera, jacobi_init
from geo4d.transfer import exit_dir_derivatives
from geo4d import wormholes as wh

from geo4d.transfer import shape_of, cameras, neighbours, central_fd, run_case

NCAM = 3
CASES = sorted(SHAPES) + ["S3/mult", "ellipsoid/mult", "tiger/fexp", "ellis/mult"]


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
