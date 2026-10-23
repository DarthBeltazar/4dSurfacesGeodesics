"""GPU-порт (DD-004 п. 3, 10): трассировка и поля Якоби против прототипа; wavefront против мегаядра; BC6H."""
import numpy as np
import jax.numpy as jnp
import pytest

from geo4d import gpu
from geo4d.shapes import SHAPES, sample
from geo4d.integrators import make_integrator, Config
from geo4d.camera import make_camera, jacobi_init
from geo4d.reference import dop853

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]

TOL = 1e-5      # ошибка усечения ≫ округления float32: GPU обязан совпасть с прототипом-f64 по порядку величины


@pytest.mark.parametrize("name", ["S3", "ellipsoid", "tiger", "ellis", "compact_wh"])
def test_trace_vs_prototype(name):
    sh = SHAPES[name]
    L = min(10.0, 10.0 * sh.scale)
    X, V = sample(sh, 32, seed=4)
    X, V = X.astype(np.float32).astype(np.float64), V.astype(np.float32).astype(np.float64)
    xr, vr = dop853(sh, X, V, L)
    g, rep = gpu.trace(name, X, V, tol=TOL, s_max=L, h0=0.1 * sh.scale)
    p = make_integrator(sh, Config(tol=TOL, s_max=L))(X, V)
    e_g = np.max(np.linalg.norm(g["x"] - xr, axis=1))
    e_p = np.max(np.linalg.norm(np.asarray(p["x"]) - xr, axis=1))
    assert np.all(g["status"] == 1)
    assert e_g < 4 * e_p + 1e-5 * sh.scale, (e_g, e_p)
    assert abs(np.mean(g["npass"]) - np.mean(np.asarray(p["nsdf"]))) < 0.05 * np.mean(np.asarray(p["nsdf"]))


@pytest.mark.parametrize("jet", ["jet15", "hvp"])
def test_jacobi_vs_prototype(jet):
    sh = SHAPES["ellipsoid"]
    X, V = sample(sh, 16, seed=5)
    rng = np.random.default_rng(6)
    E0, D0 = [], []
    for x, v in zip(X, V):
        cam = make_camera(sh, x, v, rng.normal(size=4), fov=60, width=64, height=64, focus=3.0)
        e, d = jacobi_init(cam, np.asarray(0.2), np.asarray(-0.1))
        E0.append(e)
        D0.append(d)
    E0 = np.array(E0).astype(np.float32).astype(np.float64)
    D0 = np.array(D0).astype(np.float32).astype(np.float64)
    X = X.astype(np.float32).astype(np.float64)
    ref = make_integrator(sh, Config(method="dopri5", tol=1e-12, s_max=5.0, jacobi=4))(X, D0, E0)
    g, _ = gpu.trace("ellipsoid", X, D0, E=E0, tol=TOL, s_max=5.0, h0=0.1, nj=4, jacobi_jet=jet)
    p = make_integrator(sh, Config(tol=TOL, s_max=5.0, jacobi=4))(X, D0, E0)
    sc = np.max(np.abs(np.asarray(ref["E"])))
    e_g = np.max(np.abs(g["E"] - np.asarray(ref["E"]))) / sc
    e_p = np.max(np.abs(np.asarray(p["E"]) - np.asarray(ref["E"]))) / sc
    assert e_g < 4 * e_p + 1e-5, (e_g, e_p)


def test_wavefront_matches_megakernel():
    from geo4d import gpu_modea as GM, wormholes as wh, modea as MA
    from geo4d.cubemap import bake
    from geo4d.render import background
    sh = SHAPES["ellis"]
    cube = gpu.cube_for_gpu(bake(background, 128))
    x, e = wh.ellis_point(6.0)
    cam = make_camera(sh, x, -e, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=64, height=64, focus=6.0, aperture=0.2)
    tol = 3e-3 * cam.theta_pix
    a = GM.render("ellis", cam, cube, tol, 20.0, nfb=64)
    b = GM.render_wavefront("ellis", cam, cube, tol, 20.0, K=16, nfb=64)
    assert np.array_equal(a["flags"], b["flags"])
    ok = ~a["flags"]
    assert MA.psnr(a["color"][ok], b["color"][ok]) > 50       # без фолбэка: только порядок шагов (FSAL на границе диспетчеризаций)
    assert MA.psnr(a["color"], b["color"]) > 35               # с фолбэком: разные порции сэмплов, шум 64 spp


def test_bc6h_close_to_float():
    from geo4d.cubemap import bake
    from geo4d.render import background
    cube = gpu.cube_for_gpu(bake(background, 128))
    rng = np.random.default_rng(0)
    n = 8192
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    D = gpu.f32(np.c_[d, rng.integers(0, 2, n)])
    Z = np.zeros((n, 4), np.float32)
    spv = gpu.compile_shader("cube_probe.slang", "cube_probe")
    res = {}
    for fmt in ["rgba32f", "bc6h"]:
        out, _ = gpu.run(spv, "cube_probe", [("in", D), ("in", Z), ("in", Z), ("out", ("f4", (n, 4))),
                                             ("cube", (cube, fmt, 16))], push=[("u32", n), ("u32", 0)], groups=gpu.groups_for(n))
        res[fmt] = out[3][:, :3]
    rel = np.abs(res["bc6h"] - res["rgba32f"]) / np.maximum(res["rgba32f"], 1e-3)
    assert np.median(rel) < 0.02 and np.percentile(rel, 99) < 0.15
