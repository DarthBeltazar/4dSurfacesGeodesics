"""Режим A (M18): след Якоби против эмпирической ковариации направлений стохастических лучей пикселя."""
import numpy as np
import jax.numpy as jnp
import pytest

from geo4d.shapes import SHAPES
from geo4d import wormholes as wh
from geo4d.camera import make_camera
from geo4d.cubemap import bake, build_mips
from geo4d.integrators import Config, DONE_ESCAPE
from geo4d import modea as MA

SPP = 2048
NPIX = 16


@pytest.mark.parametrize("aperture", [0.0, 0.3])
def test_footprint_covariance(aperture):
    sh = SHAPES["ellis"]
    x_cam, e_ell = wh.ellis_point(6.0)
    cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=32, height=32,
                      focus=6.0, aperture=aperture)
    cfg = Config(tol=1e-3 * cam.theta_pix, s_max=1e4, n_max=4000)
    esc, ex = wh.ellis_escape(20.0), wh.ellis_exit
    tr = MA.trace_central(sh, cam, cfg, esc, ex)
    flags, _ = MA.detector_flags(cam, tr, MA.Detector())
    pix = np.random.default_rng(1).choice(np.nonzero(~flags)[0], NPIX, replace=False)
    mips = build_mips(bake(lambda d, s: d * 0, 4))
    D, S, St = MA.stochastic_pixels(sh, cam, pix, SPP, cfg, esc, ex, mips, seed=2, return_dirs=True)
    a1, s1, a2, s2 = MA.footprint(tr["dd"][pix], cam.aperture)
    rel = []
    for k in range(NPIX):
        ok = (St[k] == DONE_ESCAPE) & (S[k] == tr["sheet"][pix[k]])
        d0 = tr["d"][pix[k]]
        dev = D[k][ok] - d0
        dev = dev - (dev @ d0)[:, None] * d0
        Sig = s1[k] ** 2 * np.outer(a1[k], a1[k]) + s2[k] ** 2 * np.outer(a2[k], a2[k])
        rel.append(np.linalg.norm(np.cov(dev.T) - Sig) / np.linalg.norm(Sig))
        assert np.linalg.norm(dev.mean(0)) < 0.15 * s1[k]          # смещение среднего — на уровне статистики
    # статистическая ошибка оценки ковариации ≈ √(2/SPP) ≈ 3 %
    assert np.median(rel) < 0.06 and np.max(rel) < 0.15, rel


def test_constant_background_is_constant():
    sh = SHAPES["ellis"]
    x_cam, e_ell = wh.ellis_point(6.0)
    cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=16, height=16,
                      focus=6.0, aperture=0.3)
    tr = MA.trace_central(sh, cam, Config(tol=1e-3 * cam.theta_pix, s_max=1e4, n_max=4000),
                          wh.ellis_escape(20.0), wh.ellis_exit)
    mips = build_mips(bake(lambda d, s: jnp.ones(d.shape) * 0.4, 32))
    col, _, _ = MA.shade_mode_a(tr, mips, cam.aperture)
    ok = tr["status"] == DONE_ESCAPE
    assert np.max(np.abs(col[ok] - 0.4)) < 1e-6
