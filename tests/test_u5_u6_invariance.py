"""U5: замена f на φ(f) и на f·(положительная функция). U6: вращение SO(4) + сдвиг всей сцены."""
import zlib

import numpy as np
import pytest

from geo4d.shapes import SHAPES, reparam, transformed, random_rotation
from helpers import run, rays, path_length, maxnorm


@pytest.mark.parametrize("kind", ["cubic", "fexp", "mult"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_u5_reparam(name, kind):
    sh = SHAPES[name]
    X, V = rays(sh)
    L = path_length(sh)
    a = run(sh, X, V, L)
    b = run(reparam(sh, kind), X, V, L)
    assert maxnorm(a["x"], b["x"]) / sh.scale < 1e-6
    assert maxnorm(a["v"], b["v"]) < 1e-6


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_u6_rigid_motion(name):
    sh = SHAPES[name]
    Q = random_rotation(zlib.crc32(name.encode()))
    t = np.array([0.3, -1.2, 2.0, 0.7])
    X, V = rays(sh)
    L = path_length(sh)
    a = run(sh, X, V, L)
    b = run(transformed(sh, Q, t), X @ Q.T + t, V @ Q.T, L)
    assert maxnorm(np.asarray(a["x"]) @ Q.T + t, b["x"]) / sh.scale < 1e-6
    assert maxnorm(np.asarray(a["v"]) @ Q.T, b["v"]) < 1e-6


def test_u6_image_ellis():
    """U6: та же картинка Эллиса после вращения SO(4) и сдвига сцены вместе с камерой."""
    from geo4d.integrators import Config
    from geo4d import wormholes as wh
    from geo4d.render import pinhole_dirs, render_wormhole

    sh = SHAPES["ellis"]
    Q, t = random_rotation(5), np.array([1.0, -2.0, 0.5, 3.0])
    x_cam, e_ell = wh.ellis_point(6.0)
    dirs, th = pinhole_dirs(32, 32, 90.0, -e_ell, np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]))
    cfg = Config(tol=1e-3 * th, s_max=1e4, n_max=4000)
    a = render_wormhole(sh, x_cam, dirs, cfg, wh.ellis_escape(20.0), wh.ellis_exit)
    Qt = np.asarray(Q.T)
    esc = lambda x, v: wh.ellis_escape(20.0)(Qt @ (x - t), Qt @ v)
    ex = lambda x, v: wh.ellis_exit((x - t) @ Q, v @ Q)
    b = render_wormhole(transformed(sh, Q, t), Q @ x_cam + t, dirs @ Q.T, cfg, esc, ex)
    assert np.max(np.abs(a[0] - b[0])) < 1e-3          # цвета
    assert np.max(np.linalg.norm(a[4] - b[4], axis=1)) < 0.05 * th   # выходные направления
