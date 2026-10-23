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
