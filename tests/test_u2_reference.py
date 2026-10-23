"""U2: рабочий метод (BS32) против DOP853 (rtol = 1e-12) и точных решений M13."""
import numpy as np
import pytest

from geo4d.shapes import SHAPES
from geo4d.reference import dop853
from helpers import run, rays, path_length, maxnorm

ERR = 1e-6   # при tol = 1e-10 (ошибка ∝ tol·число шагов)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_vs_dop853(name):
    sh = SHAPES[name]
    X, V = rays(sh)
    L = path_length(sh)
    o = run(sh, X, V, L)
    assert np.all(np.asarray(o["status"]) == 1)
    xr, vr = dop853(sh, X, V, L)
    assert maxnorm(o["x"], xr) / sh.scale < ERR
    assert maxnorm(o["v"], vr) < ERR


@pytest.mark.parametrize("name", [n for n in sorted(SHAPES) if SHAPES[n].exact is not None])
def test_vs_exact(name):
    sh = SHAPES[name]
    X, V = rays(sh)
    L = path_length(sh)
    o = run(sh, X, V, L)
    xe, ve = sh.exact(X, V, np.asarray(o["s"]))
    assert maxnorm(o["x"], xe) / sh.scale < ERR
    assert maxnorm(o["v"], ve) < ERR
    xr, vr = dop853(sh, X, V, L)          # и сам эталон против точного решения
    xe, ve = sh.exact(X, V, np.full(len(X), L))
    assert maxnorm(xr, xe) / sh.scale < 1e-9
