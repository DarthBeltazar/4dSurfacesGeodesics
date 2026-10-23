"""U4: вперёд L, разворот v, назад L — возврат в старт."""
import numpy as np
import pytest

from geo4d.shapes import SHAPES
from helpers import run, rays, path_length, maxnorm


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_reversibility(name):
    sh = SHAPES[name]
    X, V = rays(sh)
    L = path_length(sh)
    o = run(sh, X, V, L)
    b = run(sh, np.asarray(o["x"]), -np.asarray(o["v"]), L)
    assert maxnorm(b["x"], X) / sh.scale < 1e-6
    assert maxnorm(-np.asarray(b["v"]), V) < 1e-6
