"""U3: |v| = 1, f = 0, моменты Lᵢⱼ, интеграл Иоахимсталя вдоль всего пути (дрейф)."""
import numpy as np
import jax
import pytest

from geo4d.shapes import SHAPES, moments
from helpers import run, rays, path_length

REC = 12000


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_invariants(name):
    sh = SHAPES[name]
    X, V = rays(sh)
    o = run(sh, X, V, path_length(sh), record=REC)
    rec, n = np.asarray(o["rec"]), np.asarray(o["nacc"])
    assert np.all(n < REC), "буфер записи мал"
    for r, k in zip(rec, n):
        tr = r[: k + 1]
        x, v = tr[:, :4], tr[:, 4:8]
        assert np.max(np.abs(np.linalg.norm(v, axis=1) - 1)) < 1e-14
        f = np.asarray(jax.vmap(sh.f_vec)(x))
        assert np.max(np.abs(f)) < 1e-12 * max(1.0, sh.scale)
        if sh.planes:
            Lm = np.asarray(moments(x, v, sh.planes))
            assert np.max(np.abs(Lm - Lm[0])) < 1e-6
        if sh.invariants is not None:
            I = np.asarray(sh.invariants(x, v))
            assert np.max(np.abs(I / I[0] - 1)) < 1e-6
