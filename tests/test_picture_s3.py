"""Тест картинкой S³ (M21): угловой радиус маркера бисекцией по углу луча, заднее изображение, предел d → πR."""
import numpy as np
import pytest

from geo4d.shapes import SHAPES
from geo4d.integrators import Config
from geo4d.markers import ball_markers, trace_markers

R = 1.0
X0 = np.array([0.0, 0.0, 0.0, R])
F = np.array([1.0, 0.0, 0.0, 0.0])
RT = np.array([0.0, 1.0, 0.0, 0.0])
CFG = Config(tol=1e-11, s_max=2 * np.pi * R - 1e-3, n_max=200_000)
EPS_HIT = 1e-11


def marker(d, a):
    m = np.cos(d / R) * X0 + R * np.sin(d / R) * F
    return ball_markers([m], [2 * R * np.sin(a / (2 * R))])


def bisect_alpha(g, sign, iters=36):
    """Граница попадания по углу α от оси ±forward (вектор sign·(cos α f + sin α r))."""
    sh = SHAPES["S3"]
    lo, hi = 0.0, np.pi / 2
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = sign * (np.cos(mid) * F + np.sin(mid) * RT)
        ids, s, st, _ = trace_markers(sh, X0[None], v[None], g, CFG, eps_hit=EPS_HIT)
        lo, hi = (mid, hi) if ids[0] >= 0 else (lo, mid)
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("d,a", [(1.0, 0.3), (2.0, 0.3), (2.8, 0.2)])
def test_angular_radius_front_and_back(d, a):
    g = marker(d, a)
    expect = np.arcsin(np.sin(a / R) / np.sin(d / R))
    front = bisect_alpha(g, +1)
    back = bisect_alpha(g, -1)
    assert abs(front - expect) < 1e-6, (front, expect)
    assert abs(back - expect) < 1e-6, (back, expect)       # второе изображение сзади того же размера


def test_path_lengths_front_back():
    """Попадание спереди — на длине пути ≈ d − a, сзади — ≈ 2πR − d − a."""
    d, a = 1.5, 0.25
    g = marker(d, a)
    sh = SHAPES["S3"]
    ids, s, _, _ = trace_markers(sh, np.stack([X0, X0]), np.stack([F, -F]), g, CFG, eps_hit=EPS_HIT)
    assert np.all(ids == 0)
    assert abs(s[0] - (d - a)) < 1e-8 and abs(s[1] - (2 * np.pi * R - d - a)) < 1e-8


def test_fills_view_near_antipode():
    """d → πR: маркер виден во всех направлениях."""
    a = 0.3
    g = marker(np.pi * R - a / 2, a)
    rng = np.random.default_rng(0)
    V = rng.normal(size=(64, 4))
    V[:, 3] = 0.0                                   # касательная плоскость в X0 = e_w
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ids, _, st, _ = trace_markers(SHAPES["S3"], np.broadcast_to(X0, V.shape), V, g, CFG, eps_hit=1e-9)
    assert np.all(ids == 0)
