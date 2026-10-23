"""Тест картинкой S¹×R² (M22): рендер геодезическим трассировщиком против независимого евклидова
трассировщика по прямым в развёртке (σ, z, w) со сценой, размноженной с периодом 2πR."""
import numpy as np

from geo4d.shapes import SHAPES
from geo4d.integrators import Config
from geo4d.camera import make_camera, pixel_uv, pixel_dirs
from geo4d.markers import cylinder_intrinsic_markers, trace_markers

R = 1.0
BALLS = np.array([[2.0, 0.3, 0.1, 0.4], [4.5, -0.5, 0.4, 0.5], [1.0, 2.5, -0.3, 0.6],
                  [-2.5, 0.8, 0.6, 0.35], [3.2, -2.0, -0.8, 0.5], [5.5, 1.2, -1.0, 0.9],
                  [0.8, -1.5, 1.4, 0.8], [2.6, 1.8, 2.0, 0.9], [4.0, 4.0, 1.5, 1.0]])    # σ, z, w, радиус
S_MAX = 25.0


def euclid_trace(p0, dirs, balls, s_max, K=6):
    """Ближайшее пересечение прямых с шарами и их копиями σ + 2πRk. Возвращает id (−1), t, зазор силуэта."""
    n = len(dirs)
    best_t = np.full(n, np.inf)
    best_id = np.full(n, -1)
    margin = np.full(n, np.inf)   # min |расстояние наибольшего сближения − радиус| — близость к силуэту
    for j, (sg, z, w, r) in enumerate(balls):
        for k in range(-K, K + 1):
            c = np.array([sg + 2 * np.pi * R * k, z, w])
            oc = p0 - c
            b = dirs @ oc
            cc = oc @ oc - r * r
            disc = b * b - cc
            tmin = -b
            dmin = np.sqrt(np.maximum(oc @ oc - b * b, 0.0))
            near = (tmin > 0) & (tmin < s_max)
            margin = np.where(near, np.minimum(margin, np.abs(dmin - r)), margin)
            t = -b - np.sqrt(np.maximum(disc, 0.0))
            hit = (disc > 0) & (t > 0) & (t < s_max) & (t < best_t)
            best_t = np.where(hit, t, best_t)
            best_id = np.where(hit, j, best_id)
    return best_id, best_t, margin


def test_matches_euclidean_tracer():
    sh = SHAPES["S1xR2"]
    x0 = np.array([R, 0.0, 0.0, 0.0])
    fwd = np.array([0.0, np.cos(0.3), np.sin(0.3), 0.05])
    cam = make_camera(sh, x0, fwd, [0, 0, 0, 1.0], fov=90.0, width=48, height=48)
    U, V = pixel_uv(cam)
    D = pixel_dirs(cam, U, V).reshape(-1, 4)
    # геодезический трассировщик
    g = cylinder_intrinsic_markers(R, BALLS[:, :3], BALLS[:, 3])
    ids, s, st, steps = trace_markers(sh, np.broadcast_to(x0, D.shape), D, g,
                                      Config(tol=1e-10, s_max=S_MAX, n_max=50_000), eps_hit=1e-9)
    assert np.all(st != 3)
    # евклидов: в x0 касательные e_σ = (0,1,0,0), e_z, e_w
    Dsz = np.stack([D[:, 1], D[:, 2], D[:, 3]], 1)
    eid, et, margin = euclid_trace(np.zeros(3), Dsz, BALLS, S_MAX)
    silhouette = margin < 1e-6
    assert silhouette.mean() < 0.01
    ok = ~silhouette
    assert np.array_equal(ids[ok], eid[ok]), np.sum(ids[ok] != eid[ok])
    hit = ok & (eid >= 0)
    assert hit.sum() > 0.2 * len(D)
    assert np.max(np.abs(s[hit] - et[hit])) < 1e-6
