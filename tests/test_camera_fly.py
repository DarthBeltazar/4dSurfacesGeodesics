"""M33: движение камеры по M — геодезическая с параллельным переносом базиса, поворот в T_xM."""
import numpy as np
import jax
import jax.numpy as jnp

from geo4d import shapes as S
from geo4d.camera import make_camera, fly


def _frame_ok(sh, cam, tol=1e-9):
    n = np.array(jax.grad(sh.f_vec)(jnp.asarray(cam.x)))
    n /= np.linalg.norm(n)
    B = np.stack([n, cam.f, cam.r, cam.u])
    return abs(float(sh.f_vec(jnp.asarray(cam.x)))) < tol and np.max(np.abs(B @ B.T - np.eye(4))) < tol


def test_s3_great_circle_returns():
    """S³ (R = 1): 2π вперёд по большому кругу — та же точка и тот же базис (перенос вдоль геодезической без голономии);
    промежуточная точка — точное решение M13."""
    sh = S.SHAPES["S3"]
    cam = make_camera(sh, np.array([1.0, 0, 0, 0]), [0, 1, 0, 0], [0, 0, 1, 0])
    c = cam
    for k in range(32):
        c = fly(sh, c, move=(2 * np.pi / 32, 0, 0))
        if k == 7:                                          # s = π/2
            xe, _ = sh.exact(jnp.asarray(cam.x), jnp.asarray(cam.f), np.pi / 2)
            assert np.linalg.norm(c.x - np.asarray(xe)) < 1e-6
        assert _frame_ok(sh, c)
    for a, b in [(c.x, cam.x), (c.f, cam.f), (c.r, cam.r), (c.u, cam.u)]:
        assert np.linalg.norm(a - b) < 1e-6


def test_s3_strafe_transports_frame():
    """Движение вбок (вдоль r): f переносится параллельно — на S³ вдоль большого круга f постоянен (⊥ плоскости круга)."""
    sh = S.SHAPES["S3"]
    cam = make_camera(sh, np.array([1.0, 0, 0, 0]), [0, 0, 0, 1], [0, 0, 1, 0])
    c = fly(sh, cam, move=(0, 1.0, 0))
    assert np.linalg.norm(c.f - cam.f) < 1e-8 and _frame_ok(sh, c)


def test_hyperplane_exact():
    sh = S.SHAPES["hyperplane"]
    cam = make_camera(sh, np.array([0.3, -1.0, 0.2, 0]), [0, 1, 0, 0], [0, 0, 1, 0])
    c = fly(sh, cam, move=(1.5, -0.4, 0.7), rot=(0, 0, 0))
    assert np.linalg.norm(c.x - (cam.x + 1.5 * cam.f - 0.4 * cam.r + 0.7 * cam.u)) < 1e-12


def test_rotation_in_tangent_plane():
    sh = S.SHAPES["ellipsoid"]
    P = np.asarray(S.sample(sh, 1, seed=4)[0])[0]
    cam = make_camera(sh, P, [1, 0.3, 0, 0], [0, 0, 1, 0])
    c = fly(sh, cam, rot=(0.3, -0.2, 0.1))
    assert _frame_ok(sh, c) and np.allclose(c.x, cam.x)


def test_compact_wormhole_crossing():
    """Радиальный полёт сквозь горловину M11: лист меняется (w → −W), луч остаётся на оси x, базис ортонормирован.
    Длина пути симметрична: от ρ = 5 на верхнем листе до ρ = 5 на нижнем."""
    sh = S.SHAPES["compact_wh"]
    W = sh.params["W"]
    cam = make_camera(sh, np.array([5.0, 0, 0, W]), [-1, 0, 0, 0], [0, 0, 1, 0])
    c = cam
    from geo4d.reference import compact_profile
    h, dh = compact_profile(1.0, 3.0, W)
    from scipy.integrate import quad
    # меридиан от горловины до ρ₀; у горловины подынтегральное ~ 1/√(ρ − b) — замена ρ = b + σ²
    ell0 = quad(lambda q: 2 * q * np.sqrt(1 + dh(1 + q * q) ** 2 / (4 * h(1 + q * q))), 0, np.sqrt(2.0),
                epsabs=1e-12, limit=500)[0]
    L = 2 * (5.0 - 3.0) + 2 * ell0
    for _ in range(40):
        c = fly(sh, c, move=(L / 40, 0, 0), substeps=32)
        assert _frame_ok(sh, c, 1e-8)
    assert c.x[3] < 0 and abs(c.x[1]) < 1e-9 and abs(c.x[2]) < 1e-9
    assert abs(np.linalg.norm(c.x[:3]) - 5.0) < 1e-5 and abs(c.x[3] + W) < 1e-6
    assert np.linalg.norm(c.f - np.array([1.0, 0, 0, 0])) < 1e-6           # летит дальше «наружу» по x
