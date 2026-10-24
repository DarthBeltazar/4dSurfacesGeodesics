"""Режим B, прототип (DD-005): BSDF и выборка фона, печь, аналитика прямого освещения, MIS против «только NEE»,
сравнение с независимым евклидовым трассировщиком на гиперплоскости."""
import numpy as np
import jax.numpy as jnp
import pytest

from geo4d import pathtrace as PT, euclid_pt as E, shapes as S
from geo4d.objects import ball4, halfspace, Material
from geo4d.wormholes import compact_exit, ellis_point
from geo4d.camera import make_camera, pixel_uv


def _img_z(a, va, b, vb):
    return (a.mean() - b.mean()) / np.sqrt((va + vb).sum() / a.size ** 2)


@pytest.mark.parametrize("kind,a", [(0, 0.3), (1, 0.1), (1, 0.3), (1, 0.7)])   # α < 0.1: равномерная оценка ∫ слишком шумная
def test_bsdf_sampling_consistent(kind, a):
    """E[f·cos/pdf] по сэмплам BSDF = ∫ f·cos dω (равномерная полусфера); pdf ≤ 1 (выборка ниже горизонта — 0)."""
    rng = np.random.default_rng(0)
    N = 400_000
    for th in [0.2, 1.0, 1.45]:
        wo = jnp.asarray(np.tile([np.sin(th), 0, np.cos(th)], (N, 1)))
        K, A, R = jnp.full(N, kind), jnp.asarray(np.tile([0.9, 0.5, 0.2], (N, 1))), jnp.full(N, a)
        z = rng.random(N)
        ph = 2 * np.pi * rng.random(N)
        wi = jnp.asarray(np.c_[np.sqrt(1 - z * z) * np.cos(ph), np.sqrt(1 - z * z) * np.sin(ph), z])
        f, p = PT.bsdf_eval(K, A, R, wo, wi)
        int_f = np.mean(np.asarray(f), 0) * 2 * np.pi
        assert np.mean(np.asarray(p)) * 2 * np.pi < 1.01
        _, w, _ = PT.bsdf_sample(K, A, R, wo, jnp.asarray(rng.random(N)), jnp.asarray(rng.random(N)))
        assert np.allclose(np.mean(np.asarray(w), 0), int_f, rtol=0.03, atol=2e-3)


def test_env_pdf():
    """M29: pdf выборки фона интегрируется в 1 и согласована с сэмплами: E[h(d)/pdf(d)] = ∫ h dω."""
    mips = PT.procedural_env(32)
    tab = PT.env_table(mips)
    rng = np.random.default_rng(1)
    N = 400_000
    for sheet in [0, 1]:
        s = np.full(N, sheet)
        d = rng.normal(size=(N, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        assert abs(np.mean(PT.env_pdf(tab, s, d)) * 4 * np.pi - 1) < 0.02
        h = lambda q: 1 + q[:, 2] ** 2 + 2 * np.maximum(q[:, 0], 0) ** 4
        exact = np.mean(h(d)) * 4 * np.pi
        ds = PT.env_sample(tab, s, rng)
        est = np.mean(h(ds) / PT.env_pdf(tab, s, ds))
        assert abs(est / exact - 1) < 0.02


def _cam(sc, n, x, fwd, fov=70):
    return make_camera(sc.shape, np.asarray(x, float), np.asarray(fwd, float), np.array([0, 0, 1.0, 0]), fov=fov,
                       width=n, height=n)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["hyperplane", "two_wormholes"])
def test_furnace(name):
    """Печь: всё диффузное с альбедо 1, фон ≡ 1 ⇒ каждый пиксель 1 (энергия сохраняется и сквозь кривизну)."""
    if name == "hyperplane":
        sc = PT.scene_hyperplane(furnace=True)
        cam = _cam(sc, 8, [0, -2.0, 0.5, 0], [0, 1.0, -0.1, 0])
    else:
        sc = PT.scene_two_wormholes(furnace=True)
        cam = _cam(sc, 8, [0, -10.0, -1.0, 1.5], [0, 1.0, 0.05, 0], fov=80)
    m, var, st = PT.render(sc, cam, PT.PTConfig(spp=8, max_depth=64), seed=1)
    z = np.where(var > 0, (m - 1) / np.sqrt(np.maximum(var, 1e-300)), 0)
    assert np.abs(z).max() < 5
    assert np.abs(np.where(var == 0, m - 1, 0)).max() < 1e-9
    assert st["capped"] < 2e-2      # пологие лучи к бесконечному полу дальше s_max (теряется энергия → z проверяет и это)


def test_furnace_ellis():
    """Эллис (без плоских зон, только BSDF): при альбедо 1 и косинусной выборке каждый сэмпл ровно 1 — пути не теряются."""
    sc = PT.scene_ellis(furnace=True)
    x, e = ellis_point(6.0)
    cam = make_camera(sc.shape, x, -e, np.array([0, 0, 1.0, 0]), fov=60, width=6, height=6)
    m, var, st = PT.render(sc, cam, PT.PTConfig(spp=4, max_depth=64), seed=1)
    assert np.abs(m - 1).max() < 1e-9 and st["capped"] == 0


@pytest.mark.slow
def test_direct_lighting_analytic():
    """Пол (альбедо ρ) под сферическим источником, фон 0: яркость ρ L sin²α cos θ (переотражений нет).
    Аналитика усредняется по тому же гауссову фильтру пикселя. Режимы: MIS, только NEE, только BSDF."""
    c, r, Le, rho, n = np.array([-1.5, 2.0, 2.0]), 1.0, 4.0, 0.5, 6
    objs = [halfspace(2, -1.0, Material("diffuse", (rho,) * 3)), ball4((*c, 0), r, Material("emit", emit=(Le,) * 3))]
    sc = PT.Scene(S.SHAPES["hyperplane"], objs, PT.constant_env(1.0), compact_exit, PT.FlatZones((0.0,)), lights=(1,),
                  env_scale=0.0)
    cam = _cam(sc, n, [0, -2.0, 0.5, 0], [0, 1.0, -0.35, 0], fov=60)
    U, V = (q.ravel() for q in pixel_uv(cam))
    dpix = 2 * cam.tf / cam.width
    rng = np.random.default_rng(9)
    exact = np.zeros(n * n)
    for i in range(n * n):
        du, dv = rng.normal(0, 0.5, (2, 20000))
        D = cam.f + (U[i] + du * dpix)[:, None] * cam.r + (V[i] + dv * dpix)[:, None] * cam.u
        D = D[:, :3] / np.linalg.norm(D[:, :3], axis=1, keepdims=True)
        o = cam.x[:3]
        oc = o - c
        bq = D @ oc
        disc = bq ** 2 - (oc @ oc - r * r)
        tl = np.where(disc > 0, -bq - np.sqrt(np.maximum(disc, 0)), np.inf)
        tl = np.where(tl > 0, tl, np.inf)
        tf = np.where(D[:, 2] < 0, (-1.0 - o[2]) / np.where(D[:, 2] < 0, D[:, 2], -1), np.inf)
        P = o + np.where(np.isfinite(tf), tf, 0)[:, None] * D
        dc = c - P
        dist = np.linalg.norm(dc, axis=1)
        exact[i] = np.where(tl < tf, Le, np.where(np.isfinite(tf), rho * Le * (r / dist) ** 2 * dc[:, 2] / dist, 0)).mean()
    for mode, nee in [("power", True), ("nee", True), ("power", False)]:
        m, var, _ = PT.render(sc, cam, PT.PTConfig(spp=256, mis=mode, nee=nee), seed=5)
        g, vg = m[:, 0], var[:, 0]
        assert abs(_img_z(g, vg, exact, 0 * vg)) < 4, mode
        if nee:      # у «только BSDF» малый источник даёт тяжёлый хвост — попиксельный z ненадёжен
            assert np.mean(np.abs(g - exact) > 4 * np.sqrt(vg) + 1e-9) <= 2 / n ** 2, mode


@pytest.mark.slow
def test_mis_matches_nee_only_two_wormholes():
    """Сцена 3: MIS и «вся прямая освещённость через NEE» (веса 1/0) сходятся к одному среднему — веса согласованы
    с областью прямых соединений (M29), включая листы и шары кривизны."""
    sc = PT.scene_two_wormholes()
    cam = _cam(sc, 6, [0, -10.0, -1.0, 1.5], [0, 1.0, 0.05, 0], fov=80)
    a, va, _ = PT.render(sc, cam, PT.PTConfig(spp=128, mis="power"), seed=3)
    b, vb, _ = PT.render(sc, cam, PT.PTConfig(spp=128, mis="nee"), seed=4)
    assert abs(_img_z(a, va, b, vb)) < 4
    assert np.mean(np.abs(a - b) > 4 * np.sqrt(va + vb) + 1e-9) <= 3 / a.size


@pytest.mark.slow
def test_hyperplane_matches_euclidean():
    """M — гиперплоскость: трассировщик на M против независимого евклидова (аналитические пересечения, без кода M)."""
    sc = PT.scene_hyperplane()
    cam = _cam(sc, 6, [0, -2.0, 0.5, 0], [0, 1.0, -0.1, 0])
    a, va, _ = PT.render(sc, cam, PT.PTConfig(spp=256), seed=2)
    b, vb = E.render(sc, cam, PT.PTConfig(spp=1024), seed=11)
    assert abs(_img_z(a, va, b, vb)) < 4
    assert np.mean(np.abs(a - b) > 4 * np.sqrt(va + vb) + 1e-9) <= 3 / a.size
