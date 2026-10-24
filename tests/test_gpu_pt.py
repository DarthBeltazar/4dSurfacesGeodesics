"""Режим B на GPU (DD-005): мегаядро pathtrace.slang — печь, согласованность MIS, сравнение с прототипом (float64)."""
import numpy as np
import pytest

from geo4d import gpu, pathtrace as PT, gpu_pt as G
from geo4d.camera import make_camera
from geo4d.wormholes import ellis_point

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]

TOL = dict(tol_kappa=1e-3, tol_max=1e-4)


def _setup(name, n, furnace=False):
    up = np.array([0, 0, 1.0, 0])
    if name == "hyperplane":
        sc = PT.scene_hyperplane(furnace=furnace)
        return sc, make_camera(sc.shape, np.array([0, -2.0, 0.5, 0]), np.array([0, 1.0, -0.1, 0]), up, fov=70, width=n, height=n)
    if name == "two_wormholes":
        sc = PT.scene_two_wormholes(furnace=furnace)
        return sc, make_camera(sc.shape, np.array([0, -10.0, -1.0, 1.5]), np.array([0, 1.0, 0.05, 0]), up, fov=80,
                               width=n, height=n)
    sc = PT.scene_ellis(furnace=furnace)
    x, e = ellis_point(6.0)
    return sc, make_camera(sc.shape, x, -e, up, fov=60, width=n, height=n)


def _img_z(a, va, b, vb):
    return (a.mean() - b.mean()) / np.sqrt((va + vb).sum() / a.size ** 2)


@pytest.mark.parametrize("name", ["hyperplane", "two_wormholes", "ellis"])
def test_gpu_furnace(name):
    sc, cam = _setup(name, 8, furnace=True)
    m, var, st, _ = G.render(sc, cam, 2048, G.GPUPT(max_depth=64, **TOL))
    z = np.where(var > 0, (m - 1) / np.sqrt(np.maximum(var, 1e-30)), 0)
    assert abs((m.mean() - 1) / np.sqrt(var.sum() / m.size ** 2 + 1e-30)) < 4
    assert np.abs(z).max() < 5
    assert (st[:, 1] // 1e6).sum() < 1e-3 * 2048 * m.shape[0]   # попадания с обратной стороны (скользящие) — редки


def test_gpu_mis_consistent():
    """Сцена 3: степенная эвристика, «только NEE» и «только BSDF» сходятся к одному среднему."""
    sc, cam = _setup("two_wormholes", 6)
    res = [G.render(sc, cam, 8192, G.GPUPT(seed=s, **TOL, **kw))[:2]
           for s, kw in [(1, {}), (2, dict(mis=1)), (3, dict(nee=0))]]
    for (a, va), (b, vb) in [(res[0], res[1]), (res[0], res[2])]:
        assert abs(_img_z(a, va, b, vb)) < 4
        assert np.mean(np.abs(a - b) > 4 * np.sqrt(va + vb)) <= 3 / a.size


@pytest.mark.slow
@pytest.mark.parametrize("name,spp", [("hyperplane", 256), ("two_wormholes", 256), ("ellis", 512)])
def test_gpu_vs_prototype(name, spp):
    """GPU (float32) против прототипа (float64) при tol 1e-4: средние по пикселям совпадают в пределах статистики.
    Попиксельно допускается до 5 % выбросов > 4σ: у прототипа при сотнях spp тяжёлые хвосты (малые источники,
    у Эллиса — только BSDF), и выборочная дисперсия недооценена."""
    sc, cam = _setup(name, 6)
    a, va, _ = PT.render(sc, cam, PT.PTConfig(spp=spp), seed=7)
    b, vb, _, _ = G.render(sc, cam, 32768, G.GPUPT(**TOL))
    assert abs(_img_z(a, va, b, vb)) < 4
    assert np.mean(np.abs(a - b) > 4 * np.sqrt(va + vb)) <= 0.05


def test_camera_pass_classes():
    """Проход камеры: класс «объект» ⇔ центральный луч попал (сверка с альбедо AOV трассировщика: у ушедших в фон
    первичных лучей альбедо = цвет фона, у попавших — альбедо материала); фон режима A близок к трассировщику."""
    from geo4d.modea import psnr
    sc = PT.scene_two_wormholes()
    cam = make_camera(sc.shape, np.array([0, -11.0, 0.5, 1.5]), np.array([0, 1.0, -0.12, 0]), np.array([0, 0, 1.0, 0]),
                      fov=75, width=96, height=54)
    fr = G.render_frame(sc, cam, 64, nfb=256)
    cls = fr["cls"]
    assert set(np.unique(cls)) <= {0, 1, 2} and (cls == 0).mean() > 0.2 and (cls == 1).mean() > 0.3
    ref, _, _, _ = G.render(sc, cam, 4096)
    bg = cls == 0
    assert psnr(fr["color"][bg], ref[bg]) > 34          # фон режима A против трассировки; ошибка — фильтр фона (ADR-0005)


def test_denoise_improves():
    sc, cam = _setup("two_wormholes", 64)
    ref, _, _, _ = G.render(sc, cam, 4096)
    m, _, _, _ = G.render(sc, cam, 4, G.GPUPT(seed=5))
    alb, nrm = G.render.aov
    d, rep = gpu.denoise(m.reshape(64, 64, 3), alb.reshape(64, 64, 3), nrm.reshape(64, 64, 3))
    from geo4d.modea import psnr
    assert psnr(d.reshape(-1, 3), ref) > psnr(m, ref) + 5
