"""DD-010, шаг 1: граф окна из C++ (`geo4d_gpu export-graph`) = граф `app.py --graph-only` для всех сцен scenes/.

Текст графов совпадает построчно (каталоги заменены, директива rebuild — перегенерация через Python — опущена);
это проверяет и заголовок pt_scene.slang, и дефайны, и push-константы: от них зависят ключи кэша SPIR-V в путях.
Массивы (буфер камеры, кубмапа, таблица фона) — побитово, кроме базиса камеры без подсказки right: Python берёт
дополнение из SVD, C++ — из координатной оси, и нулевые компоненты расходятся в шуме округления (≤ 1e-15, ±0).
На GPU: оба графа дают побитово один и тот же кадр (SPIR-V тот же — его скомпилировал прогон Python)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "prototype" / "scripts" / "app.py"
SCENES = sorted(p.stem for p in (ROOT / "scenes").glob("*.json"))

pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")


def _export(tmp_path, scene, args):
    py, cc = tmp_path / "py", tmp_path / "c"
    r = subprocess.run([sys.executable, "-X", "utf8", str(APP), "--scene", scene, "--graph-only", str(py)] + args,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    r = subprocess.run([str(gpu.EXE), "export-graph", str(ROOT / "scenes" / f"{scene}.json"), str(cc)] + args,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return py.resolve(), cc.resolve()


def _lines(d, name):
    t = (d / name).read_text(encoding="utf-8")
    for pre in (str(d), d.as_posix()):
        t = t.replace(pre, "<DIR>")
    return [line for line in t.splitlines() if not line.startswith("rebuild ")]


CASES = [(s, ["--res", "640x360"]) for s in SCENES] + [("two_wormholes", ["--res", "1280x720", "--fov", "60", "--vsync",
                                                                          "--fullscreen", "--cube", "256"])]


@pytest.mark.parametrize("scene,args", CASES, ids=[f"{s}-{'-'.join(a)}" for s, a in CASES])
def test_export_matches_python(tmp_path, scene, args):
    py, cc = _export(tmp_path, scene, args)
    for name in ("app_graph.txt", "app_graph_dyn.txt"):
        assert _lines(cc, name) == _lines(py, name), name
    files = sorted(p.name for p in py.glob("*.npy"))
    assert files == sorted(p.name for p in cc.glob("*.npy"))
    for name in files:
        x, y = np.load(py / name), np.load(cc / name)
        assert x.shape == y.shape and x.dtype == y.dtype, name
        if name.startswith("cam"):
            # строки 1–3 — базис камеры f, r, u (M16): шум округления в нулевых компонентах
            np.testing.assert_allclose(y[1:4], x[1:4], rtol=0, atol=1e-15, err_msg=name)
            x, y = np.delete(x, [1, 2, 3], 0), np.delete(y, [1, 2, 3], 0)
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), name


@pytest.mark.gpu
@pytest.mark.parametrize("scene", ["two_wormholes", "S3", "ellis"])
def test_export_renders_same(tmp_path, scene):
    py, cc = _export(tmp_path, scene, ["--res", "320x180"])
    np.savetxt(tmp_path / "script.txt", np.zeros((3, 6)), fmt="%.6f")
    frames = {}
    for d in (py, cc):
        out = d / "dump"
        out.mkdir()
        g = d / "app_graph.txt"
        g.write_text(g.read_text(encoding="utf-8") + f"hidden\nbench {gpu.qpath(tmp_path / 'script.txt')} "
                     f"{gpu.qpath(d / 'frames.json')}\ndump 2 {gpu.qpath(out)}\n", encoding="utf-8")
        r = subprocess.run([str(gpu.EXE), "app", str(g)], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stdout + r.stderr
        frames[d] = {p.name: np.load(p) for p in out.glob("Present_*.npy")}
    assert frames[py] and frames[py].keys() == frames[cc].keys()
    for name, a in frames[py].items():
        assert np.array_equal(a, frames[cc][name]), name
