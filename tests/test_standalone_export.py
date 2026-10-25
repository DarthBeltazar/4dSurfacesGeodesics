"""DD-010, шаг 1: граф окна из C++ (`geo4d_gpu export-graph`) = граф `app.py --graph-only` для всех сцен scenes/.

Текст графов совпадает построчно (каталоги заменены, директива rebuild — перегенерация через Python — опущена);
это проверяет и заголовок pt_scene.slang, и дефайны, и push-константы: от них зависят ключи кэша SPIR-V в путях.
Массивы (буфер камеры, кубмапа, таблица фона) — побитово, кроме базиса камеры f, r, u: порядок суммирования
скалярных произведений в Грам–Шмидте (numpy против C++) и дополнение r из SVD (Python) против координатной оси (C++)
дают шум округления в почти нулевых компонентах (≤ 1e-15, знак нуля). Кроме сцен scenes/ — синтетические (on_M,
нестандартные параметры, NEE-допустимый и недопустимый излучатель, пробелы в путях).
На GPU: оба графа дают побитово один и тот же кадр (SPIR-V тот же — его скомпилировал прогон Python)."""
import json
import os
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


ENV = {k: v for k, v in os.environ.items() if k != "GEO4D_SLANG_OPT"}     # входит в ключ кэша только у Python

# синтетические сцены: on_M (проекция центров), нестандартные параметры (repr в MF_*), правило M34 на краях
W2 = 1.25
SYNTH = {
    "two_wh_custom": dict(
        manifold=dict(kind="two_compact_wh", params=dict(c1=[-4.5, 0.25, 0.0], c2=[4.0, 0.0, 0.5], b=1.3, rho0=2.75, W=W2)),
        env="procedural",
        objects=[dict(name="ball near throat", kind="ball4", center=[-4.2, 1.0, 0.3, 0.6], r=0.3, on_M=True,
                      material=dict(kind="metal", albedo=[0.9, 0.8, 0.7], rough=0.15)),
                 dict(name="torus", kind="torus", center=[3.0, 0.5, -0.5, 1.0], R=0.5, r=0.1, on_M=True),
                 dict(name="lamp ok", kind="ball4", center=[10.0, 0.0, 2.0, W2], r=0.4,
                      material=dict(kind="emit", emit=[5.0, 4.0, 3.0])),
                 dict(name="lamp in zone", kind="ball4", center=[-3.0, 0.0, 0.0, W2], r=0.4,
                      material=dict(kind="emit", emit=[1.0, 2.0, 3.0])),
                 dict(name="lamp off sheet", kind="ball4", center=[12.0, 0.0, 0.0, W2 + 1e-3], r=0.2,
                      material=dict(kind="emit", emit=[1.0, 1.0, 1.0])),
                 dict(name="wall", kind="plane", axis=1, offset=0.7, sign=-1.0)],
        camera=dict(x=[0.0, -11.0, 0.5, 1.2], f=[0.1, 1.0, -0.12, 0.0], up=[0.0, 0.0, 1.0, 0.0],
                    right=[1.0, 0.0, 0.0, 0.0], fov=63.5), speed=1.75),
    "tiger_custom": dict(
        manifold=dict(kind="tiger", params=dict(R1=1.1, R2=0.75, r=0.27)), env="procedural",
        objects=[dict(kind="ball4", center=[1.1, 0.0, 0.75, 0.3], r=0.05, on_M=True)],
        camera=dict(x=[1.37, 0.0, 0.0, 0.75], f=[0.0, 1.0, 0.0, 0.0], up=[0.0, 0.0, 1.0, 0.0], fov=80.0)),
    "ellis_custom": dict(
        manifold=dict(kind="ellis", params=dict(b=1.3)),
        objects=[dict(name="b", kind="ball4", center=[3.0, 0.0, 0.0, 0.0], r=0.5, on_M=True)],
        camera=dict(x=[0.0, -6.0, 0.0, 0.0], f=[0.0, 1.0, 0.0, 0.0], up=[0.0, 0.0, 1.0, 0.0])),
}


def _scene_file(tmp_path, scene):
    if scene in SYNTH:
        d = tmp_path / "scene dir"
        d.mkdir(exist_ok=True)
        f = d / f"{scene}.json"
        f.write_text(json.dumps(SYNTH[scene]), encoding="utf-8")
        return f
    return ROOT / "scenes" / f"{scene}.json"


def _export(tmp_path, scene, args):
    out = tmp_path / "out dir"
    py, cc = out / "py", out / "c"
    sf = _scene_file(tmp_path, scene)
    r = subprocess.run([sys.executable, "-X", "utf8", str(APP), "--scene", str(sf), "--graph-only", str(py)] + args,
                       capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([str(gpu.EXE), "export-graph", str(sf), str(cc)] + args,
                       capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stderr
    return py.resolve(), cc.resolve()


def _lines(d, name):
    t = (d / name).read_text(encoding="utf-8")
    for pre in (f'"{d}', f'"{d.as_posix()}', str(d), d.as_posix()):     # пути с пробелами — в кавычках
        t = t.replace(pre, '"<DIR>' if pre.startswith('"') else "<DIR>")
    return [line for line in t.splitlines() if not line.startswith("rebuild ")]


CASES = [(s, ["--res", "640x360"]) for s in SCENES + sorted(SYNTH)] + [("two_wormholes", ["--res", "1280x720", "--fov", "60", "--vsync",
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
@pytest.mark.parametrize("scene", ["two_wormholes", "S3", "ellis", "two_wh_custom"])
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
        r = subprocess.run([str(gpu.EXE), "app", str(g)], capture_output=True, text=True, encoding="utf-8", env=ENV)
        assert r.returncode == 0, r.stdout + r.stderr
        frames[d] = {p.name: np.load(p) for p in out.glob("Present_*.npy")}
    assert frames[py] and frames[py].keys() == frames[cc].keys()
    for name, a in frames[py].items():
        assert np.array_equal(a, frames[cc][name]), name
