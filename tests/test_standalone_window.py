"""DD-010, шаг 3: окно без Python (`geo4d_gpu [сцена.json] …`): граф в памяти, шейдеры в процессе, перегенерации
редактора — функцией C++ в потоке. Сценарий — директивы из --extra (как app.py дописывает их к своему графу).

- холодный старт = окно на графе Python (`app.py --graph-only` + `geo4d_gpu app`) побитово;
- смена разрешения (resize) = холодный старт на новом разрешении побитово;
- режим редактирования со структурной правкой и выходом (сборка быстрых шейдеров в процессе) = холодный старт
  отредактированной сцены побитово;
- сцена с другой топологией (loadscene) и выход — сборка в процессе, без Python;
- пути с кириллицей и пробелами (манифест UTF-8)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "prototype" / "scripts" / "app.py"
ENV = {k: v for k, v in os.environ.items() if k != "GEO4D_SLANG_OPT"}

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]


def _script(td, frames, dump=(), extra=""):
    td.mkdir(parents=True, exist_ok=True)
    np.savetxt(td / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    lines = ["hidden", f"bench {gpu.qpath(td / 'script.txt')} {gpu.qpath(td / 'f.json')}"]
    lines += [f"dump {k} {gpu.qpath(td)}" for k in dump]
    return "\n".join(lines) + "\n" + extra


def _standalone(td, scene, frames, dump=(), extra="", res="96x54", opts=()):
    """Окно без Python; возвращает stdout (rows_check, mf_check, standalone_build)."""
    (td / "extra.txt").parent.mkdir(parents=True, exist_ok=True)
    (td / "extra.txt").write_text(_script(td, frames, dump, extra), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), str(scene), "--res", res, "--cube", "64", "--work-dir", str(td / "work"),
                        "--extra", str(td / "extra.txt"), *opts], capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("rows_check")]
    assert rows and max(rows) < 1e-6, rows
    return r.stdout


def _dumps(td, k):
    return {n: np.load(td / f"{n}_{k}.npy") for n in ("Frame", "Present")}


def _same(a, b):
    for n in a:
        assert a[n].shape == b[n].shape and np.array_equal(a[n], b[n]), n


def test_standalone_equals_python_app(tmp_path):
    scene = ROOT / "scenes" / "two_wormholes.json"
    py = tmp_path / "py"
    r = subprocess.run([sys.executable, "-X", "utf8", str(APP), "--scene", str(scene), "--res", "96x54", "--cube", "64",
                        "--graph-only", str(py), "--python-graph"], capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stderr
    g = py / "app_graph.txt"
    g.write_text(g.read_text(encoding="utf-8") + _script(py, 8, [7]), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(g)], capture_output=True, text=True, encoding="utf-8", env=ENV)
    assert r.returncode == 0, r.stdout + r.stderr
    _standalone(tmp_path / "c", scene, 8, [7], opts=["--bake-params"])     # числа литералами — тот же текст шейдеров
    _same(_dumps(py, 7), _dumps(tmp_path / "c", 7))


def test_standalone_resize_equals_cold_start(tmp_path):
    scene = ROOT / "scenes" / "two_wormholes.json"
    out = _standalone(tmp_path / "hot", scene, 16, [15], "resize 5 128 72\n")
    assert out.count("standalone_build") == 2                 # старт и смена разрешения — в процессе
    _standalone(tmp_path / "cold", scene, 10, [9], res="128x72")
    hot, cold = _dumps(tmp_path / "hot", 15), _dumps(tmp_path / "cold", 9)
    assert hot["Frame"].shape[:2] == (72, 128)
    _same(hot, cold)
    assert np.load(tmp_path / "hot" / "swapchain_15.npy").shape == (54, 96)
    log = json.loads((tmp_path / "hot" / "f.json").read_text(encoding="utf-8"))
    assert log[6]["moving"] and log[-1]["accum"] == 10




def test_standalone_cyrillic_paths(tmp_path):
    td = tmp_path / "Документы пользователя"
    scene = td / "моя сцена.json"
    td.mkdir()
    scene.write_text((ROOT / "scenes" / "ellis.json").read_text(encoding="utf-8"), encoding="utf-8")
    out = _standalone(td / "запуск", scene, 12, [11], "resize 4 64 36\n")
    assert out.count("standalone_build") == 2
    assert np.load(td / "запуск" / "Present_11.npy").shape == (36, 64)


def test_installed_layout(tmp_path):
    """Установленная программа (zip, DD-010/DD-012): exe и DLL, shaders/ и scenes/ рядом; кэш SPIR-V, заголовков и
    конвейеров — в пользовательском кэше (GEO4D_CACHE_DIR); каталог программы не меняется. Без сцены в аргументах —
    scenes/two_wormholes.json рядом с exe."""
    import shutil
    inst = tmp_path / "Программа geo4d"
    inst.mkdir()
    for f in gpu.EXE.parent.glob("*.dll"):
        shutil.copy2(f, inst)
    exe = inst / gpu.EXE.name
    shutil.copy2(gpu.EXE, exe)
    shutil.copytree(ROOT / "gpu" / "shaders", inst / "shaders")
    shutil.copytree(ROOT / "scenes", inst / "scenes")
    shutil.copytree(ROOT / "nodes", inst / "nodes")                 # библиотека нод (DD-011)
    before = sorted(p.relative_to(inst) for p in inst.rglob("*"))
    td = tmp_path / "run"
    td.mkdir()
    (td / "extra.txt").write_text(_script(td, 4, [3]), encoding="utf-8")
    cache = tmp_path / "кэш"
    r = subprocess.run([str(exe), "--res", "96x54", "--cube", "64", "--work-dir", str(td / "work"), "--extra",
                        str(td / "extra.txt")], capture_output=True, text=True, encoding="utf-8",
                       env=dict(ENV, GEO4D_CACHE_DIR=str(cache)), cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(p.relative_to(inst) for p in inst.rglob("*")) == before
    assert len(list((cache / "spv").glob("*.spv"))) == 27 and list((cache / "gen").rglob("pt_scene.slang"))
    assert (cache / "pipelines.bin").exists()
    assert not list((td / "work").glob("edit_*.json"))          # перегенераций не было
    assert (np.load(td / "Present_3.npy") & 0xFFFFFF).std() > 0


def test_standalone_graph_edits_equal_cold_start(tmp_path):
    """Правки графа в окне (директивы холста редактора нод, DD-011 §8): сдвиг шара, шероховатость металла, цвет
    источника — сразу (строки), новый источник из нод — сборка быстрых шейдеров в процессе; итог = холодный старт
    отредактированной сцены побитово (раньше — режим редактирования DD-008 и выход из него)."""
    d = json.loads((ROOT / "scenes" / "two_wormholes.json").read_text(encoding="utf-8"))
    c1 = list(d["objects"][1]["center"])
    c1[0] += 1.0
    d["objects"][1]["center"] = c1
    d["objects"][2]["material"]["rough"] = 0.6
    d["objects"][5]["material"]["emit"] = [20.0, 40.0, 30.0]
    d["objects"].append(dict(name="new_light", kind="ball4", center=[0.0, 3.0, 1.0, 1.5], r=0.3,
                             material=dict(kind="emit", albedo=[0.8, 0.8, 0.8], rough=0.3, emit=[5.0, 20.0, 5.0])))
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(d), encoding="utf-8")
    ops = (f"setparam 5 shape_1 c {' '.join(map(str, c1))}\nsetparam 5 mat_2 rough 0.6\nsetparam 5 mat_5 emit 20 40 30\n"
           "addnode 5 nl obj.ball4\naddnode 5 nlm mat.emit\naddnode 5 nlo out.object\n"
           "setparam 5 nl c 0 3 1 1.5\nsetparam 5 nl r 0.3\nsetparam 5 nlm albedo 0.8 0.8 0.8\nsetparam 5 nlm rough 0.3\n"
           "setparam 5 nlm emit 5 20 5\n"
           "link 6 nl.out nlo.shape\nlink 6 nlm.out nlo.material\nlink 6 nlo.out scene.objects\n")
    out = _standalone(tmp_path / "hot", ROOT / "scenes" / "two_wormholes.json", 14, [13], ops)
    builds = [json.loads(t.split(" ", 1)[1]) for t in out.splitlines() if t.startswith("standalone_build")]
    assert len(builds) == 2                                   # новый источник — новая структура: сборка в процессе
    _standalone(tmp_path / "cold", edited, 7, [6])            # кадр 7 (после сборки в кадре 6) = кадр 0
    _same(_dumps(tmp_path / "hot", 13), _dumps(tmp_path / "cold", 6))
