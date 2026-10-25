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


def _standalone(td, scene, frames, dump=(), extra="", res="96x54"):
    """Окно без Python; возвращает stdout (rows_check, mf_check, standalone_build)."""
    (td / "extra.txt").parent.mkdir(parents=True, exist_ok=True)
    (td / "extra.txt").write_text(_script(td, frames, dump, extra), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), str(scene), "--res", res, "--cube", "64", "--work-dir", str(td / "work"),
                        "--extra", str(td / "extra.txt")], capture_output=True, text=True, encoding="utf-8", env=ENV)
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
    _standalone(tmp_path / "c", scene, 8, [7])
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


def test_standalone_structural_cycle_equals_cold_start(tmp_path):
    """Режим редактирования, правка со сменой структуры (новый источник), выход — быстрые шейдеры собираются в
    процессе; итог = холодный старт отредактированной сцены (как test_app_edit_mode_structural_cycle для Python)."""
    d = json.loads((ROOT / "scenes" / "two_wormholes.json").read_text(encoding="utf-8"))
    d["objects"][1]["center"][0] += 1.0
    d["objects"][2]["material"]["rough"] = 0.6
    d["objects"][5]["material"]["emit"] = [20.0, 40.0, 30.0]
    d["objects"].append(dict(name="new_light", kind="ball4", center=[0.0, 3.0, 1.0, 1.5], r=0.3,
                             material=dict(kind="emit", emit=[5.0, 20.0, 5.0])))
    edited = tmp_path / "edited.json"
    SF.save(edited, d)
    sc1, _, _ = SF.build(d, PT.procedural_env(8))
    rows = tmp_path / "rows.npy"
    np.save(rows, G.object_rows(sc1))
    out = _standalone(tmp_path / "hot", ROOT / "scenes" / "two_wormholes.json", 40, [39],
                      f"editmode 5 on\nedit 6 {gpu.qpath(rows)}\neditmode 10 off\n")
    builds = [json.loads(t.split(" ", 1)[1]) for t in out.splitlines() if t.startswith("standalone_build")]
    assert len(builds) == 2 and builds[1]["shaders"] == 27     # выход: новая структура — граф и шейдеры в процессе
    _standalone(tmp_path / "cold", edited, 29, [28])
    _same(_dumps(tmp_path / "hot", 39), _dumps(tmp_path / "cold", 28))
    log = json.loads((tmp_path / "hot" / "f.json").read_text(encoding="utf-8"))
    assert log[5]["moving"] and log[6]["moving"] and log[11]["moving"] and log[-1]["accum"] == 29


@pytest.mark.parametrize("target", ["S3", "ellis"])
def test_standalone_load_other_topology(tmp_path, target):
    """Открыть сцену с другим многообразием (loadscene) — сразу в режиме редактирования; выход — быстрые шейдеры
    нового многообразия собираются в процессе (без Python), кадр после — не пустой."""
    path = ROOT / "scenes" / f"{target}.json"
    out = _standalone(tmp_path / "hot", ROOT / "scenes" / "two_wormholes.json", 30, [29],
                      f"loadscene 5 {gpu.qpath(path)}\neditmode 10 off\n")
    assert out.count("standalone_build") == 2 and out.count("rows_check") == 3
    edit = json.loads((tmp_path / "hot" / "work" / "edit_1.json").read_text(encoding="utf-8"))
    assert edit["manifold"]["kind"] == json.loads(path.read_text(encoding="utf-8"))["manifold"]["kind"]
    assert (np.load(tmp_path / "hot" / "Present_29.npy") & 0xFFFFFF).std() > 0


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
