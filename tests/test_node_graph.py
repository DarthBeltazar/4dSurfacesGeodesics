"""DD-011, шаг 2: граф сцены `.g4d` (scene_graph.cpp) и библиотека нод (nodes/*.json).
- перевод version 1 → `.g4d` (`geo4d_gpu migrate`): граф окна из `.g4d` = граф окна из исходного JSON (текст и
  массивы побитово, кроме пути файла сцены в директиве editor);
- ошибки графа называют ноду: неизвестный тип, несовместимые порты, двойная связь ко входу, нет ноды «Сцена»."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu

ROOT = Path(__file__).resolve().parents[1]
SCENES = sorted(p.stem for p in (ROOT / "scenes").glob("*.json"))
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")


def _exe(*args):
    return subprocess.run([str(gpu.EXE), *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def _lines(d):
    t = (d / "app_graph.txt").read_text(encoding="utf-8")
    t = t.replace(str(d), "<DIR>").replace(d.as_posix(), "<DIR>")
    return [l for l in t.splitlines() if not l.startswith("editor ")]


@pytest.mark.parametrize("scene", SCENES)
def test_migrated_graph_equals_v1(tmp_path, scene):
    src = ROOT / "scenes" / f"{scene}.json"
    g4d = tmp_path / f"{scene}.g4d"
    r = _exe("migrate", src, g4d)
    assert r.returncode == 0, r.stderr
    doc = json.loads(g4d.read_text(encoding="utf-8"))
    assert doc["format"] == "geo4d" and doc["version"] == 2 and "main" in doc["graphs"]
    a, b = tmp_path / "a", tmp_path / "b"
    for s, d in ((src, a), (g4d, b)):
        r = _exe("export-graph", s, d, "--res", "96x54")
        assert r.returncode == 0, r.stderr
    assert _lines(a.resolve()) == _lines(b.resolve())
    for f in sorted(a.glob("*.npy")):
        assert np.array_equal(np.load(f).view(np.uint32), np.load(b / f.name).view(np.uint32)), f.name


def _doc(nodes, links):
    return {"format": "geo4d", "version": 2, "graphs": {"main": {"nodes": nodes, "links": links}},
            "camera": {"x": [0, -11, 0.5, 1.5], "f": [0, 1, 0, 0], "up": [0, 0, 1, 0]}}


BASE = [{"id": "m", "type": "mf.ellis", "params": {"b": 1.0}}, {"id": "scene", "type": "out.scene"},
        {"id": "ball", "type": "obj.ball4", "params": {"c": [3, 0, 0, 0], "r": 0.5}},
        {"id": "red", "type": "mat.diffuse"}, {"id": "o", "type": "out.object", "params": {"name": "шар"}}]
LINKS = [{"from": "m.out", "to": "scene.manifold"}, {"from": "ball.out", "to": "o.shape"},
         {"from": "red.out", "to": "o.material"}, {"from": "o.out", "to": "scene.objects"}]
BAD = {
    "unknown_type": (BASE + [{"id": "q", "type": "obj.nonexistent"}], LINKS, "«q»"),
    "port_types": (BASE, LINKS[:1] + [{"from": "red.out", "to": "o.shape"}] + LINKS[2:], "нельзя соединить"),
    "double_link": (BASE, LINKS + [{"from": "m.out", "to": "scene.manifold"}], "уже подключено"),
    "no_scene": ([n for n in BASE if n["id"] != "scene"], LINKS[1:3], "нет ноды «Сцена»"),
    "unlinked_input": (BASE, LINKS[:2] + LINKS[3:], "не подключён вход «material»"),
}


def test_hand_written_graph_loads(tmp_path):
    p = tmp_path / "ok.g4d"
    p.write_text(json.dumps(_doc(BASE, LINKS), ensure_ascii=False), encoding="utf-8")
    r = _exe("export-graph", p, tmp_path / "g", "--res", "96x54")
    assert r.returncode == 0, r.stderr
    assert "objname 0 шар" in (tmp_path / "g" / "app_graph.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("case", sorted(BAD))
def test_graph_errors(tmp_path, case):
    nodes, links, word = BAD[case]
    p = tmp_path / "bad.g4d"
    p.write_text(json.dumps(_doc(nodes, links), ensure_ascii=False), encoding="utf-8")
    r = _exe("export-graph", p, tmp_path / "g", "--res", "96x54")
    assert r.returncode != 0 and word in r.stderr, r.stderr


def _frames(d, scene, legacy, tmp_path):
    # граф: числа запечены (--bake-params) — побитовое сравнение с видами scenes.slang; слоты — test_node_slots.py
    args = ["export-graph", scene, d, "--res", "96x54", "--compile"] + (["--legacy-codegen"] if legacy else ["--bake-params"])
    r = _exe(*args)
    assert r.returncode == 0, r.stderr
    np.savetxt(tmp_path / "s.txt", np.zeros((3, 6)), fmt="%.6f")
    g = d / "app_graph.txt"
    g.write_text(g.read_text(encoding="utf-8") + f"hidden\nbench {gpu.qpath(tmp_path / 's.txt')} "
                 f"{gpu.qpath(d / 'f.json')}\ndump 2 {gpu.qpath(d)}\n", encoding="utf-8")
    r = _exe("app", g)
    assert r.returncode == 0, r.stdout + r.stderr
    return {k: np.load(d / f"{k}_2.npy") for k in ("Frame", "Present")}


@pytest.mark.gpu
@pytest.mark.parametrize("scene", SCENES)
def test_graph_codegen_equals_legacy(tmp_path, scene):
    """DD-011 §4: многообразие и объекты из формул библиотеки (SCENE_GRAPH, mf_scene.slang, obj_g из формул) дают
    тот же кадр, что виды scenes.slang с дефайнами MF_* — побитово (96×54: known-issues, «пурпурный пиксель»)."""
    src = ROOT / "scenes" / f"{scene}.json"
    a = _frames(tmp_path / "graph", src, False, tmp_path)
    b = _frames(tmp_path / "legacy", src, True, tmp_path)
    spv = lambda d: {l.split()[2] for l in (d / "app_graph.txt").read_text(encoding="utf-8").splitlines()
                     if l.startswith("pipeline camera")}
    assert spv(tmp_path / "graph") != spv(tmp_path / "legacy")          # действительно другой шейдер
    for k in a:
        assert np.array_equal(a[k], b[k]), k
