"""DD-011, шаг 5б: ядро предпросмотра (pt_preview: первичная геодезическая, «фонарь у камеры», G-буфер) — нативное в
быстром графе (секции preview / preview_move, флажок P) и на интерпретаторе (граф предпросмотра: шейдеры от сцены не
зависят, сцена — строки и программы байткода).

- кадр интерпретатора = нативный предпросмотр той же сцены в пределах округления float32 (многообразия графа, составные
  объекты-программы, прежние виды);
- структурная правка (setparam числа многообразия, запечённого в быстрых шейдерах): кадр правки — предпросмотр на
  интерпретаторе уже изменённой сцены (= нативный предпросмотр её холодного старта), затем в фоне — быстрый граф;
- G-буфер: исход и id объекта согласованы с кадром."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu
from test_node_manifolds import sheets_doc
from test_node_slots import csg_doc, set_param

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]
RES = "160x90"


def _scene(tmp_path, name):
    if (ROOT / "scenes" / f"{name}.json").exists():
        return ROOT / "scenes" / f"{name}.json"
    doc = {"throats": sheets_doc, "csg": csg_doc}[name]()
    p = tmp_path / f"{name}.g4d"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def _extra(d, frames, dumps, extra):
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "s.txt", np.zeros((frames, 6)), fmt="%.6f")
    (d / "extra.txt").write_text(f"hidden\nbench {gpu.qpath(d / 's.txt')} {gpu.qpath(d / 'f.json')}\n" +
                                 "".join(f"dump {k} {gpu.qpath(d)}\n" for k in dumps) + extra, encoding="utf-8")
    return d / "extra.txt"


def _window(d, scene, frames, dumps, extra=""):
    ex = _extra(d, frames, dumps, extra)
    r = subprocess.run([str(gpu.EXE), str(scene), "--res", RES, "--cube", "64", "--work-dir", str(d / "w"), "--extra", str(ex)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _interp(d, scene, frames, dumps, objs=False):
    """Граф предпросмотра сам по себе: --graph-only, затем `geo4d_gpu app app_graph_prev.txt` со сценарием."""
    g = d / "g"
    r = subprocess.run([str(gpu.EXE), str(scene), "--res", RES, "--cube", "64", "--graph-only", str(g)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "prevgraph" in (g / "app_graph.txt").read_text(encoding="utf-8")
    prev = g / ("app_graph_prev_o8.txt" if objs else "app_graph_prev_p8.txt")   # с составными объектами — свой вариант
    ex = _extra(d, frames, dumps, "")
    prev.write_text(prev.read_text(encoding="utf-8") + ex.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(prev)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr


def _close(a, b):
    a, b = a[..., :3], b[..., :3]
    rel = np.abs(a - b).mean() / np.abs(b).mean()
    frac = (np.abs(a - b).max(-1) > 1e-3).mean()
    return rel, frac


SCENES = sorted(p.stem for p in (ROOT / "scenes").glob("*.json"))


@pytest.mark.parametrize("name", ["throats", "csg"] + SCENES)
def test_interpreter_equals_native_preview(tmp_path, name):
    """Все прежние виды (выход Эллиса, компактные пространства, плоские зоны) — ветви SCENE_DYNAMIC, на которых работают
    графы предпросмотра, — и многообразия и объекты графа."""
    scene = _scene(tmp_path, name)
    _window(tmp_path / "nat", scene, 3, [1], "previewmode 0 on\n")
    _interp(tmp_path / "int", scene, 3, [1], objs=name == "csg")
    a, b = (np.load(tmp_path / d / "Frame_1.npy") for d in ("int", "nat"))
    assert np.isfinite(a).all() and np.abs(b[..., :3]).mean() > 0
    rel, frac = _close(a, b)
    assert rel < 1e-4 and frac < 0.01, (rel, frac)
    ga, gb = (np.load(tmp_path / d / "GB_1.npy") for d in ("int", "nat"))
    assert np.mean(ga[..., 0, 2] == gb[..., 0, 2]) > 0.99          # исход луча
    hit = gb[..., 0, 2] == 1
    if name == "csg":
        assert hit.mean() > 0.05 and set(np.unique(gb[..., 0, 3][hit])) <= {0.0, 1.0}
    assert np.all(gb[..., 0, 3][~hit] == -1)


def test_structural_edit_shows_interpreter_preview(tmp_path):
    """b горловины (число многообразия запечено): кадр правки — интерпретатор с новым b, затем быстрый граф."""
    doc = sheets_doc()
    scene = tmp_path / "t.g4d"
    scene.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    out = _window(tmp_path / "hot", scene, 8, [2, 6], "previewmode 0 on\nsetparam 2 t1 b 0.5\n")
    assert 'params_apply {"instant": 2' in out and 'graph_switch {"kind": 3' in out and out.count("standalone_build") == 2
    new = tmp_path / "n.g4d"
    new.write_text(json.dumps(set_param(doc, "t1", "b", 0.5), ensure_ascii=False), encoding="utf-8")
    _window(tmp_path / "cold", new, 3, [1], "previewmode 0 on\n")
    old = tmp_path / "old"
    _window(old, scene, 3, [1], "previewmode 0 on\n")
    a, b, c = (np.load(p) for p in (tmp_path / "hot" / "Frame_2.npy", tmp_path / "cold" / "Frame_1.npy", old / "Frame_1.npy"))
    rel, frac = _close(a, b)
    assert rel < 1e-4 and frac < 0.01, (rel, frac)
    assert _close(c, b)[1] > 0.01                                   # правка действительно видна
    # после сборки — быстрый граф с новым b (нативный предпросмотр)
    rel, frac = _close(np.load(tmp_path / "hot" / "Frame_6.npy"), b)
    assert rel < 1e-6 and frac == 0, (rel, frac)


def test_structural_edit_with_composite_objects(tmp_path):
    """Ось поворота составной формы (целый параметр): предпросмотр — вариант графа с программами объектов."""
    doc = csg_doc()
    scene = tmp_path / "c.g4d"
    scene.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    out = _window(tmp_path / "hot", scene, 6, [2], "previewmode 0 on\nsetparam 2 xr j 3\n")
    assert 'params_apply {"instant": 2' in out
    new = tmp_path / "n.g4d"
    new.write_text(json.dumps(set_param(doc, "xr", "j", 3), ensure_ascii=False), encoding="utf-8")
    _window(tmp_path / "cold", new, 3, [1], "previewmode 0 on\n")
    rel, frac = _close(np.load(tmp_path / "hot" / "Frame_2.npy"), np.load(tmp_path / "cold" / "Frame_1.npy"))
    assert rel < 1e-4 and frac < 0.01, (rel, frac)


def test_wide_program_uses_32_register_variant(tmp_path):
    """Программа больше 8 регистров (правая вложенность произведений): граф на 32 регистра = нативный предпросмотр."""
    from test_node_objects import N, scene as obj_scene
    f = "1"
    for i in range(10, 0, -1):
        f = f"(0.9 + 0.1 * sin(0.3 * x + {i})) * ({f})"
    doc = obj_scene([], manifold=N("m", "mf.formula", formula=f"w - 0.2 * {f}", a=0.0, b=0.0, c=0.0, d=0.0, scale=1.0,
                                   s_max=40.0))
    sc = tmp_path / "wide.g4d"
    sc.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "bytecode", str(sc), str(tmp_path / "bc.json")], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 0, r.stderr
    nreg = json.loads((tmp_path / "bc.json").read_text(encoding="utf-8"))["manifold"]["f32"][0][1]
    assert 8 < nreg <= 32
    _window(tmp_path / "nat", sc, 3, [1], "previewmode 0 on\n")
    d = tmp_path / "int"
    g = d / "g"
    r = subprocess.run([str(gpu.EXE), str(sc), "--res", RES, "--cube", "64", "--graph-only", str(g)], capture_output=True,
                       text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    prev = g / "app_graph_prev_p32.txt"
    ex = _extra(d, 3, [1], "")
    prev.write_text(prev.read_text(encoding="utf-8") + ex.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(prev)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    rel, frac = _close(np.load(d / "Frame_1.npy"), np.load(tmp_path / "nat" / "Frame_1.npy"))
    assert rel < 1e-4 and frac < 0.01, (rel, frac)


@pytest.mark.parametrize("target", ["S3", "ellis", "tiger"])
def test_open_other_topology(tmp_path, target):
    """«Открыть» в редакторе нод (директива opengraph): кадр открытия — предпросмотр на интерпретаторе новой сцены с её
    камерой (= нативный предпросмотр её холодного старта), затем в фоне быстрый граф — кадры = холодный старт побитово."""
    new = ROOT / "scenes" / f"{target}.json"
    out = _window(tmp_path / "hot", ROOT / "scenes" / "two_wormholes.json", 10, [2, 9], f"opengraph 2 {new.as_posix()}\n")
    assert 'params_apply {"instant": 2' in out and out.count("standalone_build") == 2
    _window(tmp_path / "pv", new, 3, [1], "previewmode 0 on\n")
    rel, frac = _close(np.load(tmp_path / "hot" / "Frame_2.npy"), np.load(tmp_path / "pv" / "Frame_1.npy"))
    assert rel < 1e-4 and frac < 0.01, (rel, frac)
    _window(tmp_path / "cold", new, 7, [6])                       # кадр 3 (после сборки) = кадр 0 холодного старта
    for k in ("Frame", "Present"):
        assert np.array_equal(np.load(tmp_path / "hot" / f"{k}_9.npy"), np.load(tmp_path / "cold" / f"{k}_6.npy")), k
    # камера: у холодного старта кадр 0 — секция move, cam_step заново ортонормирует базис (нули → ~1e-9)
    np.testing.assert_allclose(np.load(tmp_path / "hot" / "cam_9.npy"), np.load(tmp_path / "cold" / "cam_6.npy"), rtol=0, atol=1e-7)


def test_change_manifold_kind_by_nodes(tmp_path):
    """Смена вида многообразия нодами (удалить ноду многообразия, добавить S³, связать): камера проецируется на новое M
    на CPU с касательным базисом, кадр предпросмотра сразу, затем быстрый граф."""
    out = _window(tmp_path / "hot", ROOT / "scenes" / "two_wormholes.json", 8, [7],
                  "delnode 2 manifold\naddnode 2 m2 mf.S3\nsetparam 2 m2 R 6.0\nlink 3 m2.out scene.manifold\n")   # в кадре: холст, затем setparam
    assert out.count("standalone_build") == 2 and 'graph_switch {"kind": 3' in out
    cam = np.load(tmp_path / "hot" / "cam_7.npy").astype(np.float64)
    x, f, r, u = cam[:4]
    assert abs(np.linalg.norm(x) - 6.0) < 1e-5                       # точка камеры — на S³ радиуса 6
    B = np.stack([x / np.linalg.norm(x), f, r, u])
    assert np.max(np.abs(B @ B.T - np.eye(4))) < 1e-5
    # S³ — без фона, объекты двух горловин — на прежних листах (не на новом M): кадр чёрный, но конечный
    assert np.isfinite(np.load(tmp_path / "hot" / "Frame_7.npy")).all()
