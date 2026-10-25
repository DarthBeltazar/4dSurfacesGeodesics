"""DD-011, шаг 4: слоты параметров графа сцены (§5) — числа нод в строках буфера камеры, правка без перекомпиляции.

- та же структура с другими числами — тот же текст шейдеров и ключ структуры (slotkey), другие только строки буфера;
- кадр со слотами = кадр с запечёнными числами (--bake-params) в пределах округления float32 (как DD-009);
- правка параметра (директива setparam — код панели «Параметры») = холодный старт изменённой сцены, побитово;
- структурная правка (целый параметр: ось поворота) — перегенерация быстрого графа по документу, затем числа снова
  правятся без перекомпиляции; недопустимое значение (b ≥ ρ₀) не применяется."""
import copy
import json
import subprocess

from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu
from test_node_manifolds import sheets_doc
from test_node_objects import CASES, L, N, scene

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")


def csg_doc():
    objs = [("u", copy.deepcopy(CASES["union"][0]), CASES["union"][1]),
            ("s", copy.deepcopy(CASES["rotate_scale"][0]), CASES["rotate_scale"][1])]
    for n in objs[1][1]:
        n["id"] = "x" + n["id"]
        if n["type"] == "obj.torus":
            n["params"]["c"] = [0.5, 1.5, 0.0, 0.0]
    objs[1] = ("xs", objs[1][1], [{"from": "x" + l["from"], "to": "x" + l["to"]} for l in objs[1][2]])
    for n in objs[0][1]:                      # объединение — над M (w = 0.3); «на M» сдвигает его (сдвиг — тоже слоты)
        n.get("params", {}).get("c", [0, 0, 0, 0])[3] = 0.3
    doc = scene(objs)
    g = doc["graphs"]["main"]
    g["nodes"].append(N("on", "xf.on_M"))
    for l in g["links"]:
        if l["from"] == "u.out" and l["to"].endswith(".shape"):
            l["to"] = "on.shape"
    g["links"].append(L("on", "o0.shape"))
    return doc


def set_param(doc, node, param, value):
    d = copy.deepcopy(doc)
    for n in d["graphs"]["main"]["nodes"]:
        if n["id"] == node:
            n.setdefault("params", {})[param] = value
    return d


DOCS = {"throats": lambda: sheets_doc(), "csg": csg_doc}


def _export(tmp_path, doc, name, *opts):
    d = tmp_path / name
    d.mkdir()
    (d / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "export-graph", str(d / "s.g4d"), str(d / "g"), "--res", "96x54", *opts],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return d / "g"


def _run(tmp_path, doc, name, frames, dumps, extra="", opts=()):
    d = tmp_path / name
    d.mkdir()
    (d / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    np.savetxt(d / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    (d / "extra.txt").write_text(f"hidden\nbench {gpu.qpath(d / 'script.txt')} {gpu.qpath(d / 'f.json')}\n" +
                                 "".join(f"dump {k} {gpu.qpath(d)}\n" for k in dumps) + extra, encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), str(d / "s.g4d"), "--res", "96x54", "--cube", "64", "--work-dir", str(d / "w"),
                        "--extra", str(d / "extra.txt"), *opts], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return d, r.stdout


@pytest.mark.parametrize("name,node,param,value", [("throats", "t1", "b", 0.65), ("throats", "base", "W", 1.7),
                                                    ("csg", "u", "k", 0.12), ("csg", "xr", "angle", 50.0)])
def test_same_structure_same_shaders(tmp_path, name, node, param, value):
    doc = DOCS[name]()
    a = _export(tmp_path, doc, "a")
    b = _export(tmp_path, set_param(doc, node, param, value), "b")
    ta, tb = ((g / "app_graph.txt").read_text(encoding="utf-8").replace(str(g.parent), "<D>").replace(g.parent.as_posix(), "<D>")
              for g in (a, b))
    key = lambda t: [l for l in t.splitlines() if l.startswith("slotkey")]
    assert key(ta) and key(ta) == key(tb)
    pipes = lambda t: [l for l in t.splitlines() if l.startswith("pipeline")]
    assert pipes(ta) == pipes(tb)                                     # те же SPIR-V (ключ кэша включает заголовок)
    ca, cb = np.load(a / "cam.npy"), np.load(b / "cam.npy")
    assert ca.shape == cb.shape and not np.array_equal(ca[216:], cb[216:])   # слоты — после строк многообразия (216)
    # с запечёнными числами — другой текст шейдеров
    c = _export(tmp_path, set_param(doc, node, param, value), "c", "--bake-params")
    tc = (c / "app_graph.txt").read_text(encoding="utf-8")
    assert not key(tc) and "slots " not in tc


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["throats", "csg", "two_wormholes"])
def test_slots_match_bake(tmp_path, name):
    """Кадр после 20 сэмплов: числа из слотов = запечённые литералы (округление float32 в слотах и в константах)."""
    if name == "two_wormholes":
        r = subprocess.run([str(gpu.EXE), "migrate", str(ROOT / "scenes" / "two_wormholes.json"),
                            str(tmp_path / "tw.g4d")], capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stderr
        doc = json.loads((tmp_path / "tw.g4d").read_text(encoding="utf-8"))
    else:
        doc = DOCS[name]()
    a, _ = _run(tmp_path, doc, "slots", 21, [20])
    b, _ = _run(tmp_path, doc, "bake", 21, [20], opts=["--bake-params"])
    fa, fb = (np.load(d / "Frame_20.npy")[..., :3] for d in (a, b))
    assert np.isfinite(fa).all() and np.abs(fb).mean() > 0
    rel = np.abs(fa - fb).mean() / np.abs(fb).mean()
    assert rel < 1e-3, rel


@pytest.mark.gpu
@pytest.mark.parametrize("name,node,param,value", [("throats", "t1", "b", 0.65), ("throats", "t0", "c", [-5.0, 0.0, 0.3]),
                                                    ("throats", "base", "W", 1.5), ("csg", "u", "k", 0.12),
                                                    ("csg", "xr", "angle", 50.0), ("csg", "a", "c", [0.1, 0.0, 0.0, 0.5])])
def test_setparam_equals_cold_start(tmp_path, name, node, param, value):
    """setparam (код панели «Параметры») перед кадром 2: строки объектов, многообразия и слотов — те же функции, что у
    холодного старта; кадр секцией edit (зерно 1000, как кадр 0). Кадр 7 правки = кадр 5 холодного старта, побитово.
    W листов уводит M от камеры — камера проецируется на CPU (от float32-камеры окна: совпадение — до округления)."""
    doc = DOCS[name]()
    v = value if isinstance(value, list) else [value]
    hot, out = _run(tmp_path, doc, "hot", 8, [7], extra=f"setparam 2 {node} {param} " + " ".join(map(str, v)) + "\n")
    assert 'params_apply {"instant": 1' in out and "standalone_build" in out and out.count("standalone_build") == 1
    cold, _ = _run(tmp_path, set_param(doc, node, param, value), "cold", 6, [5])
    if param == "W":
        a, b = np.load(hot / "cam_7.npy"), np.load(cold / "cam_5.npy")
        assert np.max(np.abs(a[:4] - b[:4])) < 1e-5 and np.array_equal(a[4:], b[4:])
        assert abs(a[0, 3] - 1.5) < 1e-6                           # камера на новом листе w = W
        return
    for k in ("Frame", "Present", "cam"):
        assert np.array_equal(np.load(hot / f"{k}_7.npy"), np.load(cold / f"{k}_5.npy")), k


@pytest.mark.gpu
def test_structural_param_rebuilds_then_slots_again(tmp_path):
    """Ось поворота (целый параметр) меняет текст шейдера: перегенерация по документу (edit_1.json — граф с новой осью),
    затем число той же сцены снова применяется без перекомпиляции. b ≥ ρ₀ у горловины — ошибка, не применяется."""
    doc = csg_doc()
    hot, out = _run(tmp_path, doc, "hot", 12, [11], extra="setparam 2 xr j 3\nsetparam 6 u k 0.2\n")
    assert 'params_apply {"instant": 0' in out and out.count("standalone_build") == 2
    assert 'graph_switch {"kind": 3' in out
    edit = json.loads((hot / "w" / "edit_1.json").read_text(encoding="utf-8"))
    xr = [n for n in edit["graphs"]["main"]["nodes"] if n["id"] == "xr"][0]
    assert xr["params"]["j"] == 3
    assert out.count('params_apply {"instant": 1') == 2      # правка k после сборки + повтор документа после смены графа
    fr = np.load(hot / "Frame_11.npy")
    assert np.isfinite(fr).all() and np.abs(fr[..., :3]).mean() > 0
    _, out = _run(tmp_path, sheets_doc(), "bad", 4, [3], extra="setparam 1 t0 b 3.0\n")
    assert 'params_apply {"error": 1}' in out and "standalone_build" in out and out.count("standalone_build") == 1
