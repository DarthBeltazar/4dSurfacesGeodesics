"""DD-011, шаг 6б: редактор нод — действия на холсте директивами сценария (addnode, link, unlink, delnode,
graphundo, graphredo — тот же код, что у холста). Новый объект, собранный из нод, = холодный старт документа с этими
нодами (побитово); undo возвращает исходную сцену; связь несовместимых портов и цикл отвергаются."""
import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu
from test_node_slots import csg_doc

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]
RES = "96x54"


def _window(d, doc, frames, dumps, extra=""):
    d.mkdir(parents=True, exist_ok=True)
    (d / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    np.savetxt(d / "s.txt", np.zeros((frames, 6)), fmt="%.6f")
    (d / "extra.txt").write_text(f"hidden\nbench {gpu.qpath(d / 's.txt')} {gpu.qpath(d / 'f.json')}\n" +
                                 "".join(f"dump {k} {gpu.qpath(d)}\n" for k in dumps) + extra, encoding="utf-8")
    return subprocess.run([str(gpu.EXE), str(d / "s.g4d"), "--res", RES, "--cube", "64", "--work-dir", str(d / "w"),
                           "--extra", str(d / "extra.txt")], capture_output=True, text=True, encoding="utf-8")


def _ok(r):
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


NEW = [("nb", "obj.ball4", {"c": [0.0, 1.5, 0.3, 0.0], "r": 0.4}), ("nm", "mat.diffuse", {}), ("no", "out.object", {})]
LINKS = [("nb.out", "no.shape"), ("nm.out", "no.material"), ("no.out", "scene.objects")]


def test_new_object_from_nodes_equals_cold_start(tmp_path):
    doc = csg_doc()
    ops = "".join(f"addnode 2 {i} {t}\n" for i, t, _ in NEW)
    ops += "setparam 2 nb c 0.0 1.5 0.3 0.0\nsetparam 2 nb r 0.4\n" + "".join(f"link 2 {a} {b}\n" for a, b in LINKS)
    out = _ok(_window(tmp_path / "hot", doc, 10, [9], ops))
    assert out.count("standalone_build") == 2                       # новый объект — новая структура: сборка
    cold = copy.deepcopy(doc)
    g = cold["graphs"]["main"]
    g["nodes"] += [{"id": i, "type": t, "params": p} for i, t, p in NEW]
    g["links"] += [{"from": a, "to": b} for a, b in LINKS]
    _ok(_window(tmp_path / "cold", cold, 7, [6]))                # кадр 3 (после сборки) = кадр 0 холодного старта
    for k in ("Frame", "Present"):
        a, b = np.load(tmp_path / "hot" / f"{k}_9.npy"), np.load(tmp_path / "cold" / f"{k}_6.npy")
        assert np.array_equal(a, b), k
    saved = json.loads((tmp_path / "hot" / "w" / "edit_1.json").read_text(encoding="utf-8"))
    assert {n["id"] for n in saved["graphs"]["main"]["nodes"]} >= {"nb", "nm", "no"}


def test_undo_restores_scene(tmp_path):
    doc = csg_doc()
    # разные кадры: правка с ожиданием сборки, переход на граф — в конце кадра
    ops = "delnode 2 xr\nlink 2 xt.out xs.shape\n" + "graphundo 4\ngraphundo 6\n"
    out = _ok(_window(tmp_path / "hot", doc, 14, [13], ops))
    assert out.count("standalone_build") >= 2
    _ok(_window(tmp_path / "cold", doc, 7, [6]))                 # кадр 7 (после последней сборки) = кадр 0
    for k in ("Frame", "Present"):
        assert np.array_equal(np.load(tmp_path / "hot" / f"{k}_13.npy"), np.load(tmp_path / "cold" / f"{k}_6.npy")), k


@pytest.mark.parametrize("ops,msg", [("link 1 nm.out scene.manifold\n", "типы портов"),
                                     ("link 1 xs.out xr.shape\n", "цикл"),
                                     ("delnode 1 scene\n", "Сцена")])
def test_rejected_edits(tmp_path, ops, msg):
    doc = csg_doc()
    doc["graphs"]["main"]["nodes"].append({"id": "nm", "type": "mat.diffuse"})
    r = _window(tmp_path / "x", doc, 3, [], ops)
    assert r.returncode != 0 and msg in r.stderr, r.stderr


def test_click_on_object_selects_its_node(tmp_path):
    """Клик по объекту в кадре (ядро pt_pick, G-буфер одного пикселя) выделяет его ноду «Объект»; клик по фону —
    взгляд мышью. Окно 1280×720 с оверлеем (раскладка DD-011 §8): лампа сцены с тремя горловинами — около (465, 172)."""
    d = tmp_path / "w"
    d.mkdir()
    np.savetxt(d / "s.txt", np.zeros((10, 6)), fmt="%.6f")
    (d / "extra.txt").write_text(f"hidden\nui\nbench {gpu.qpath(d / 's.txt')} {gpu.qpath(d / 'f.json')}\n"
                                 "mouse 2 465 172\nmouse 3 465 172 click\nmouse 6 300 80 click\n", encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), str(Path(__file__).resolve().parents[1] / "scenes" / "three_wormholes.g4d"),
                        "--res", "1280x720", "--cube", "64", "--work-dir", str(d / "work"), "--extra", str(d / "extra.txt")],
                       capture_output=True, text=True, encoding="utf-8")
    out = _ok(r)
    assert "pick 3 obj 0 node o1" in out
    grab = {int(l.split()[1]): int(l.split()[3]) for l in out.splitlines() if l.startswith("mouse ")}
    assert grab[4] == 0 and grab[7] == 1


def test_open_rejects_invalid_scene(tmp_path):
    """Открыть сцену с недопустимым многообразием (шары кривизны пересекаются): граф не применяется — сообщение,
    сцена прежняя (кадр = холодный старт исходной), сборки нет."""
    d = json.loads((Path(__file__).resolve().parents[1] / "scenes" / "two_wormholes.json").read_text(encoding="utf-8"))
    d["manifold"]["params"]["rho0"] = 4.5                     # |c₁ − c₂| = 8 < 2ρ₀
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d), encoding="utf-8")
    doc = csg_doc()
    out = _ok(_window(tmp_path / "hot", doc, 8, [7], f"opengraph 3 {bad.as_posix()}\n"))
    assert 'params_apply {"error": 1}' in out and out.count("standalone_build") == 1
    _ok(_window(tmp_path / "cold", doc, 8, [7]))
    for k in ("Frame", "Present"):
        assert np.array_equal(np.load(tmp_path / "hot" / f"{k}_7.npy"), np.load(tmp_path / "cold" / f"{k}_7.npy")), k
