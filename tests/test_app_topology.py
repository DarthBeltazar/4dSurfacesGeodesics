"""Окно, режим редактирования с многообразием из строк (DD-009): сцена с другой топологией открывается сразу (шейдеры
SCENE_DYNAMIC), быстрые шейдеры компилируются при выходе из режима редактирования."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF
from geo4d.camera import Camera, make_camera

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]

SCENES = ["two_wormholes", "ellis", "S3", "spheritorus", "torisphere", "tiger", "hyperplane"]


def _compact_json(tmp_path):
    """Сцена с одной компактной червоточиной (её нет среди scenes/*.json): объекты и камера — из two_wormholes."""
    d = json.loads(SF.resolve("two_wormholes").read_text(encoding="utf-8"))
    d["manifold"] = dict(kind="compact_wh", params=dict(b=1.2, rho0=3.5, W=1.5))
    d["objects"] = [o for o in d["objects"] if o["name"] != "ball" or o["center"][0] == 0.0]
    path = tmp_path / "compact_wh.json"
    SF.save(path, d)
    return path


def _editor_cfg(td, scene_path, res="96x54"):
    return dict(scene=Path(scene_path).as_posix(), work_dir=td.as_posix(), fov=75.0,
                env=json.loads(Path(scene_path).read_text(encoding="utf-8")).get("env"),
                python=Path(sys.executable).as_posix(),
                script=(Path(__file__).resolve().parents[1] / "prototype/scripts/app.py").as_posix(),
                args=["--res", res, "--cube", "64", "--bake-params"])   # эталон — граф Python (литералы, без слотов DD-011)


def _cam(sc, c, w=96, h=54):
    return make_camera(sc.shape, c["x"], c["f"], c["up"], c.get("right"), fov=c.get("fov", 75.0), width=w, height=h)


def _run(td, scene_path, frames, dump=(), extra="", editor=True, cam=None):
    sc, c, speed = SF.load(scene_path, PT.procedural_env(64))
    td.mkdir(parents=True, exist_ok=True)
    np.savetxt(td / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    graph = G.app_graph(sc, cam or _cam(sc, c), td, speed=speed, hidden=True,
                        bench=(td / "script.txt", td / "f.json"), dumps=[(k, td.as_posix()) for k in dump],
                        editor=_editor_cfg(td, scene_path) if editor else None, dyn_mips=PT.procedural_env(64))
    graph.write_text(graph.read_text(encoding="utf-8") + extra, encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("rows_check")]
    mf = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("mf_check")]
    assert rows and max(rows) < 1e-6
    if editor:                                   # строки многообразия C++ (manifold_info) = Python (manifold_rows)
        assert mf and max(mf) < 1e-6, mf
    return r.stdout


@pytest.mark.parametrize("name", SCENES + ["compact_wh"])
def test_dynamic_manifold_matches_baked(tmp_path, name):
    """Шейдеры с многообразием из строк (режим редактирования) и быстрые (многообразие зашито) считают одно и то же для
    каждого вида: кадр после 20 сэмплов совпадает с точностью округления float32. Заодно — mf_check на обоих графах."""
    path = _compact_json(tmp_path) if name == "compact_wh" else SF.resolve(name)
    _run(tmp_path / "dyn", path, 25, dump=[24], extra="editmode 4 on\n")
    _run(tmp_path / "baked", path, 21, dump=[20], editor=False)
    a = np.load(tmp_path / "dyn" / "Frame_24.npy")[..., :3]
    b = np.load(tmp_path / "baked" / "Frame_20.npy")[..., :3]
    rel = np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-12)
    assert rel < 1e-3, rel


@pytest.mark.parametrize("target", ["S3", "ellis", "compact_wh"])
def test_load_other_topology_then_compile(tmp_path, target):
    """Открыть сцену с другим многообразием (директива loadscene — код кнопки «Открыть»): сразу режим редактирования,
    камера спроецирована на новое M с касательным базисом; выход — перегенерация быстрых шейдеров (Python) и переход
    на них с той же камерой. Итог побитово равен холодному старту быстрого графа этой сцены с той же камерой."""
    path = _compact_json(tmp_path) if target == "compact_wh" else SF.resolve(target)
    hot = tmp_path / "hot"
    out = _run(hot, SF.resolve("two_wormholes"), 40, dump=[8, 39],
               extra=f"loadscene 5 {path.as_posix()}\neditmode 10 off\n")
    assert out.count("rows_check") == 3                       # исходный граф, режим редактирования, новый быстрый
    edit = json.loads((hot / "edit_1.json").read_text(encoding="utf-8"))
    assert edit["manifold"]["kind"] == json.loads(path.read_text(encoding="utf-8"))["manifold"]["kind"]
    sc, c, _ = SF.load(path, PT.procedural_env(64))
    cam8, cam39 = np.load(hot / "cam_8.npy"), np.load(hot / "cam_39.npy")
    assert np.array_equal(cam8[:4], cam39[:4])                # выход из режима редактирования — камера та же
    x, f, r, u = (cam39[i].astype(np.float64) for i in range(4))
    assert abs(float(sc.shape.f(x))) < 1e-5 * max(1.0, sc.shape.scale)
    import jax
    n = np.array(jax.grad(sc.shape.f)(x))
    n /= np.linalg.norm(n)
    B = np.stack([n, f, r, u])
    assert np.max(np.abs(B @ B.T - np.eye(4))) < 1e-5 and np.linalg.det(B) > 0
    ref = _cam(sc, c)                       # Python (make_camera) по тем же подсказкам из файла — тот же базис
    assert np.max(np.abs(np.stack([ref.x, ref.f, ref.r, ref.u]) - np.stack([x, f, r, u]))) < 1e-4
    assert (np.load(hot / "Present_8.npy") & 0xFFFFFF).std() > 0        # кадр в режиме редактирования — не пустой
    cold_cam = Camera(x, f, r, u, fov=75.0, width=96, height=54)
    _run(tmp_path / "cold", path, 29, dump=[28], editor=False, cam=cold_cam)
    for k in ["Frame", "Present"]:
        assert np.array_equal(np.load(hot / f"{k}_39.npy"), np.load(tmp_path / "cold" / f"{k}_28.npy")), k
    log = json.loads((hot / "f.json").read_text(encoding="utf-8"))
    assert log[6]["moving"] and log[11]["moving"] and log[-1]["accum"] == 29


def test_load_rejects_invalid_manifold(tmp_path):
    """Недопустимые параметры в открываемом файле — отказ сразу в C++ (без Python), сцена прежняя (в сценарии — ошибка
    с сообщением)."""
    d = json.loads(SF.resolve("two_wormholes").read_text(encoding="utf-8"))
    d["manifold"]["params"]["rho0"] = 4.5                     # |c₁ − c₂| = 8 < 2ρ₀
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d), encoding="utf-8")
    sc, c, speed = SF.load(SF.resolve("two_wormholes"), PT.procedural_env(64))
    np.savetxt(tmp_path / "script.txt", np.zeros((10, 6)), fmt="%.6f")
    graph = G.app_graph(sc, _cam(sc, c), tmp_path, speed=speed, hidden=True,
                        bench=(tmp_path / "script.txt", tmp_path / "f.json"),
                        editor=_editor_cfg(tmp_path, SF.resolve("two_wormholes")), dyn_mips=PT.procedural_env(64))
    graph.write_text(graph.read_text(encoding="utf-8") + f"loadscene 3 {bad.as_posix()}\n", encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "недопустимые параметры" in r.stderr, r.stderr
    assert not (tmp_path / "edit_1.json").exists()
