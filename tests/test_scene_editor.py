"""Редактор сцен (DD-008): файлы сцен, строки объектов на GPU, правило NEE-источников M34, параметры многообразия."""
import json

import numpy as np
import pytest

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF
from geo4d.camera import make_camera

APP_SCENES = ["two_wormholes", "ellis", "S3", "spheritorus", "torisphere", "tiger"]
ENV = None


def _env():
    global ENV
    if ENV is None:
        ENV = PT.constant_env()
    return ENV


@pytest.mark.parametrize("name", APP_SCENES)
def test_scene_file_matches_app_scene(name):
    """scenes/<имя>.json даёт ту же сцену, что и gpu_pt.app_scene: строки объектов, заголовок, дефайны, meta, камера."""
    sc, c0, speed = G.app_scene(name, _env())
    sc2, c2, sp2 = SF.load(SF.resolve(name), _env())
    assert np.array_equal(G.object_rows(sc), G.object_rows(sc2))
    assert G.scene_header(sc) == G.scene_header(sc2) and G._defs(sc) == G._defs(sc2)
    assert sc.meta == sc2.meta and sc.lights == sc2.lights and sp2 == speed
    assert np.max(np.abs(c0["x"] - c2["x"])) < 1e-12        # проекция на M (у Эллиса — сдвиг ~1e-16)


@pytest.mark.parametrize("name", APP_SCENES + ["hyperplane"])
def test_scene_file_roundtrip(name, tmp_path):
    sc, cam, speed = SF.load(SF.resolve(name), _env())
    d = SF.to_dict(sc, cam, speed)
    SF.save(tmp_path / "s.json", d)
    sc2, _, _ = SF.load(tmp_path / "s.json", _env())
    assert np.array_equal(G.object_rows(sc), G.object_rows(sc2))
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8")) == json.loads(json.dumps(d))


def test_nee_rule_matches_manual_lights():
    """M34 на сценах этапа 4 совпадает с заданными вручную списками источников (и с печью — без источников)."""
    for sc in [PT.scene_hyperplane(_env()), PT.scene_two_wormholes(_env()), PT.scene_ellis(_env()),
               PT.scene_hyperplane(furnace=True), PT.scene_two_wormholes(furnace=True)]:
        assert PT.nee_lights(sc.objs, sc.flat) == tuple(sc.lights)


def test_nee_rule_excludes_invalid_lights():
    """Источник вне листа или задевающий шар кривизны — не NEE (только BSDF)."""
    from geo4d.objects import ball4, Material
    sc = PT.scene_two_wormholes(_env())
    em = Material("emit", emit=(1.0, 1.0, 1.0))
    W = 1.5
    objs = [ball4((0, 0, 4, W), 0.5, em),             # в зоне — да
            ball4((0, 0, 4, W + 0.1), 0.5, em),       # не на листе
            ball4((-4, 3.2, 0, W), 0.5, em),          # центр вне шара ρ₀ = 3, но шар источника его задевает
            ball4((-4, 3.6, 0, -W), 0.5, em)]         # нижний лист, вплотную к ρ₀ + r — да
    assert PT.nee_lights(objs, sc.flat) == (0, 3)


@pytest.mark.parametrize("kind,params", [("two_compact_wh", dict(rho0=4.5)), ("compact_wh", dict(b=3.0)),
                                         ("spheritorus", dict(r=1.2)), ("S3", dict(R=-1.0))])
def test_invalid_manifold_params(kind, params):
    with pytest.raises(ValueError):
        SF.make_shape(kind, params)


def edited_two_wormholes():
    """Сцена 3 с другими параметрами многообразия; объекты на M (on_M), камера — на верхнем листе w = W."""
    d = json.loads(SF.resolve("two_wormholes").read_text(encoding="utf-8"))
    W = 2.0
    d["manifold"]["params"].update(b=1.3, rho0=3.5, W=W, c1=[-4.5, 0.5, 0.0], c2=[4.5, 0.0, 0.3])
    for o in d["objects"]:
        if o["kind"] != "plane":
            o["on_M"] = True
            o["center"][3] = np.sign(o["center"][3]) * W
    d["camera"]["x"] = [0, -11.0, -1.0, W]
    return d


def test_on_M_projection():
    sc, cam, _ = SF.build(edited_two_wormholes(), _env())
    for o in sc.objs:
        if o.on_M:
            assert abs(float(sc.shape.f_vec(np.asarray(o.center)))) < 1e-10
    assert abs(float(sc.shape.f_vec(np.asarray(cam["x"])))) < 1e-10
    assert PT.nee_lights(sc.objs, sc.flat) == (5, 6)


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_cam_step_edited_manifold():
    """M33 на GPU с параметрами многообразия из дефайнов (DD-008) против прототипа: полёт сквозь сдвинутую горловину c₁."""
    from geo4d.camera import fly
    sc, _, _ = SF.build(edited_two_wormholes(), _env())
    c1 = np.array([-4.5, 0.5, 0.0])
    cam = make_camera(sc.shape, np.r_[c1 + [0, -6.0, 0.3], 2.0], [0, 1, 0, 0], [0, 0, 1, 0])
    c = cam
    for _ in range(40):
        c = fly(sc.shape, c, move=(0.3, 0, 0), rot=(0, 0, 0), substeps=16)
    g = G.cam_step(sc, cam, (0.3, 0, 0), (0, 0, 0), frames=40, margin=-1e9)
    for a, b in zip(g, [c.x, c.f, c.r, c.u]):
        assert np.linalg.norm(a - b) < 2e-3
    assert c.x[3] < 0


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
@pytest.mark.parametrize("which", ["two_wormholes", "ellis"])
def test_gpu_vs_prototype_edited_manifold(which):
    """Как test_gpu_pt.py::test_gpu_vs_prototype, но на параметрах не по умолчанию (дефайны многообразия, объекты из
    буфера): средние по пикселям GPU (float32) и прототипа (float64) совпадают в пределах статистики. Доля выбросов
    > 4σ — до 10 % (из 36 пикселей 5 % — меньше двух): у прототипа тяжёлые хвосты от малых источников; при 2048 spp
    прототипа выброс сцены 3 (z = 12.5 при 256 spp) даёт z ≈ 1 (журнал 2026-10-24-08)."""
    if which == "ellis":
        d = json.loads(SF.resolve("ellis").read_text(encoding="utf-8"))
        d["manifold"]["params"]["b"] = 1.5
        for o in d["objects"]:
            o["on_M"] = True
        spp = 512
    else:
        d, spp = edited_two_wormholes(), 512
    sc, c, _ = SF.build(d)                          # процедурный фон, как в test_gpu_vs_prototype
    cam = make_camera(sc.shape, c["x"], c["f"], c["up"], fov=70, width=6, height=6)
    tol = dict(tol_kappa=1e-3, tol_max=1e-4)
    a, va, _ = PT.render(sc, cam, PT.PTConfig(spp=spp), seed=7)
    b, vb, _, _ = G.render(sc, cam, 32768, G.GPUPT(**tol))
    assert abs((a.mean() - b.mean()) / np.sqrt((va + vb).sum() / a.size ** 2)) < 4
    assert np.mean(np.abs(a - b) > 4 * np.sqrt(va + vb)) <= 0.1


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_gpu_snap_matches_projection():
    """Ядро pt_snap (float32, Ньютон по ∇f) против проекции прототипа (float64) для центров вне M, в том числе в зоне
    кривизны; объекты без on_M и полупространства не трогаются."""
    d = edited_two_wormholes()
    sc, _, _ = SF.build(d, _env())                 # центры уже на M (Python)
    for o in d["objects"]:                         # сдвинуть с M: GPU должен вернуть туда же
        if o["kind"] != "plane":
            o["center"][3] *= 1.15
            o["center"][0] += 0.05
    sc_off, _, _ = SF.build({**d, "objects": [{**o, "on_M": False} for o in d["objects"]]}, _env())
    for a, b in zip(sc_off.objs, d["objects"]):
        a.on_M = b.get("on_M", False)
    rows = G.snap_objects(sc_off)
    for k, (o, o_ref) in enumerate(zip(sc_off.objs, sc.objs)):
        c = rows[9 + 6 * k]
        if o.on_M:
            ref = SF._project(sc.shape, np.r_[o.center])[0]
            assert np.max(np.abs(c - ref)) < 2e-5, (k, c, ref)
        elif o.center is not None:
            assert np.array_equal(c, np.float32(o.center))


def _editor_cfg(td, res="96x54"):
    import sys
    from pathlib import Path
    return dict(scene=SF.resolve("two_wormholes").as_posix(), work_dir=td.as_posix(), fov=75.0, env=None,
                python=Path(sys.executable).as_posix(),
                script=(Path(__file__).resolve().parents[1] / "prototype/scripts/app.py").as_posix(),
                args=["--res", res, "--cube", "64", "--bake-params"])   # эталон — граф Python (литералы, без слотов DD-011)


def _app_run(sc, cam, td, frames, edits=(), dump=None, editor=None, extra=""):
    import subprocess
    td.mkdir(parents=True, exist_ok=True)
    np.savetxt(td / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    graph = G.app_graph(sc, cam, td, hidden=True, bench=(td / "script.txt", td / "f.json"), edits=edits,
                        dumps=[(dump, td.as_posix())] if dump is not None else (), editor=editor)
    if extra:
        graph.write_text(graph.read_text(encoding="utf-8") + extra, encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    checks = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("rows_check")]
    assert checks and max(checks) < 1e-6          # упаковка строк C++ (editor.cpp) = Python (object_rows)
    return r.stdout


def _edited(structural):
    env = PT.procedural_env(64)
    d = json.loads(SF.resolve("two_wormholes").read_text(encoding="utf-8"))
    sc0, c, _ = SF.build(d, env)
    d["objects"][1]["center"][0] += 1.0                                   # сдвиг шара
    d["objects"][2]["material"]["rough"] = 0.6                           # металл тора
    d["objects"][5]["material"]["emit"] = [20.0, 40.0, 30.0]             # цвет источника
    if structural:
        d["objects"].append(dict(name="new_light", kind="ball4", center=[0.0, 3.0, 1.0, 1.5], r=0.3,
                                 material=dict(kind="emit", emit=[5.0, 20.0, 5.0])))
    sc1, _, _ = SF.build(d, env)
    cam = make_camera(sc0.shape, c["x"], c["f"], c["up"], fov=75, width=96, height=54)
    return sc0, sc1, cam


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_app_param_edit_equals_cold_start(tmp_path):
    """Правка параметров объектов в быстром графе (строки буфера, секция edit, сброс накопления) после N кадров
    покоя даёт тот же кадр, что холодный старт отредактированной сцены с тем же числом кадров — побитово."""
    sc0, sc1, cam = _edited(False)
    assert G.structure_key(sc0) == G.structure_key(sc1)
    _app_run(sc0, cam, tmp_path / "hot", 30, edits=[(10, G.object_rows(sc1))], dump=29)
    _app_run(sc1, cam, tmp_path / "cold", 20, dump=19)
    for k in ["Frame", "Present"]:
        assert np.array_equal(np.load(tmp_path / "hot" / f"{k}_29.npy"), np.load(tmp_path / "cold" / f"{k}_19.npy")), k
    log = json.loads((tmp_path / "hot" / "f.json").read_text(encoding="utf-8"))
    assert log[10]["moving"] and log[-1]["accum"] == 20


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_app_edit_mode_structural_cycle(tmp_path):
    """Режим редактирования (DD-008): вход — граф с динамическими шейдерами, правка со сменой структуры (новый
    источник), выход — фоновая перегенерация быстрых шейдеров и переход на них с той же камерой. Итог побитово равен
    холодному старту отредактированной сцены с тем же числом кадров после сброса."""
    sc0, sc1, cam = _edited(True)
    assert PT.nee_lights(sc1.objs, sc1.flat) == (5, 6, 7)
    extra = "editmode 5 on\neditmode 10 off\n"
    _app_run(sc0, cam, tmp_path / "hot", 40, edits=[(6, G.object_rows(sc1))], dump=39,
             editor=_editor_cfg(tmp_path / "hot"), extra=extra)
    _app_run(sc1, cam, tmp_path / "cold", 29, dump=28)
    edit = json.loads((tmp_path / "hot" / "edit_1.json").read_text(encoding="utf-8"))
    assert len(edit["objects"]) == 8 and edit["manifold"]["params"]["W"] == 1.5
    for k in ["Frame", "Present"]:
        assert np.array_equal(np.load(tmp_path / "hot" / f"{k}_39.npy"), np.load(tmp_path / "cold" / f"{k}_28.npy")), k
    log = json.loads((tmp_path / "hot" / "f.json").read_text(encoding="utf-8"))
    assert log[5]["moving"] and log[6]["moving"] and log[11]["moving"] and log[-1]["accum"] == 29


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_app_dynamic_matches_baked(tmp_path):
    """Динамические шейдеры режима редактирования и быстрые (структура зашита) считают одно и то же: кадр после
    20 сэмплов совпадает с точностью округления float32 (разный машинный код)."""
    sc0, _, cam = _edited(False)
    _app_run(sc0, cam, tmp_path / "dyn", 25, dump=24, editor=_editor_cfg(tmp_path / "dyn"), extra="editmode 4 on\n")
    _app_run(sc0, cam, tmp_path / "baked", 21, dump=20)
    a = np.load(tmp_path / "dyn" / "Frame_24.npy")[..., :3]
    b = np.load(tmp_path / "baked" / "Frame_20.npy")[..., :3]
    rel = np.abs(a - b).mean() / np.abs(b).mean()
    assert rel < 1e-3, rel


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
@pytest.mark.parametrize("bad", [False, True])
def test_app_remanifold(tmp_path, bad):
    """Смена параметров многообразия из окна (тот же код, что кнопка «Применить»): сразу — режим редактирования с
    многообразием из строк (DD-009), камера на листе w = 1.5 проецируется на новый лист w = W = 2 (GPU); выход из него —
    хост пишет JSON с текущей камерой, запускает app.py --graph-only, перезагружает граф без закрытия окна и заново
    импортирует буферы денойзера; денойзер после перезагрузки = денойзер через копии. Недопустимые параметры —
    отказ сразу (C++), без Python."""
    import subprocess
    import sys
    from pathlib import Path
    env = PT.procedural_env(64)
    sc, c, speed = SF.load(SF.resolve("two_wormholes"), env)
    cam = make_camera(sc.shape, c["x"], c["f"], c["up"], fov=75, width=96, height=54)
    np.savetxt(tmp_path / "script.txt", np.zeros((40, 6)), fmt="%.6f")
    ed = dict(scene=SF.resolve("two_wormholes").as_posix(), work_dir=tmp_path.as_posix(), fov=75.0, env=None,
              python=Path(sys.executable).as_posix(),
              script=(Path(__file__).resolve().parents[1] / "prototype/scripts/app.py").as_posix(),
              args=["--res", "96x54", "--cube", "64", "--bake-params"])
    graph = G.app_graph(sc, cam, tmp_path, speed=speed, hidden=True, bench=(tmp_path / "script.txt", tmp_path / "f.json"),
                        dumps=[(9, tmp_path.as_posix()), (39, tmp_path.as_posix())], editor=ed)
    rho0 = 4.5 if bad else 3.5                               # 4.5: шары кривизны пересекаются (|c₁ − c₂| = 8 < 2ρ₀)
    graph.write_text(graph.read_text(encoding="utf-8") + f"remanifold 10 two_compact_wh W=2.0 rho0={rho0}\n"
                     "editmode 15 off\n", encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
    if bad:      # в окне — сообщение и старая сцена; в сценарии — ошибка с текстом (как у Python)
        assert r.returncode != 0 and "недопустимые параметры" in r.stderr, r.stderr
        return
    assert r.returncode == 0, r.stdout + r.stderr
    checks = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("rows_check")]
    assert len(checks) == 3 and max(checks) < 1e-6      # исходный граф, режим редактирования, перегенерированный
    edit = json.loads((tmp_path / "edit_1.json").read_text(encoding="utf-8"))
    assert edit["manifold"]["params"]["W"] == 2.0 and edit["manifold"]["params"]["rho0"] == 3.5
    cam0, cam1 = np.load(tmp_path / "cam_9.npy"), np.load(tmp_path / "cam_39.npy")
    assert abs(cam0[0, 3] - 1.5) < 1e-6 and abs(cam1[0, 3] - 2.0) < 1e-5
    assert np.array_equal(cam1[G.MF_BASE:G.MF_BASE + 6, :3], G.manifold_rows(SF.build(edit)[0])[:6, :3])
    fr, a, n, d = (np.load(tmp_path / f"{k}_39.npy") for k in ["Frame", "AlbF", "NrmF", "Den"])
    ref, _ = gpu.denoise(fr, a, n)
    assert np.max(np.abs(ref - d[..., :3])) < 1e-5
    u = np.load(tmp_path / "Present_39.npy")
    assert (u & 0xFFFFFF).std() > 0
    log = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert len(log) == 40 and log[11]["moving"] and log[16]["moving"] and log[-1]["accum"] == 24
