"""Окно приложения (DD-009): масштабирование кадра в образ swapchain другого размера, смена разрешения рендера."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF
from geo4d.camera import make_camera

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]


def _scene(w, h):
    sc, c, speed = SF.load(SF.resolve("two_wormholes"), PT.procedural_env(64))
    return sc, make_camera(sc.shape, c["x"], c["f"], c["up"], fov=75, width=w, height=h), speed


def _editor_cfg(td, res):
    return dict(scene=SF.resolve("two_wormholes").as_posix(), work_dir=td.as_posix(), fov=75.0, env=None,
                python=Path(sys.executable).as_posix(),
                script=(Path(__file__).resolve().parents[1] / "prototype/scripts/app.py").as_posix(),
                args=["--res", res, "--cube", "64"])


def _run(td, w, h, frames, dump, extra="", editor=False):
    sc, cam, speed = _scene(w, h)
    td.mkdir(parents=True, exist_ok=True)
    np.savetxt(td / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    graph = G.app_graph(sc, cam, td, speed=speed, hidden=True, bench=(td / "script.txt", td / "f.json"),
                        dumps=[(dump, td.as_posix())], editor=_editor_cfg(td, f"{w}x{h}") if editor else None)
    graph.write_text(graph.read_text(encoding="utf-8") + extra, encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _rgb(u):
    u = np.asarray(u).astype(np.uint32)
    return np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.float64)


def test_letterbox_blit(tmp_path):
    """Кадр 96×54 в окне 192×162: масштаб 2 по меньшей стороне, кадр 192×108 по центру (поля по 27 строк — чёрные),
    среднее 2×2 образа swapchain = кадр рендера с точностью линейной фильтрации."""
    _run(tmp_path, 96, 54, 4, 3, extra="window 192 162\n")
    sw, pr = np.load(tmp_path / "swapchain_3.npy"), np.load(tmp_path / "Present_3.npy")
    assert sw.shape == (162, 192) and pr.shape == (54, 96)
    assert np.all((sw[:27] & 0xFFFFFF) == 0) and np.all((sw[135:] & 0xFFFFFF) == 0)
    img = _rgb(sw[27:135])
    down = img.reshape(54, 2, 96, 2, 3).mean((1, 3))
    ref = _rgb(pr)
    assert np.abs(down - ref).mean() < 4.0, np.abs(down - ref).mean()      # из 255; края — сглаживание соседями
    assert np.abs(down.mean((0, 1)) - ref.mean((0, 1))).max() < 1.0


def test_resize_equals_cold_start(tmp_path):
    """Смена разрешения рендера (директива resize — тот же код, что выбор в панели «Рендер»): перегенерация графа
    Python с новым --res, перезагрузка без закрытия окна с той же камерой, повторный импорт денойзера. Итог побитово
    равен холодному старту на новом разрешении с тем же числом кадров после сброса."""
    _run(tmp_path / "hot", 96, 54, 16, 15, extra="resize 5 128 72\n", editor=True)
    _run(tmp_path / "cold", 128, 72, 10, 9)
    hot, cold = tmp_path / "hot", tmp_path / "cold"
    assert json.loads((hot / "edit_1.json").read_text(encoding="utf-8"))["camera"]["fov"] == 75.0
    for k in ["Frame", "Present"]:
        a, b = np.load(hot / f"{k}_15.npy"), np.load(cold / f"{k}_9.npy")
        assert a.shape[:2] == (72, 128) and np.array_equal(a, b), k
    assert np.load(hot / "swapchain_15.npy").shape == (54, 96)        # окно прежнее: кадр 128×72 уменьшен в него
    log = json.loads((hot / "f.json").read_text(encoding="utf-8"))
    assert log[6]["moving"] and log[-1]["accum"] == 10
    fr, a, n, d = (np.load(hot / f"{k}_15.npy") for k in ["Frame", "AlbF", "NrmF", "Den"])
    ref, _ = gpu.denoise(fr, a, n)
    assert np.max(np.abs(ref - d[..., :3])) < 1e-5


def test_click_on_panel_is_not_camera_grab(tmp_path):
    """Клик ЛКМ по панели редактора не включает взгляд мышью, даже если мышь пришла на панель со сцены в той же пачке
    событий (решение — после ImGui::NewFrame, а не по WantCaptureMouse прошлого кадра); клик по сцене включает,
    повторный — выключает. События мыши — директива mouse (как от SDL), оверлей в скрытом окне — директива ui."""
    out = _run(tmp_path, 1280, 720, 14, 13, editor=True,    # (640, 660) — кадр вне панелей при любом масштабе экрана
               extra="ui\nmouse 2 640 660\nmouse 3 100 15 click\nmouse 7 640 660 click\nmouse 10 640 660 click\n")
    st = {}
    for line in out.splitlines():
        if line.startswith("mouse "):
            w = line.split()
            st[int(w[1])] = dict(grab=int(w[3]), imgui=int(w[5]))
    assert st[3]["grab"] == 0 and st[4]["grab"] == 0 and st[4]["imgui"] == 1     # заголовок панели «Сцена»
    assert st[8]["grab"] == 1                                                    # сцена — взгляд мышью
    assert st[11]["grab"] == 0                                                   # повторный щелчок — отпустить


def test_paths_with_spaces(tmp_path):
    """Каталог графа с пробелом (архив редактора в «C:\My Programs», пользователь с пробелом в имени): пути в
    директивах — в кавычках (gpu.qpath, words.hpp); перегенерация (resize → Python с --graph-only) тоже проходит."""
    td = tmp_path / "dir with space"
    _run(td, 96, 54, 8, 7, extra="resize 3 128 72\n", editor=True)
    assert np.load(td / "Present_7.npy").shape == (72, 128)
