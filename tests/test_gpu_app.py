"""Этап 5 (DD-007): приложение geo4d_gpu app (скрытое окно) — полёт сквозь горловину по сценарию, переход на нижний
лист, денойзер на памяти Vulkan (импорт в CUDA) совпадает с денойзером через копии, вывод не пуст."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geo4d import gpu, gpu_pt as G, pathtrace as PT
from geo4d.camera import make_camera

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]


def test_app_flight_through_throat(tmp_path):
    sc, c0, speed = G.app_scene("two_wormholes", PT.procedural_env(64))
    W, H = 160, 90
    cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], fov=75, width=W, height=H)
    script = [[0, 0, 0, -0.0175, 0, 0]] * 20 + [[0.25, 0, 0, 0, 0, 0]] * 60 + [[0, 0, 0, 0, 0, 0]] * 10
    np.savetxt(tmp_path / "script.txt", np.array(script), fmt="%.6f")
    graph = G.app_graph(sc, cam, tmp_path, speed=speed, hidden=True, bench=(tmp_path / "script.txt", tmp_path / "f.json"),
                        dumps=[(0, tmp_path.as_posix()), (89, tmp_path.as_posix())])
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    log = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert len(log) == len(script) and log[-1]["accum"] == 11      # 1 spp кадра движения + 10 кадров покоя
    cam0, cam1 = np.load(tmp_path / "cam_0.npy"), np.load(tmp_path / "cam_89.npy")
    assert cam0[0, 3] > 0 and cam1[0, 3] < 0                  # верхний лист → нижний (M33 сквозь горловину c₁)
    for f in [0, 89]:
        fr, a, n, d = (np.load(tmp_path / f"{k}_{f}.npy") for k in ["Frame", "AlbF", "NrmF", "Den"])
        ref, _ = gpu.denoise(fr, a, n)
        assert np.max(np.abs(ref - d[..., :3])) < 1e-5            # обмен памятью = копии
        u = np.load(tmp_path / f"Present_{f}.npy")
        assert np.isfinite(d).all() and (u & 0xFFFFFF).std() > 0
