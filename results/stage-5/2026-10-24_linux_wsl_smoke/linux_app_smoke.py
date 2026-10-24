"""Проверка приложения Linux-сборки (WSL, lavapipe): полёт сквозь горловину по сценарию, переход на нижний лист,
вывод не пуст; денойзер в WSL недоступен — ожидается работа без него."""
import json, subprocess, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, r"C:\Users\AlexandrGeorgiev\CLionProjects\4dSurfacesHeodesics\prototype")
from geo4d import gpu, gpu_pt as G, pathtrace as PT
from geo4d.camera import make_camera

td = Path(sys.argv[1]); td.mkdir(parents=True, exist_ok=True)
sc, c0, speed = G.app_scene("two_wormholes", PT.procedural_env(64))
W, H = 160, 90
cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], fov=75, width=W, height=H)
script = [[0, 0, 0, -0.0175, 0, 0]] * 20 + [[0.25, 0, 0, 0, 0, 0]] * 60 + [[0, 0, 0, 0, 0, 0]] * 10
np.savetxt(td / "script.txt", np.array(script), fmt="%.6f")
graph = G.app_graph(sc, cam, td, speed=speed, hidden=sys.argv[2] == "hidden", bench=(td / "script.txt", td / "f.json"),
                    dumps=[(0, td.as_posix()), (89, td.as_posix())])
r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True)
print("rc", r.returncode, r.stdout[-500:], r.stderr[-800:])
log = json.loads((td / "f.json").read_text(encoding="utf-8"))
print("frames", len(log), "accum", log[-1]["accum"], "extent", log[0]["extent"], "format", log[0]["format"],
      "total_ms median", np.median([f["total_ms"] for f in log]))
cam0, cam1 = np.load(td / "cam_0.npy"), np.load(td / "cam_89.npy")
print("sheet w:", cam0[0, 3], "->", cam1[0, 3])
for f in [0, 89]:
    u = np.load(td / f"Present_{f}.npy")
    print("Present", f, "std", (u & 0xFFFFFF).std())
    sw = td / f"swapchain_{f}.npy"
    if sw.exists():
        s = np.load(sw); print("swapchain", f, s.shape, "max diff", int(np.max(np.abs(s.astype(np.int64) - u.astype(np.int64)))))
