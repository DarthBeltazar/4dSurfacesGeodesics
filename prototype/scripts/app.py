"""Этап 5 (DD-007): интерактивное окно — генерирует сцену, шейдеры и граф, запускает `geo4d_gpu app`.

    python prototype/scripts/app.py --scene two_wormholes --res 1280x720
    python prototype/scripts/app.py --scene S3 --res 960x540
    python prototype/scripts/app.py --bench --scene two_wormholes --res 1920x1080   # замер: сценарий, скрытое окно

Управление: W/S, A/D — вперёд/вбок; Q/E — вниз/вверх; ЛКМ — захват мыши (взгляд); стрелки; Z/C — крен; Shift — ×4;
R — денойзер вкл/выкл; Esc — выход. Сцены: two_wormholes, ellis, S3, spheritorus, torisphere.

Замер (--bench): 60 кадров полёта вперёд (у сцены 3 — сквозь горловину c₁ на нижний лист), затем 60 кадров покоя
(накопление); журнал времени — results/stage-5/<дата>_app_bench_<сцена>_<разрешение>/frames.json."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo4d import gpu, gpu_pt as G, pathtrace as PT
from geo4d.camera import make_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="two_wormholes")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--fov", type=float, default=75)
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--vsync", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--dump", default="", help="кадры для сохранения через запятую (в каталог результатов замера)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    mips = PT.procedural_env(a.cube)
    sc, c0, speed = G.app_scene(a.scene, mips)
    cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], fov=a.fov, width=W, height=H)
    out = None
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bench = None
        if a.bench:
            from geo4d.results import new_run
            out = new_run(5, f"app_bench_{a.scene}_{W}x{H}", dict(vars(a), gpupt=G.GPUPT().__dict__,
                                                                  slangc=gpu.slangc_version()))
            if a.scene == "two_wormholes":     # к горловине c₁ = (−4, 0, 0): поворот налево, затем вперёд сквозь неё
                script = [[0, 0, 0, -0.0175, 0.0, 0]] * 20 + [[0.25, 0, 0, 0, 0.0, 0]] * (a.frames - 20)
            else:
                script = [[0.05 * speed, 0, 0, 0.005, 0.002, 0]] * a.frames
            script += [[0, 0, 0, 0, 0, 0]] * a.frames
            np.savetxt(td / "script.txt", np.array(script), fmt="%.6f")
            bench = (td / "script.txt", out / "frames.json")
        dumps = [(int(f), (out or td).as_posix()) for f in a.dump.split(",") if f] if a.bench else ()
        graph = G.app_graph(sc, cam, td, speed=speed, hidden=a.bench, vsync=a.vsync, bench=bench, dumps=dumps)
        r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=a.bench, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu app: {r.stdout}\n{r.stderr}")
    if out is not None:
        fr = json.loads((out / "frames.json").read_text(encoding="utf-8"))
        mv = [f for f in fr[5:] if f["moving"]]
        st = [f for f in fr if not f["moving"]][5:]
        keys = ["total_ms", "graph_ms", "denoise_ms", "post_present_ms", "camera", "lists", "pt", "fallback", "compose"]
        summ = {}
        for name, rows in [("moving", mv), ("static", st)]:
            summ[name] = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
            summ[name]["frames"] = len(rows)
        (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summ, indent=1, ensure_ascii=False))
        import matplotlib.pyplot as plt
        for f in sorted(out.glob("Present_*.npy")):
            u = np.load(f).astype(np.uint32)
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8)     # BGRA8 → RGB
            plt.imsave(out / (f.stem + ".png"), rgb)


if __name__ == "__main__":
    main()
