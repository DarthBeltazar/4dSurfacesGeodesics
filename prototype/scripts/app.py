"""Этап 5 (DD-007): интерактивное окно — генерирует сцену, шейдеры и граф, запускает `geo4d_gpu app`.

    python prototype/scripts/app.py --scene two_wormholes --res 1280x720
    python prototype/scripts/app.py --scene S3 --res 960x540
    python prototype/scripts/app.py --bench --scene two_wormholes --res 1920x1080   # замер: сценарий, скрытое окно
    python prototype/scripts/app.py --graph-only gpu/build-debug/app/two_wormholes  # только граф — запуск хоста из IDE

Управление: W/S, A/D — вперёд/вбок; Q/E — вниз/вверх; ЛКМ — захват мыши (взгляд); стрелки; Z/C — крен; Shift — ×4;
R — денойзер вкл/выкл; Esc — выход. Сцены: two_wormholes, ellis, S3, spheritorus, torisphere, tiger (T³).

Замер (--bench): 60 кадров покоя на стартовом виде (накопление), затем 60 кадров полёта (у сцены 3 — сквозь
горловину c₁ на нижний лист); журнал времени — results/stage-5/<дата>_app_bench_<сцена>_<разрешение>/frames.json."""
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
    ap.add_argument("--visible", action="store_true", help="замер в видимом окне (проверка пути на экран)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--dump", default="", help="кадры для сохранения через запятую (в каталог результатов замера)")
    ap.add_argument("--graph-only", metavar="DIR", default="",
                    help="записать граф и его входные данные в DIR и выйти (запуск/отладка `geo4d_gpu app DIR/app_graph.txt` из IDE)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    mips = PT.procedural_env(a.cube)
    sc, c0, speed = G.app_scene(a.scene, mips)
    cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], fov=a.fov, width=W, height=H)
    out = None
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(a.graph_only).resolve() if a.graph_only else Path(tmp)
        td.mkdir(parents=True, exist_ok=True)
        bench = None
        if a.bench:
            from geo4d.results import new_run
            out = new_run(5, f"app_bench_{a.scene}_{W}x{H}" + ("_visible" if a.visible else ""), dict(vars(a), gpupt=G.GPUPT().__dict__,
                                                                  slangc=gpu.slangc_version()))
            # сначала покой на стартовом виде (накопление), затем полёт
            script = [[0, 0, 0, 0, 0, 0]] * a.frames
            if a.scene == "two_wormholes":     # к горловине c₁ = (−4, 0, 0): поворот налево, затем вперёд сквозь неё
                script += [[0, 0, 0, -0.0175, 0.0, 0]] * 20 + [[0.25, 0, 0, 0, 0.0, 0]] * (a.frames - 20)
            else:
                script += [[0.05 * speed, 0, 0, 0.005, 0.002, 0]] * a.frames
            np.savetxt(td / "script.txt", np.array(script), fmt="%.6f")
            bench = (td / "script.txt", out / "frames.json")
        dumps = [(int(f), (out or td).as_posix()) for f in a.dump.split(",") if f] if a.bench else ()
        graph = G.app_graph(sc, cam, td, speed=speed, hidden=a.bench and not a.visible, vsync=a.vsync, bench=bench,
                            dumps=dumps)
        if a.graph_only:
            print(f"граф: {graph}\nзапуск: {gpu.EXE} app {graph}")
            return
        r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=a.bench, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu app: {r.stdout}\n{r.stderr}")
    if out is not None:
        fr = json.loads((out / "frames.json").read_text(encoding="utf-8"))
        mv = [f for f in fr[1:] if f["moving"]][5:]
        st = [f for f in fr if not f["moving"]][5:]
        keys = ["total_ms", "graph_ms", "denoise_ms", "post_present_ms", "camera", "lists", "pt", "fallback", "compose"]
        summ = {}
        for name, rows in [("moving", mv), ("static", st)]:
            summ[name] = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
            summ[name]["frames"] = len(rows)
        (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        summ["swapchain"] = dict(extent=fr[0]["extent"], format=fr[0]["format"])
        for f in sorted(out.glob("swapchain_*.npy")):          # путь на экран: образ swapchain = буфер вывода
            sw = np.load(f)
            pr = np.load(out / f.name.replace("swapchain_", "Present_"))
            h, w = min(sw.shape[0], pr.shape[0]), min(sw.shape[1], pr.shape[1])
            summ["swapchain"][f.stem] = dict(same_size=sw.shape == pr.shape, max_abs_diff=int(np.max(np.abs(
                sw[:h, :w].astype(np.int64) - pr[:h, :w].astype(np.int64)))))
        (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summ, indent=1, ensure_ascii=False))
        import matplotlib.pyplot as plt
        for f in sorted(out.glob("Present_*.npy")):
            u = np.load(f).astype(np.uint32)
            rgb = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8)     # BGRA8 → RGB
            plt.imsave(out / (f.stem + ".png"), rgb)


if __name__ == "__main__":
    main()
