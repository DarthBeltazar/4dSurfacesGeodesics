"""Этап 5 (DD-007): офлайн-ролик по пути камеры — тот же конвейер, что в окне (geo4d_gpu app, скрытое окно).

Путь — сценарий шагов камеры M33 (перемещение в базисе камеры и повороты за кадр ролика). Каждый кадр ролика:
шаг камеры (проход камеры, режим A, 1 spp) + (spp − 1) кадров накопления → денойзер OptiX → фон режима A → запись
BGRA8; затем ffmpeg (H.264, yuv420p). Сцена 3: поворот на горловину c₁, полёт сквозь неё на нижний лист, разворот.

    python prototype/scripts/video.py --res 1920x1080 --spp 64
Результат: results/stage-5/<дата>_video/ — video.mp4 (не в git), кадры-превью PNG, путь камеры, время."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d import gpu, gpu_pt as G, pathtrace as PT
from geo4d.camera import make_camera
from geo4d.results import new_run


def path_two_wormholes(fps):
    """Шаги на кадр: (a, b, c, ψ, θ, φ). Длительность ≈ 10 с при 30 к/с."""
    k = 30.0 / fps
    steps = []
    steps += [(0.03 * k, 0, 0, -0.35 / (30 / k), 0, 0)] * int(30 / k)           # поворот на c₁ = (−4, 0, 0)
    steps += [(0.11 * k, 0, 0, 0, 0, 0)] * int(150 / k)                          # полёт сквозь горловину
    steps += [(0.02 * k, 0, 0, np.pi / (60 / k), 0, 0.004 * k)] * int(60 / k)    # разворот на 180°
    steps += [(-0.03 * k, 0, 0, 0, 0, -0.004 * k)] * int(60 / k)                 # отлёт назад, вид на горловину
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cube", type=int, default=512)
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    out = new_run(5, "video", dict(vars(a), gpupt=G.GPUPT().__dict__, slangc=gpu.slangc_version()))
    sc, c0, speed = G.app_scene("two_wormholes", PT.procedural_env(a.cube))
    cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], fov=75, width=W, height=H)
    steps = path_two_wormholes(a.fps)
    script = []
    for s in steps:
        script.append(list(s))
        script += [[0, 0, 0, 0, 0, 0]] * (a.spp - 1)
    np.savetxt(out / "camera_path.txt", np.array(steps), fmt="%.6f", header="a b c yaw pitch roll (за кадр ролика)")
    ffmpeg = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "gpu" / "build") as td:
        td = Path(td)
        np.savetxt(td / "script.txt", np.array(script), fmt="%.6f")
        raw = td / "frames.raw"
        graph = G.app_graph(sc, cam, td, speed=speed, hidden=True, bench=(td / "script.txt", td / "frames.json"),
                            record=(raw, a.spp))
        t0 = time.time()
        r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu app: {r.stdout}\n{r.stderr}")
        wall = time.time() - t0
        nfr = raw.stat().st_size // (4 * W * H)
        for i in [0, nfr // 4, nfr // 2, 3 * nfr // 4, nfr - 1]:
            u = np.fromfile(raw, dtype=np.uint32, count=W * H, offset=4 * W * H * i).reshape(H, W)
            plt.imsave(out / f"frame_{i:04d}.png", np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8))
        log = json.loads((td / "frames.json").read_text(encoding="utf-8"))
        tot = [f["total_ms"] for f in log]
        info = dict(video_frames=int(nfr), duration_s=nfr / a.fps, wall_s=wall,
                    s_per_video_frame=wall / max(nfr, 1), app_frames=len(log),
                    median_frame_ms_moving=float(np.median([f["total_ms"] for f in log if f["moving"]])),
                    median_frame_ms_static=float(np.median([f["total_ms"] for f in log if not f["moving"]])),
                    sum_frame_ms=float(np.sum(tot)))
        if ffmpeg:
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}", "-r", str(a.fps),
                   "-i", str(raw), "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", str(out / "video.mp4")]
            subprocess.run(cmd, check=True)
            info["mp4_bytes"] = (out / "video.mp4").stat().st_size
        else:
            info["mp4_bytes"] = None
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
