"""DD-009: время открытия сцены с другим многообразием в окне (директива loadscene — код кнопки «Открыть») и цена
режима редактирования с многообразием из строк: исходная сцена two_wormholes, на кадре 5 — открыть сцену-цель
(сразу, шейдеры SCENE_DYNAMIC), кадры 6…frames−1 — покой в режиме редактирования, затем выход (перегенерация быстрых
шейдеров в Python, ожидание) и покой на быстрых.

    python prototype/scripts/topology_switch_time.py --res 1280x720 --frames 30"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF
from geo4d.camera import make_camera
from geo4d.results import new_run

APP = Path(__file__).resolve().parent / "app.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--targets", default="S3,ellis,tiger,hyperplane")
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    out = new_run(5, f"topology_switch_{W}x{H}" + (f"_{a.tag}" if a.tag else ""), vars(a))
    sys.path.insert(0, str(APP.parent))
    from app import procedural_env_cached
    mips = procedural_env_cached(a.cube)
    src = SF.resolve("two_wormholes")
    sc, c, speed = SF.load(src, mips)
    cam = make_camera(sc.shape, c["x"], c["f"], c["up"], fov=c["fov"], width=W, height=H)
    F, res = a.frames, {}
    for tgt in a.targets.split(","):
        td = out / f"run_{tgt}"
        td.mkdir(exist_ok=True)
        np.savetxt(td / "script.txt", np.zeros((3 * F, 6)), fmt="%.6f")
        ed = dict(scene=src.as_posix(), work_dir=td.as_posix(), fov=c["fov"], env=None, python=Path(sys.executable).as_posix(),
                  script=APP.as_posix(), args=["--res", a.res, "--cube", str(a.cube)])
        graph = G.app_graph(sc, cam, td, speed=speed, hidden=True, bench=(td / "script.txt", td / "frames.json"),
                            editor=ed, dyn_mips=mips)
        graph.write_text(graph.read_text(encoding="utf-8") + f"loadscene 5 {SF.resolve(tgt).as_posix()}\n"
                         f"editmode {2 * F} off\n", encoding="utf-8")
        r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(r.stdout + r.stderr)
        fr = json.loads((td / "frames.json").read_text(encoding="utf-8"))
        apply_ms = [float(t.split()[1]) for t in r.stdout.splitlines() if t.startswith("apply_scene_ms")]
        med = lambda rows: float(np.median([f["total_ms"] for f in rows]))
        res[tgt] = dict(apply_scene_ms=apply_ms[0], edit_mode_static_ms=med(fr[F:2 * F - 2]),
                        exit_frame_ms=fr[2 * F]["total_ms"], baked_static_ms=med(fr[2 * F + 5:]))
        for f in td.glob("*.npy"):
            f.unlink()                                  # входы графов (кубмапа — десятки МБ)
        for d in td.glob("edit_*"):
            if d.is_dir():
                for f in d.glob("*.npy"):
                    f.unlink()
    (out / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
