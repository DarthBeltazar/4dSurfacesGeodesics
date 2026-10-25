"""DD-011, шаг 5б: время кадра предпросмотра (нативно и на интерпретаторе) и задержка структурной правки.

- кадр предпросмотра (метка preview журнала bench): быстрый граф с флажком предпросмотра (`previewmode 0 on`) и граф
  предпросмотра на интерпретаторе (`--graph-only`, затем `geo4d_gpu app app_graph_prev.txt`); 60 кадров полёта
  (поворот камеры — кадр каждый раз заново), медиана по кадрам с 5-го;
- структурная правка (setparam b горловины, число многообразия запечено): переход на граф предпросмотра
  (preview_switch_ms), сборка быстрого графа в фоне (rebuild_ms; тёплый — SPIR-V этого b уже в кэше, холодный — новый
  b и пустой каталог SPIR-V) и смена графа (switch_ms).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/preview_bench.py
Результат: results/stage-5/<дата>_preview_bench/."""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo4d import gpu                                                       # noqa: E402
from geo4d.results import new_run                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENES = ["scenes/two_wormholes.json", "scenes/three_wormholes.g4d", "scenes/csg_objects.g4d"]
RESES = ["1280x720", "1920x1080"]
FRAMES = 60


def extra(td, frames, more=""):
    td.mkdir(parents=True, exist_ok=True)
    np.savetxt(td / "script.txt", np.array([[0, 0, 0, 0.004, 0.0, 0]] * frames), fmt="%.6f")
    return f"hidden\nbench {gpu.qpath(td / 'script.txt')} {gpu.qpath(td / 'f.json')}\n" + more


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return r.stdout


def median_preview(td):
    fr = json.loads((td / "f.json").read_text(encoding="utf-8"))
    return float(np.median([f["preview"] for f in fr[5:] if "preview" in f]))


def native(td, scene, res):
    (td / "extra.txt").write_text(extra(td, FRAMES, "previewmode 0 on\n"), encoding="utf-8")
    run([gpu.EXE, ROOT / scene, "--res", res, "--work-dir", td / "w", "--extra", td / "extra.txt"])
    return median_preview(td)


def interp(td, scene, res):
    g = td / "g"
    run([gpu.EXE, ROOT / scene, "--res", res, "--graph-only", g])
    prev = g / ("app_graph_prev_o8.txt" if "csg" in scene else "app_graph_prev_p8.txt")   # программы этих сцен — ≤ 8 регистров
    prev.write_text(prev.read_text(encoding="utf-8") + extra(td, FRAMES), encoding="utf-8")
    run([gpu.EXE, "app", prev])
    return median_preview(td)


def edit(td, b, spv=None):
    (td / "extra.txt").write_text(extra(td, 30, f"previewmode 0 on\nsetparam 10 t1 b {b}\n"), encoding="utf-8")
    cmd = [gpu.EXE, ROOT / "scenes/three_wormholes.g4d", "--res", "1280x720", "--work-dir", td / "w", "--extra", td / "extra.txt"]
    out = run(cmd + (["--spv-dir", spv] if spv else []))
    sw = re.search(r'graph_switch \{"kind": 3, "rebuild_ms": ([\d.]+), "switch_ms": ([\d.]+)\}', out)
    fr = json.loads((td / "f.json").read_text(encoding="utf-8"))
    return dict(preview_switch_ms=float(re.search(r"preview_switch_ms ([\d.]+)", out).group(1)),
                rebuild_ms=float(sw.group(1)), switch_ms=float(sw.group(2)), edit_frame_total_ms=fr[10]["total_ms"],
                preview_frame_ms=fr[10].get("preview"))


def main():
    out = new_run(5, "preview_bench", dict(scenes=SCENES, reses=RESES, frames=FRAMES))
    res = {"frame_ms": {}, "edit": {}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for scene in SCENES:
            for r in RESES:
                key = f"{Path(scene).stem}_{r}"
                res["frame_ms"][key] = dict(native=native(tmp / key / "n", scene, r), interp=interp(tmp / key / "i", scene, r))
                print(key, res["frame_ms"][key], flush=True)
        edit(tmp / "warmup", 0.55)                 # SPIR-V для b = 0.55 — в кэш
        res["edit"]["warm"] = edit(tmp / "warm", 0.55)
        res["edit"]["cold"] = edit(tmp / "cold", 0.5 + 0.1 * np.random.default_rng().random(), spv=tmp / "spv")
        print(res["edit"], flush=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
