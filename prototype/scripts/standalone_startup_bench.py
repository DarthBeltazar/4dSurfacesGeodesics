"""DD-010, шаг 3: окно без Python — время запуска, смены разрешения и структурной перегенерации.

- холодный старт: пустые кэши SPIR-V и VkPipelineCache, многообразие с возмущённым b (новый SPIR-V — промах и кэша
  драйвера); тёплый: всё в кэшах. Время — от запуска процесса до выхода после 3 кадров (скрытое окно, bench),
  плюс разбивка standalone_build (сцена и фон, граф, компиляция);
- для сравнения — путь через Python: `app.py --graph-only` (кэш фона и SPIR-V тёплые) + `geo4d_gpu app`;
- смена разрешения (resize 1280×720 → 1920×1080) и выход из режима редактирования с новой структурой объектов
  (новый источник; кэш SPIR-V пустой) — строки graph_switch (перегенерация и смена графа).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/standalone_startup_bench.py
Результат: results/stage-5/<дата>_standalone_startup/."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF      # noqa: E402
from geo4d.results import new_run                                          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = "1280x720"
REPEAT = 3


def script(td, frames, extra=""):
    np.savetxt(td / "script.txt", np.zeros((frames, 6)), fmt="%.6f")
    (td / "extra.txt").write_text(f"hidden\nbench {gpu.qpath(td / 'script.txt')} {gpu.qpath(td / 'f.json')}\n" + extra,
                                  encoding="utf-8")


def standalone(td, scene, env, frames=3, extra="", spv=None, res=RES):
    td.mkdir(parents=True, exist_ok=True)
    script(td, frames, extra)
    cmd = [str(gpu.EXE), str(scene), "--res", res, "--work-dir", str(td / "work"), "--extra", str(td / "extra.txt")]
    if spv:
        cmd += ["--spv-dir", str(spv)]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env)
    wall = time.perf_counter() - t
    assert r.returncode == 0, r.stdout + r.stderr
    tag = lambda k: [json.loads(s.split(" ", 1)[1]) for s in r.stdout.splitlines() if s.startswith(k + " ")]
    return dict(wall_s=wall, build=tag("standalone_build"), switch=tag("graph_switch"))


def perturbed(td, k):
    d = json.loads((ROOT / "scenes" / "two_wormholes.json").read_text(encoding="utf-8"))
    d["manifold"]["params"]["b"] = 1.0 + (int(time.time() * 1000) % 100000 + 1 + 1000 * k) * 1e-9
    p = td / f"scene_{k}.json"
    SF.save(p, d)
    return p


def main():
    out = new_run(5, "standalone_startup", dict(res=RES, repeat=REPEAT, slangc=gpu.slangc_version()))
    base = {k: v for k, v in os.environ.items() if k != "GEO4D_SLANG_OPT"}
    rep = dict(cold=[], warm=[], python_warm=[], resize=[], structural=[])
    scene = ROOT / "scenes" / "two_wormholes.json"
    with tempfile.TemporaryDirectory() as tdir:
        td = Path(tdir)
        for k in range(REPEAT):                    # холодный старт
            env = dict(base, GEO4D_CACHE_DIR=str(td / f"cache_cold_{k}"))
            rep["cold"].append(standalone(td / f"cold_{k}", perturbed(td, k), env, spv=td / f"spv_cold_{k}"))
        warm_env = dict(base, GEO4D_CACHE_DIR=str(td / "cache_warm"))
        standalone(td / "warm_prime", scene, warm_env)
        for k in range(REPEAT):                    # тёплый старт
            rep["warm"].append(standalone(td / f"warm_{k}", scene, warm_env))
        for k in range(REPEAT):                    # Python: граф (кэш фона и SPIR-V тёплые) + окно
            d = td / f"py_{k}"
            t = time.perf_counter()
            r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "prototype/scripts/app.py"), "--scene", str(scene),
                                "--res", RES, "--graph-only", str(d)], capture_output=True, text=True, encoding="utf-8",
                               env=warm_env)
            assert r.returncode == 0, r.stderr
            t_graph = time.perf_counter() - t
            script(d, 3)
            g = d / "app_graph.txt"
            g.write_text(g.read_text(encoding="utf-8") + (d / "extra.txt").read_text(encoding="utf-8"), encoding="utf-8")
            r = subprocess.run([str(gpu.EXE), "app", str(g)], capture_output=True, text=True, encoding="utf-8", env=warm_env)
            assert r.returncode == 0, r.stdout + r.stderr
            rep["python_warm"].append(dict(wall_s=time.perf_counter() - t, graph_s=t_graph))
        for k in range(REPEAT):                    # смена разрешения
            rep["resize"].append(standalone(td / f"resize_{k}", scene, warm_env, frames=8, extra="resize 3 1920 1080\n"))
        d0 = json.loads(scene.read_text(encoding="utf-8"))
        d0["objects"].append(dict(name="new_light", kind="ball4", center=[0.0, 3.0, 1.0, 1.5], r=0.3,
                                  material=dict(kind="emit", emit=[5.0, 20.0, 5.0])))
        sc1, _, _ = SF.build(d0, PT.procedural_env(8))
        np.save(td / "rows.npy", G.object_rows(sc1))
        for k in range(REPEAT):                    # выход из режима редактирования с новой структурой; кэш SPIR-V пуст
            rep["structural"].append(standalone(
                td / f"struct_{k}", scene, warm_env, frames=10, spv=td / f"spv_struct_{k}",
                extra=f"editmode 2 on\nedit 3 {gpu.qpath(td / 'rows.npy')}\neditmode 5 off\n"))
    med = lambda v: float(np.median(v))
    summ = dict(
        cold_wall_s=med([r["wall_s"] for r in rep["cold"]]),
        cold_compile_ms=med([r["build"][0]["compile_ms"] for r in rep["cold"]]),
        warm_wall_s=med([r["wall_s"] for r in rep["warm"]]),
        warm_build_ms=med([sum(r["build"][0][k] for k in ("scene_env_ms", "graph_ms", "compile_ms")) for r in rep["warm"]]),
        python_warm_wall_s=med([r["wall_s"] for r in rep["python_warm"]]),
        python_warm_graph_s=med([r["graph_s"] for r in rep["python_warm"]]),
        resize_rebuild_ms=med([r["switch"][0]["rebuild_ms"] for r in rep["resize"]]),
        resize_switch_ms=med([r["switch"][0]["switch_ms"] for r in rep["resize"]]),
        structural_rebuild_ms=med([r["switch"][-1]["rebuild_ms"] for r in rep["structural"]]),
        structural_switch_ms=med([r["switch"][-1]["switch_ms"] for r in rep["structural"]]),
    )
    (out / "runs.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
