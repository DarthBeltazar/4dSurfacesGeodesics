"""DD-010, шаг 2: время сборки всех шейдеров окна (быстрый граф + граф режима редактирования) через Slang API в
процессе (`geo4d_gpu export-graph --compile`) при пустом кэше, по числу потоков. Каждая сборка — в новый каталог
кэша; повторы — чтобы видеть разброс. Сравнение — probe_compile (slangc.exe процессами).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/standalone_compile_bench.py
Результат: results/stage-5/<дата>_standalone_compile/."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from geo4d import gpu
from geo4d.results import new_run

ROOT = Path(__file__).resolve().parents[2]
THREADS = [1, 4, 8, 16]
SCENES = ["two_wormholes", "S3"]
REPEAT = 3


def main():
    out = new_run(5, "standalone_compile", dict(threads=THREADS, scenes=SCENES, repeat=REPEAT, res="1280x720",
                                                slangc=gpu.slangc_version(), cpus=os.cpu_count()))
    env = {k: v for k, v in os.environ.items() if k != "GEO4D_SLANG_OPT"}
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for scene in SCENES:
            for t in THREADS:
                for k in range(REPEAT):
                    spv = td / f"spv_{scene}_{t}_{k}"
                    r = subprocess.run([str(gpu.EXE), "export-graph", str(ROOT / "scenes" / f"{scene}.json"),
                                        str(td / "g"), "--compile", "--threads", str(t), "--spv-dir", str(spv)],
                                       capture_output=True, text=True, encoding="utf-8", env=env)
                    assert r.returncode == 0, r.stderr
                    j = json.loads(r.stdout)
                    rows.append(dict(scene=scene, threads=t, run=k, compile_ms=j["compile_ms"], shaders=j["shaders"],
                                     env_ms=j["env_ms"], graph_ms=j["graph_ms"], session_s=j["session_s"],
                                     kernels=j["kernels"]))
                    print(scene, t, k, round(j["compile_ms"]), "мс")
    summ = {}
    for scene in SCENES:
        for t in THREADS:
            v = sorted(r["compile_ms"] for r in rows if r["scene"] == scene and r["threads"] == t)
            summ[f"{scene}_{t}"] = dict(median_ms=v[len(v) // 2], min_ms=v[0], max_ms=v[-1])
    (out / "runs.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
