"""Проба быстрой перекомпиляции (concept-standalone-nodes.md, п. 3; журнал 2026-10-25-02): сколько стоит собрать
шейдеры окна заново после структурной правки многообразия — альтернатива интерпретатору байткода.

Сцена two_wormholes с возмущённым параметром b (новый SPIR-V — промах и кэша slangc, и кэша драйвера). Для каждого
уровня оптимизации slangc: время каждого зависящего от многообразия ядра, затем сборка конвейеров драйвером
(geo4d_gpu app, GEO4D_PIPE_TIME) при промахе и при повторном запуске. Отдельно — запуск slangc на пустом ядре
(цена процесса и базового модуля, которую уберёт Slang внутри хоста).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/probe_compile.py
Результат: results/stage-5/<дата>_probe_compile/."""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import procedural_env_cached                                     # noqa: E402
from geo4d import gpu, gpu_pt as G, scenefile as SF                        # noqa: E402
from geo4d.camera import make_camera                                       # noqa: E402
from geo4d.results import new_run                                          # noqa: E402

RES = (1280, 720)


def kernels(sc):
    """Ядра окна, зависящие от многообразия (как в gpu_pt.app_graph): (исходник, вход, дефайны, include)."""
    inc = G.write_header(sc, False)
    df = G._defs(sc, 0, False)
    return [("pathtrace.slang", "pt_camera", G._defs(sc, 4, False), inc),
            ("pathtrace.slang", "pt_sample", {**df, "PT_STAGE": 1}, inc),
            ("pathtrace.slang", "cam_step", {**df, "CAM_STEP": 1}, inc),
            ("pathtrace.slang", "pt_snap", {**df, "CAM_STEP": 1}, inc),
            ("modea.slang", "modea_shade", {**df, "KERNEL": 2}, None),
            ("modea.slang", "modea_detect", {**df, "KERNEL": 3}, None)]


def slangc(src, entry, defs, inc, opt, out):
    cmd = [str(gpu.SLANGC), str(gpu.GPU / "shaders" / src), "-target", "spirv", "-profile", "spirv_1_6", "-entry", entry,
           "-fvk-use-entrypoint-name", opt, "-o", str(out)] + [f"-D{k}={v}" for k, v in defs.items()]
    if inc:
        cmd += ["-I", str(inc)]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return dt


def pipe_times(graph):
    """Время сборки конвейеров драйвером по строкам pipe_time хоста: {вход: мс} (у входа может быть несколько)."""
    env = dict(os.environ, GEO4D_PIPE_TIME="1")
    r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    res = []
    for line in r.stderr.splitlines():
        if line.startswith("pipe_time "):
            _, e, size, ms = line.split()
            res.append((e, int(size), float(ms)))
    return res


def main():
    mips = procedural_env_cached(512)
    out = new_run(5, "probe_compile", dict(scene="two_wormholes", res=RES, opts=["-O0", "-O1", "-O2"],
                                           slangc=gpu.slangc_version()))
    rng = np.random.default_rng(int(time.time()))
    td = Path(tempfile.mkdtemp())
    t_empty = [slangc("cube_probe.slang", "cube_probe", {}, None, "-O2", td / "e.spv") for _ in range(3)]
    rep = dict(slangc_empty_s=t_empty, opts={})
    for opt in ["-O0", "-O1", "-O2"]:
        path = SF.resolve("two_wormholes")
        d = json.loads(path.read_text(encoding="utf-8"))
        d["manifold"]["params"]["b"] = 1.0 + float(rng.integers(1, 10 ** 6)) * 1e-9      # новый SPIR-V
        sp = td / f"scene{opt}.json"
        sp.write_text(json.dumps(d), encoding="utf-8")
        sc, c0, speed = SF.load(sp, mips)
        kt = {e: slangc(src, e, df, inc, opt, td / f"{e}{opt}.spv") for src, e, df, inc in kernels(sc)}
        # те же ядра параллельно (по процессу на ядро) — после нового возмущения b, чтобы снова был промах
        d["manifold"]["params"]["b"] += 1e-9
        sp.write_text(json.dumps(d), encoding="utf-8")
        scp, _, _ = SF.load(sp, mips)
        t = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda k: slangc(*k, opt, td / f"{k[1]}{opt}_par.spv"), kernels(scp)))
        t_par = time.perf_counter() - t
        # граф окна в режиме замера (3 кадра покоя) — шейдеры через compile_shader с тем же уровнем оптимизации
        os.environ["GEO4D_SLANG_OPT"] = opt
        cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], c0["right"], fov=c0["fov"], width=RES[0], height=RES[1])
        gd = td / f"g{opt}"
        gd.mkdir()
        np.savetxt(gd / "script.txt", np.zeros((3, 6)), fmt="%.6f")
        graph = G.app_graph(sc, cam, gd, speed=speed, hidden=True, bench=(gd / "script.txt", gd / "frames.json"),
                            dyn_mips=mips)
        cold, warm = pipe_times(graph), pipe_times(graph)
        rep["opts"][opt] = dict(b=d["manifold"]["params"]["b"], slangc_s=kt, slangc_sum_s=sum(kt.values()),
                                slangc_parallel_s=t_par,
                                pipe_cold_ms=cold, pipe_warm_ms=warm,
                                pipe_cold_sum_ms=sum(c[2] for c in cold), pipe_warm_sum_ms=sum(c[2] for c in warm))
        print(opt, f"slangc {sum(kt.values()):.2f} s (параллельно {t_par:.2f} s)", {k: round(v, 2) for k, v in kt.items()},
              f"конвейеры: промах {rep['opts'][opt]['pipe_cold_sum_ms']:.0f} мс, кэш {rep['opts'][opt]['pipe_warm_sum_ms']:.0f} мс")
    os.environ.pop("GEO4D_SLANG_OPT", None)
    print("slangc на пустом ядре:", [round(t, 3) for t in t_empty])
    (out / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
