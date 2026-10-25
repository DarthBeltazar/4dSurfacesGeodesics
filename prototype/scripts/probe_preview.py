"""Проба режима предпросмотра (concept-standalone-nodes.md, п. 3; журнал 2026-10-25-02): кадр из одной первичной
геодезической на пиксель (ядро pt_preview: Jet7, без полей Якоби, отскоков и фолбэка, затенение «фонарь у камеры»).

Для сцен two_wormholes и S3 и нескольких разрешений: время ядра с f, зашитой в шейдер (как быстрый граф), и с f из
интерпретатора байткода (SCENE_DYNAMIC + MF_PROG, geo4d.bytecode). Плюс время сборки самого ядра pt_preview при
промахе кэшей (новое b): slangc и драйвер (GEO4D_PIPE_TIME). Кадры 720p сохраняются PNG для сравнения на глаз.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/probe_preview.py
Результат: results/stage-5/<дата>_probe_preview/."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import procedural_env_cached                                     # noqa: E402
from geo4d import bytecode, gpu, gpu_pt as G, scenefile as SF              # noqa: E402
from geo4d.camera import make_camera                                       # noqa: E402
from geo4d.results import new_run                                          # noqa: E402

RES = [(640, 360), (960, 540), (1280, 720), (1920, 1080)]
REPEAT = 20


def program(sc):
    prm = sc.shape.params
    if sc.shape.name == "two_compact_wh":
        return bytecode.two_compact_program(prm["c1"], prm["c2"], prm["b"], prm["rho0"], prm["W"])
    return bytecode.s3_program(prm["R"])


def preview(sc, cam, interp, png=None, s_max=None):
    """Время ядра pt_preview (мс, медиана по REPEAT прогонам) и кадр Present (H, W, 4) uint8. s_max — предел длины
    луча предпросмотра (None — как у сцены)."""
    from geo4d.gpu import cube_for_gpu
    from geo4d.gpu_modea import Detector
    N = cam.width * cam.height
    inc = G.write_header(sc, interp)
    defs = {**G._defs(sc, 0, interp), **({"MF_PROG": 1, "PROG_NREG": 8} if interp else {})}
    pv = gpu.compile_shader("pathtrace.slang", "pt_preview", defs, include=inc)
    pr = gpu.compile_shader("pathtrace.slang", "pt_present", defs, include=inc)
    cube = cube_for_gpu(np.asarray(sc.mips[0]))
    envb, env_n = G.env_buffer(sc)
    camb = G.pt_camera_buffer(sc, cam, 2.0, 0.0, Detector(), 0, cube.shape[2])
    p = G.GPUPT(**({"s_max": s_max} if s_max else {}))
    camb[len(camb) - G.MF_ROWS + 5, 0] = min(p.s_max, sc.meta.get("s_max", p.s_max))      # как manifold_rows(p)
    if interp:
        camb = np.concatenate([camb, program(sc)])
    with tempfile.TemporaryDirectory() as td:
        td = gpu.QPath(td)
        np.save(td / "cam.npy", gpu.f32(camb))
        np.save(td / "env.npy", envb)
        np.save(td / "cube.npy", gpu.f32(cube))
        L = [f"buffer cam in {td / 'cam.npy'}", f"buffer Env in {td / 'env.npy'}", f"cube Cube {td / 'cube.npy'} rgba16f 16"]
        for name, b in [("Acc", 16 * N), ("Acc2", 16 * N), ("Stats", 16 * N), ("Pix", 8 * N), ("Alb", 16 * N),
                        ("Nrm", 16 * N), ("Dir", 16 * N), ("Dd", 64 * N), ("Status", 4 * N), ("Class", 4 * N),
                        ("Flag", 4 * N), ("Cnt", 16), ("Args", 32), ("Frame", 16 * N), ("AlbF", 16 * N), ("NrmF", 16 * N),
                        ("Den", 16 * N), ("Present", 4 * N), ("Color", 16 * N)]:
            L.append(f"buffer {name} zero {b}")
        g = gpu.groups_for(N)[0]
        L += [f"pipeline preview {pv} pt_preview {G.PT_BIND}", f"pipeline present {pr} pt_present {G.PT_BIND}",
              G._push_line("preview", G._push(sc, p, N, env_n)),
              G._push_line("present", G._push(sc, G.GPUPT(), N, env_n, use_den=0, bgra=0)),
              "mark start", f"dispatch preview {g} 1 1", "mark preview", f"dispatch present {g} 1 1", "mark present",
              f"repeat {REPEAT}", f"output Present {td / 'present.npy'} u4 {N}", f"report {td / 'report.json'}"]
        (td / "graph.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
        r = subprocess.run([str(gpu.EXE), "graph", str(td / "graph.txt")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu graph: {r.stdout}\n{r.stderr}")
        rep = json.loads((td / "report.json").read_text(encoding="utf-8"))
        img = np.load(td / "present.npy").view(np.uint8).reshape(cam.height, cam.width, 4)
    ms = float(np.median([run["preview"] for run in rep["runs"]])) / 1e6
    if png is not None:
        import matplotlib.pyplot as plt
        plt.imsave(png, img[:, :, :3])
    return ms, img


def compile_cold(mips, td, opt):
    """Сборка pt_preview при промахе кэшей (новое b): slangc отдельным процессом и драйвер (по строке pipe_time)."""
    d = json.loads(SF.resolve("two_wormholes").read_text(encoding="utf-8"))
    d["manifold"]["params"]["b"] = 1.0 + float(np.random.default_rng(time.time_ns()).integers(1, 10 ** 6)) * 1e-9
    sp = td / "scene.json"
    sp.write_text(json.dumps(d), encoding="utf-8")
    sc, _, _ = SF.load(sp, mips)
    inc = G.write_header(sc, False)
    spv = td / f"pv{opt}.spv"
    cmd = [str(gpu.SLANGC), str(gpu.GPU / "shaders" / "pathtrace.slang"), "-target", "spirv", "-profile", "spirv_1_6",
           "-entry", "pt_preview", "-fvk-use-entrypoint-name", opt, "-o", str(spv), "-I", str(inc)]
    cmd += [f"-D{k}={v}" for k, v in G._defs(sc, 0, False).items()]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t_slangc = time.perf_counter() - t
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    job = td / "job.txt"
    job.write_text(f"spv {spv}
entry pt_preview
groups 0 1 1
repeat 1
"
                   + "".join(f"buf {b} out {td / f'o{b}.npy'} f4 4 4
" for b in range(G.PT_BIND.count(" ") + 1)),
                   encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "run", str(job)], capture_output=True, text=True,
                       env=dict(os.environ, GEO4D_PIPE_TIME="1"))
    pipe = [float(line.split()[3]) for line in r.stderr.splitlines() if line.startswith("pipe_time pt_preview")]
    return dict(slangc_s=t_slangc, pipe_ms=pipe[0] if pipe else None, run_ok=r.returncode == 0,
                spv_kb=spv.stat().st_size // 1024)


def main():
    mips = procedural_env_cached(512)
    out = new_run(5, "probe_preview", dict(scenes=["two_wormholes", "S3"], res=RES, repeat=REPEAT,
                                           slangc=gpu.slangc_version()))
    rep = {}
    for name in ["two_wormholes", "S3"]:
        sc, c0, _ = SF.load(SF.resolve(name), mips)
        for W, H in RES:
            cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], c0["right"], fov=c0["fov"], width=W, height=H)
            row = {}
            for interp in (False, True):
                png = out / f"{name}_{W}x{H}_{'interp' if interp else 'native'}.png" if W == 1280 else None
                row["interp" if interp else "native"], img = preview(sc, cam, interp, png)
            if name == "S3":     # предел длины — один оборот 2πR вместо 8πR сцены
                for interp in (False, True):
                    png = out / f"{name}_{W}x{H}_{'interp' if interp else 'native'}_s2pi.png" if W == 1280 else None
                    row[("interp" if interp else "native") + "_s2pi"], _ = preview(
                        sc, cam, interp, png, s_max=2 * np.pi * sc.shape.params["R"])
            rep[f"{name}_{W}x{H}"] = row
            print(name, f"{W}x{H}", {k: round(v, 2) for k, v in row.items()}, f"×{row['interp'] / row['native']:.1f}")
    with tempfile.TemporaryDirectory() as td:
        rep["compile_cold"] = {opt: compile_cold(mips, Path(td), opt) for opt in ("-O2", "-O0")}
    print("сборка pt_preview при промахе:", rep["compile_cold"])
    (out / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
