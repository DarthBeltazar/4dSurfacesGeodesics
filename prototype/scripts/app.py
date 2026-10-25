"""Этап 5 (DD-007) / DD-010: окно — обёртка над `geo4d_gpu [сцена.json]` (граф, шейдеры и перегенерации — в C++,
без Python). Python-генерация графа (как до DD-010) — флагом --python-graph: эталон для тестов и проба интерпретатора.

    python prototype/scripts/app.py --scene two_wormholes --res 1280x720
    python prototype/scripts/app.py --scene two_wormholes --res 1920x1080 --fullscreen   # на весь экран (F11)
    python prototype/scripts/app.py --scene S3 --res 960x540
    python prototype/scripts/app.py --bench --scene two_wormholes --res 1920x1080   # замер: сценарий, скрытое окно
    python prototype/scripts/app.py --graph-only gpu/build-debug/app/two_wormholes  # только граф — запуск хоста из IDE
То же без Python: gpu/build/geo4d_gpu[.exe] scenes/two_wormholes.json --res 1920x1080 --fullscreen

Управление: W/S, A/D — вперёд/вбок; Q/E — вниз/вверх; ЛКМ — захват мыши (взгляд); стрелки; C/Z — крен; Shift — ×4;
R — денойзер вкл/выкл; F1 — панели редактора; Tab — режим редактирования; F11 — полноэкранный режим; Esc — выход.
--res — разрешение рендера; окно можно растянуть или развернуть на весь экран — кадр масштабируется с сохранением
пропорций; другое разрешение рендера (доля окна или фиксированное) выбирается в панели «Рендер» (DD-009). Сцены — файлы scenes/*.json (DD-008): two_wormholes,
ellis, S3, spheritorus, torisphere, tiger (T³), hyperplane, или путь к своему JSON. Редактор в окне правит объекты
сразу; другую сцену («Открыть») или другое многообразие («Применить») — тоже сразу, в режиме редактирования (шейдеры
с многообразием из строк, DD-009), а быстрые шейдеры собирает при выходе из него (в процессе, DD-010).

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

from geo4d import gpu, gpu_pt as G, pathtrace as PT, scenefile as SF
from geo4d.camera import make_camera


def procedural_env_cached(n):
    """Фон окна: тот же при каждой перегенерации графа (новое разрешение, структура объектов, многообразие), а его
    построение в JAX — ~8 с из ~10 (профиль, DD-009). Кэш на диске; ключ — исходники фона и размер грани."""
    import hashlib
    import inspect
    import jax.numpy as jnp
    from geo4d import cubemap, render
    src = inspect.getsource(cubemap) + inspect.getsource(render) + inspect.getsource(PT.procedural_env) + str(n)
    f = gpu.GPU / "build" / "cache" / f"procedural_env_{n}_{hashlib.sha256(src.encode()).hexdigest()[:16]}.npz"
    if f.exists():
        z = np.load(f)
        return [jnp.asarray(z[f"m{i}"]) for i in range(len(z.files))]
    mips = PT.procedural_env(n)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + f".{__import__('os').getpid()}.npz")        # параллельные запуски не портят файл
    np.savez(tmp, **{f"m{i}": np.asarray(m) for i, m in enumerate(mips)})
    tmp.replace(f)
    return mips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="two_wormholes", help="имя сцены из scenes/ или путь к JSON (DD-008)")
    ap.add_argument("--res", default="1280x720", help="разрешение рендера W×H")
    ap.add_argument("--fullscreen", action="store_true", help="полноэкранный режим при запуске (F11 — переключение)")
    ap.add_argument("--fov", type=float, default=0, help="угол обзора, ° (0 — из файла сцены)")
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--vsync", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--visible", action="store_true", help="замер в видимом окне (проверка пути на экран)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--tag", default="", help="суффикс имени каталога результатов замера")
    ap.add_argument("--edit-mode", action="store_true",
                    help="замер в режиме редактирования (DD-008): вход на кадре 1, выход за 3 кадра до конца; "
                         "в summary — время кадров переключения графа")
    ap.add_argument("--probe-interp", type=int, default=0, metavar="NREG",
                    help="проба интерпретатора байткода (concept-standalone-nodes.md): в графе режима редактирования f "
                         "двух горловин исполняется из программы geo4d.bytecode, NREG — размер массива регистров")
    ap.add_argument("--probe-variant", default="", help="вариант пробы: const — программа константным массивом в шейдере")
    ap.add_argument("--bake-params", action="store_true",
                    help="числа графа сцены — литералами в шейдерах, без слотов (DD-011 §5: замер цены слотов)")
    ap.add_argument("--manifold-slots", action="store_true",
                    help="числа многообразия — тоже слоты: ползунок без перекомпиляции, кадр дороже (DD-011 §5)")
    ap.add_argument("--dump", default="", help="кадры для сохранения через запятую (в каталог результатов замера)")
    ap.add_argument("--python-graph", action="store_true",
                    help="граф строит Python (как до DD-010), окно — `geo4d_gpu app <граф>`: эталон для тестов")
    ap.add_argument("--graph-only", metavar="DIR", default="",
                    help="записать граф и его входные данные в DIR и выйти (запуск/отладка `geo4d_gpu app DIR/app_graph.txt` из IDE)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.res.split("x"))
    scene_path = SF.resolve(a.scene).resolve()
    name = scene_path.stem
    speed = float(json.loads(scene_path.read_text(encoding="utf-8")).get("speed", 1.0))
    legacy = a.python_graph or a.probe_interp            # граф строит Python (до DD-010)
    if legacy:
        mips = procedural_env_cached(a.cube)
        sc, c0, speed = SF.load(scene_path, mips)
        cam_fov = a.fov or c0["fov"]
        cam = make_camera(sc.shape, c0["x"], c0["f"], c0["up"], c0["right"], fov=cam_fov, width=W, height=H)
    out = None
    switch = ()
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(a.graph_only).resolve() if a.graph_only else Path(tmp)
        td.mkdir(parents=True, exist_ok=True)
        bench = None
        if a.bench:
            from geo4d.results import new_run
            out = new_run(5, f"app_bench_{name}_{W}x{H}" + ("_visible" if a.visible else "") + (f"_{a.tag}" if a.tag else ""), dict(vars(a), gpupt=G.GPUPT().__dict__,
                                                                  slangc=gpu.slangc_version()))
            # сначала покой на стартовом виде (накопление), затем полёт
            script = [[0, 0, 0, 0, 0, 0]] * a.frames
            if name == "two_wormholes":     # к горловине c₁ = (−4, 0, 0): поворот налево, затем вперёд сквозь неё
                script += [[0, 0, 0, -0.0175, 0.0, 0]] * 20 + [[0.25, 0, 0, 0, 0.0, 0]] * (a.frames - 20)
            else:
                script += [[0.05 * speed, 0, 0, 0.005, 0.002, 0]] * a.frames
            np.savetxt(td / "script.txt", np.array(script), fmt="%.6f")
            bench = (td / "script.txt", out / "frames.json")
            if a.edit_mode:
                switch = (1, len(script) - 3)
        dumps = [(int(f), (out or td).as_posix()) for f in a.dump.split(",") if f] if a.bench else ()
        rebuild_args = ["--res", a.res, "--cube", str(a.cube)] + (["--vsync"] if a.vsync else [])
        if not legacy:
            # DD-010: всё в geo4d_gpu; сценарий замера — директивами через --extra (как ниже их дописывает Python)
            cmd = [str(gpu.EXE), str(scene_path)] + rebuild_args + (["--fov", str(a.fov)] if a.fov else []) +                 (["--bake-params"] if a.bake_params else []) + (["--manifold-slots"] if a.manifold_slots else [])
            if a.fullscreen and not a.bench:
                cmd.append("--fullscreen")
            if a.graph_only:
                r = subprocess.run(cmd + ["--graph-only", str(td)], capture_output=True, text=True, encoding="utf-8")
                if r.returncode != 0:
                    raise RuntimeError(f"geo4d_gpu: {r.stdout}\n{r.stderr}")
                # перегенерация в хосте `geo4d_gpu app <граф>` (отладка, тесты окна) — этим же скриптом
                q = gpu.qpath
                line = " ".join(["rebuild", q(Path(sys.executable).as_posix()), q(Path(__file__).resolve().as_posix())]
                                + rebuild_args)
                for g in ("app_graph.txt", "app_graph_dyn.txt"):       # граф режима редактирования — только у прежнего кодгена
                    if (td / g).exists():
                        (td / g).write_text((td / g).read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
                print(f"граф: {td / 'app_graph.txt'}\nзапуск: {gpu.EXE} app {td / 'app_graph.txt'}")
                return
            extra = []
            if a.bench:
                extra += ([] if a.visible else ["hidden"]) + [f"bench {gpu.qpath(bench[0])} {gpu.qpath(bench[1])}"]
                extra += [f"dump {k} {gpu.qpath(d)}" for k, d in dumps]
                if switch:
                    extra += [f"editmode {switch[0]} on", f"editmode {switch[1]} off"]
            if extra:
                (td / "extra.txt").write_text("\n".join(extra) + "\n", encoding="utf-8")
                cmd += ["--extra", str(td / "extra.txt")]
            r = subprocess.run(cmd + ["--work-dir", str(td / "work")], capture_output=a.bench, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise RuntimeError(f"geo4d_gpu: {r.stdout}\n{r.stderr}")
        else:
            editor = None
            if not a.bench or a.edit_mode:          # редактор сцен (DD-008): окно перегенерирует граф этим же скриптом
                editor = dict(scene=scene_path.as_posix(), work_dir=td.as_posix(), fov=cam_fov,
                              env=json.loads(scene_path.read_text(encoding="utf-8")).get("env"),
                              python=Path(sys.executable).as_posix(), script=Path(__file__).resolve().as_posix(),
                              args=rebuild_args + ["--python-graph"])
            prog = None
            if a.probe_interp:
                from geo4d import bytecode
                prm = sc.shape.params
                if sc.shape.name == "two_compact_wh":
                    rows = bytecode.two_compact_program(prm["c1"], prm["c2"], prm["b"], prm["rho0"], prm["W"])
                elif sc.shape.name == "S3":
                    rows = bytecode.s3_program(prm["R"])
                else:
                    raise SystemExit("проба интерпретатора — только для двух горловин и S³")
                prog = (rows, a.probe_interp, a.probe_variant)
                assert a.probe_interp >= prog[0][0, 1], f"программе нужно {int(prog[0][0, 1])} регистров"
            graph = G.app_graph(sc, cam, td, speed=speed, hidden=a.bench and not a.visible, vsync=a.vsync, bench=bench,
                                dumps=dumps, editor=editor, fullscreen=a.fullscreen and not a.bench, dyn_mips=mips,
                                prog=prog)
            if switch:
                graph.write_text(graph.read_text(encoding="utf-8") + f"editmode {switch[0]} on\neditmode {switch[1]} off\n",
                                 encoding="utf-8")
            if a.graph_only:
                print(f"граф: {graph}\nзапуск: {gpu.EXE} app {graph}")
                return
            r = subprocess.run([str(gpu.EXE), "app", str(graph)], capture_output=a.bench, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"geo4d_gpu app: {r.stdout}\n{r.stderr}")
    if out is not None:
        fr = json.loads((out / "frames.json").read_text(encoding="utf-8"))
        mv = [f for f in fr[1:] if f["moving"] and f["frame"] not in switch][5:]
        st = [f for f in fr if not f["moving"] and f["frame"] not in switch][5:]
        keys = ["total_ms", "graph_ms", "denoise_ms", "post_present_ms", "camera", "lists", "pt", "fallback", "compose"]
        summ = {}
        for name, rows in [("moving", mv), ("static", st)]:
            summ[name] = {k: float(np.median([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
            summ[name]["frames"] = len(rows)
        if switch:               # кадры смены графа: вход в режим редактирования и выход (структура прежняя)
            summ["switch_ms"] = {"enter": fr[switch[0]]["total_ms"], "exit": fr[switch[1]]["total_ms"]}
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
