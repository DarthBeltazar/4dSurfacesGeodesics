"""Режим B на GPU (DD-005): описание сцены прототипа → сгенерированный заголовок Slang (объекты, материалы,
источники, плоские зоны), таблица выборки фона, запуск мегаядра pathtrace.slang через стенд geo4d.gpu."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import gpu
from .gpu_modea import camera_buffer
from . import pathtrace as PT
from .pathtrace import Scene, env_table

KIND = {"diffuse": 0, "metal": 1, "emit": 2}
GEN = gpu.GPU / "build" / "gen"
MAX_REPEAT = 256


def _f(x):
    return f"{float(x):.9g}"


def _arr(typ, name, rows, n):
    rows = rows or ["0"]
    return f"static const {typ} {name}[{max(n, 1)}] = {{ {', '.join(rows)} }};"


PT_MAXOBJ = 32                     # DD-008: объекты — строки float4 в хвосте буфера камеры (pathtrace.slang, obj_*)
OBJ_BASE = 8                       # первая строка: (n_obj, n_light, 0, 0), затем 8 строк индексов источников
OBJ_ROWS = 9 + 6 * PT_MAXOBJ       # объект k — строки 17 + 6k …: центр, вид, PL, альбедо, излучение, граница


def object_rows(scene: Scene):
    """Строки объектов сцены (OBJ_ROWS, 4) float32. NEE-источники — по правилу M34 (`pathtrace.nee_lights`)."""
    objs = scene.objs
    if len(objs) > PT_MAXOBJ:
        raise ValueError(f"объектов {len(objs)} > PT_MAXOBJ = {PT_MAXOBJ}")
    lights = PT.nee_lights(objs, scene.flat)
    R = np.zeros((OBJ_ROWS, 4))
    R[0, :2] = len(objs), len(lights)
    R[1:9].ravel()[:len(lights)] = lights
    for k, o in enumerate(objs):
        sp, m = o.spec, o.mat
        b = 9 + 6 * k
        if sp[0] == "ball4":
            R[b], R[b + 1] = sp[1], (0, sp[2], 0, 0)
        elif sp[0] == "torus":
            R[b], R[b + 1] = sp[1], (1, sp[2], sp[3], 0)
        elif sp[0] == "plane":
            R[b + 1] = (2, sp[1], sp[2], sp[3])
            R[b + 2] = (1, sp[1], sp[2], sp[3])
        else:
            raise ValueError(f"нет вида объекта {sp[0]} на GPU")
        R[b + 3] = (*m.albedo, KIND[m.kind])
        R[b + 4] = (*m.emit, m.rough)
        R[b + 5] = (min(o.bound, 1e30), float(o.on_M), 0, 0)
    return gpu.f32(R)


def pt_camera_buffer(scene: Scene, cam, *args, **kw):
    """Буфер камеры (gpu_modea.camera_buffer) + строки объектов сцены."""
    return np.concatenate([camera_buffer(cam, *args, **kw), object_rows(scene)])


def scene_header(scene: Scene) -> str:
    """Данные многообразия для трассировщика: плоские зоны (M29). Объекты — в буфере (object_rows)."""
    L = ["// Сгенерировано geo4d.gpu_pt.scene_header из описания сцены прототипа — не редактировать.",
         f"#define PT_MAXOBJ {PT_MAXOBJ}"]
    fz = scene.flat
    if fz is not None:
        L += ["#define PT_HAS_FLAT 1", f"#define PT_NBALL {len(fz.centers)}", f"static const float PT_RHO0 = {_f(fz.rho0)};",
              _arr("float3", "PT_BALL_C", [f"float3({', '.join(_f(a) for a in c)})" for c in fz.centers], len(fz.centers))]
    else:
        L += ["#define PT_HAS_FLAT 0", "#define PT_NBALL 0", "static const float PT_RHO0 = 0;",
              "static const float3 PT_BALL_C[1] = { float3(0) };"]
    return "\n".join(L) + "\n"


def write_header(scene: Scene):
    """Пишет заголовок в gpu/build/gen/<хэш>/pt_scene.slang и возвращает каталог для -I."""
    text = scene_header(scene)
    d = GEN / hashlib.sha256(text.encode()).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    (d / "pt_scene.slang").write_text(text, encoding="utf-8")
    return d


def env_buffer(scene: Scene):
    """Таблица выборки фона (M29) для GPU: [cdf листа 0 | cdf листа 1 | pmf листа 0 | pmf листа 1], float32."""
    cdf, pmf, n = env_table(scene.mips)
    if cdf.shape[0] == 1:
        cdf, pmf = np.repeat(cdf, 2, 0), np.repeat(pmf, 2, 0)
    return gpu.f32(np.concatenate([cdf.ravel(), pmf.ravel()])), n


@dataclass
class GPUPT:
    tol_kappa: float = 3e-3      # ADR-0004: tol = κ·max(θ_pix, γ) (M30)
    tol_max: float = 1e-2       # на измерениях не достигается (tol = κ·γ меньше)
    c_hit: float = 0.05          # ε_hit = c_hit·w (M26)
    k_cone: float = 0.25         # M31: прирост раствора конуса на отскоке (доля лепестка BSDF) → допуск M30
    k_lod: float = 0.0           # M31: множитель раствора отскоков для мипа фона (0 — мип 0)
    max_depth: int = 16
    rr_depth: int = 3
    nee: int = 1
    mis: int = 0                 # 0 — степенная эвристика, 1 — «только NEE» (проверка)
    s_max: float = 1000.0
    n_max: int = 4000
    env_scale: float = 1.0
    seed: int = 1


def _push(scene, p: GPUPT, n, env_n, use_list=0, c_mix=1.0, list_id=0, list_off=0, use_den=0, bgra=1, exposure=1.0):
    return [("u32", n), ("u32", p.seed), ("f32", p.tol_kappa), ("f32", p.tol_max), ("f32", p.c_hit),
            ("f32", p.k_cone), ("u32", p.max_depth), ("u32", p.rr_depth), ("u32", p.nee), ("u32", p.mis),
            ("f32", min(p.s_max, scene.meta.get("s_max", p.s_max))), ("u32", p.n_max), ("f32", scene.meta.get("ell_max", 0.0)), ("u32", env_n),
            ("f32", p.env_scale * scene.env_scale), ("f32", p.k_lod), ("u32", use_list), ("f32", c_mix),
            ("u32", list_id), ("u32", list_off), ("u32", use_den), ("u32", bgra), ("f32", exposure)]


# параметры фигур geo4d.shapes → дефайны scenes.slang (DD-008)
_MF = {"S3": {"R": "MF_R"}, "spheritorus": {"R": "MF_R", "r": "MF_RT"}, "torisphere": {"R": "MF_R", "r": "MF_RT"},
       "tiger": {"R1": "MF_R1", "R2": "MF_R2", "r": "MF_RT"}, "ellis": {"b": "MF_B"},
       "compact_wh": {"b": "MF_B", "rho0": "MF_RHO0", "W": "MF_W"},
       "two_compact_wh": {"b": "MF_B", "rho0": "MF_RHO0", "W": "MF_W"}}


def manifold_defines(shape):
    """Дефайны параметров многообразия для scenes.slang (значения — как у фигуры прототипа)."""
    lit = lambda x: repr(float(x))                     # вещественный литерал Slang (не «1» — целое)
    d = {v: lit(shape.params[k]) for k, v in _MF.get(shape.name, {}).items() if k in shape.params}
    if shape.name == "two_compact_wh":
        for i in (1, 2):
            for a, c in zip("XYZ", shape.params[f"c{i}"]):
                d[f"MF_C{i}{a}"] = lit(c)
    return d


def _defs(scene, nj=0):
    return {"SCENE": scene.meta["scene_id"], "EXIT": scene.meta["exit"], "KERNEL": 0, "NJ": nj,
            **manifold_defines(scene.shape)}


def render(scene: Scene, cam, spp, p: GPUPT = GPUPT(), fmt="rgba32f", pix=None, cube=None):
    """Трассировщик по всем пикселям (pix = None) или по списку. Возвращает средний цвет (H·W, 3), дисперсию среднего,
    статистику, отчёты и AOV (альбедо, нормаль — средние по сэмплам). Пиксели вне списка — нули."""
    from .gpu import cube_for_gpu
    inc = write_header(scene)
    spv = gpu.compile_shader("pathtrace.slang", "pt_sample", _defs(scene), include=inc)
    npix = cam.width * cam.height
    envb, env_n = env_buffer(scene)
    if cube is None:
        cube = cube_for_gpu(np.asarray(scene.mips[0]))
    lst = np.zeros(1, np.uint32) if pix is None else np.asarray(pix, np.uint32)
    n = npix if pix is None else len(lst)
    push = _push(scene, p, n, env_n, use_list=int(pix is not None))
    z4 = np.zeros((npix, 4), np.float32)
    acc, acc2, stats, alb, nrm = z4, z4.copy(), z4.copy(), z4.copy(), z4.copy()
    reps, done = [], 0
    while done < spp and n > 0:          # порциями: у хоста ограничен пул меток времени
        r = min(spp - done, MAX_REPEAT)
        bufs = [("in", pt_camera_buffer(scene, cam, cube_n=cube.shape[2])), ("inout", acc), ("inout", acc2), ("in", envb),
                ("cube", (cube, fmt, 16)), ("inout", stats), ("in", lst), ("inout", alb), ("inout", nrm)]
        out, rep = gpu.run(spv, "pt_sample", bufs, push=push, groups=gpu.groups_for(n), repeat=r)
        acc, acc2, stats, alb, nrm = out[1], out[2], out[5], out[7], out[8]
        reps.append(rep)
        done += r
    cnt = np.maximum(acc[:, 3:4], 1)
    mean = acc[:, :3] / cnt
    var = np.maximum(acc2[:, :3] / cnt - mean ** 2, 0) / np.maximum(cnt - 1, 1)
    render.aov = (alb[:, :3] / cnt, nrm[:, :3] / cnt)
    return mean, var, stats, reps


def camera_pass(scene: Scene, cam, p: GPUPT = GPUPT(), c_mix=1.0, repeat=1):
    """Центральные лучи с полями Якоби и объектами: класс пикселя (0 — фон, 1 — объект, 2 — смешанный), d∞, ∂d∞."""
    inc = write_header(scene)
    spv = gpu.compile_shader("pathtrace.slang", "pt_camera", _defs(scene, nj=4), include=inc)
    N = cam.width * cam.height
    envb, env_n = env_buffer(scene)
    d1 = np.zeros((1, 4), np.float32)
    bufs = [("in", pt_camera_buffer(scene, cam)), ("in", d1), ("in", d1), ("in", envb), ("in", d1), ("out", ("f4", (N, 4))),
            ("in", np.zeros(1, np.uint32)), ("in", d1), ("in", d1), ("out", ("f4", (N, 4))), ("out", ("f4", (4 * N, 4))),
            ("out", ("u4", (N,))), ("out", ("u4", (N,)))]
    out, rep = gpu.run(spv, "pt_camera", bufs, push=_push(scene, p, N, env_n, c_mix=c_mix), groups=gpu.groups_for(N),
                       repeat=repeat)
    return dict(cls=out[12], d=out[9], dd=out[10].reshape(N, 4, 4), status=out[11], stats=out[5], report=rep)


def render_frame(scene: Scene, cam, spp, p: GPUPT = GPUPT(), nfb=64, c_mix=1.0, det=None, k_grad=2.0, fmt="rgba32f",
                 repeat=1):
    """Кадр режима B (DD-005 п. 2): проход камеры → фон режимом A (SampleGrad, детектор M20) → пиксели объектов,
    смешанные и помеченные детектором — трассировщиком (spp и nfb сэмплов). Возвращает цвет (H·W, 3) и данные.
    repeat — повторы одиночных проходов (камера, оттенение, детектор) для замера времени (медиана)."""
    from .gpu import cube_for_gpu
    from .gpu_modea import Detector
    det = det or Detector()
    cube = cube_for_gpu(np.asarray(scene.mips[0]))
    N = cam.width * cam.height
    cp = camera_pass(scene, cam, p, c_mix, repeat)
    camb = camera_buffer(cam, k_grad, 0.0, det, 0, cube.shape[2])
    defs = lambda k: {**_defs(scene), "KERNEL": k}
    dd = gpu.f32(cp["dd"].reshape(4 * N, 4))
    st = cp["status"].astype(np.uint32)
    out, rep_s = gpu.run(gpu.compile_shader("modea.slang", "modea_shade", defs(2)), "modea_shade",
                         [("in", camb), ("in", cp["d"]), ("in", dd), ("in", st), ("out", ("f4", (N, 4))),
                          ("cube", (cube, fmt, 16))], groups=gpu.groups_for(N), repeat=repeat)
    col = out[4][:, :3].astype(np.float64)
    out, rep_d = gpu.run(gpu.compile_shader("modea.slang", "modea_detect", defs(3)), "modea_detect",
                         [("in", camb), ("in", cp["d"]), ("in", dd), ("in", st), ("out", ("u4", (N,)))],
                         groups=gpu.groups_for(N), repeat=repeat)
    flags = out[4].astype(bool) & (cp["cls"] == 0)
    obj = np.nonzero(cp["cls"] != 0)[0]
    fb = np.nonzero(flags)[0]
    ms = lambda reps: [t / 1e6 for rp in reps for t in rp["times_ns"]]
    res = dict(cls=cp["cls"], flags=flags, color_modea=col.copy(), camera=cp,
               times=dict(camera=ms([cp["report"]]), shade=ms([rep_s]), detect=ms([rep_d]), pt=[], fallback=[]),
               reports=dict(camera=cp["report"], shade=rep_s, detect=rep_d))
    if len(obj):
        m, v, st, rep = render(scene, cam, spp, p, fmt=fmt, pix=obj, cube=cube)
        col[obj] = m[obj]
        res["aov"] = render.aov
        res["var"], res["stats"] = v, st
        res["times"]["pt"] = ms(rep)
        res["reports"]["pt"] = rep[0]
    if len(fb) and nfb > 0:
        m, _, _, rep = render(scene, cam, nfb, GPUPT(**{**p.__dict__, "seed": p.seed + 7919}), fmt=fmt, pix=fb, cube=cube)
        col[fb] = m[fb]
        res["times"]["fallback"] = ms(rep)
    res["color"] = col
    return res


def _push_line(pipe, items):
    return " ".join([f"push {pipe}"] + [f"{t} {float(v):.9g}" if t == "f32" else f"{t} {int(v)}" for t, v in items])


def render_frame_graph(scene: Scene, cam, spp, p: GPUPT = GPUPT(), nfb=64, c_mix=1.0, det=None, k_grad=2.0, fmt="bc6h",
                       repeat=1):
    """Кадр режима B одним командным буфером (geo4d_gpu graph): проход камеры → оттенение и детектор режима A →
    списки пикселей на GPU → spp косвенных диспетчеризаций трассировщика по списку объектов и nfb — по фолбэку.
    Время — между метками (по прогонам, repeat); накопление трассировщика суммируется по прогонам.
    Возвращает цвет (H·W, 3), классы, флаги, AOV, отчёт."""
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    from .gpu import cube_for_gpu
    from .gpu_modea import Detector
    det = det or Detector()
    cube = cube_for_gpu(np.asarray(scene.mips[0]))
    N = cam.width * cam.height
    inc = write_header(scene)
    envb, env_n = env_buffer(scene)
    ptc = gpu.compile_shader("pathtrace.slang", "pt_camera", _defs(scene, 4), include=inc)
    pts = lambda e: gpu.compile_shader("pathtrace.slang", e, _defs(scene, 0), include=inc)
    md = lambda k, e: gpu.compile_shader("modea.slang", e, {**_defs(scene), "KERNEL": k})
    bind = "cam Acc Acc2 Env Cube Stats Pix Alb Nrm Dir Dd Status Class Flag Cnt Args"
    g = gpu.groups_for(N)[0]
    pfb = GPUPT(**{**p.__dict__, "seed": p.seed + 7919})
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        np.save(td / "cam.npy", pt_camera_buffer(scene, cam, k_grad, 0.0, det, 0, cube.shape[2]))
        np.save(td / "env.npy", envb)
        np.save(td / "cube.npy", gpu.f32(cube))
        enc = f" {gpu.compile_shader('bc6h_encode.slang', 'bc6h_encode')}" if fmt == "bc6h" else ""
        L = [f"buffer cam in {td / 'cam.npy'}", f"buffer Env in {td / 'env.npy'}", f"cube Cube {td / 'cube.npy'} {fmt} 16{enc}"]
        for name, b in [("Acc", 16 * N), ("Acc2", 16 * N), ("Stats", 16 * N), ("Pix", 8 * N), ("Alb", 16 * N), ("Nrm", 16 * N),
                        ("Dir", 16 * N), ("Dd", 64 * N), ("Status", 4 * N), ("Class", 4 * N), ("Flag", 4 * N), ("Cnt", 16),
                        ("Args", 32), ("Color", 16 * N)]:
            L.append(f"buffer {name} zero {b}")
        L += [f"pipeline camera {ptc} pt_camera {bind}",
              f"pipeline shade {md(2, 'modea_shade')} modea_shade cam Dir Dd Status Color Cube",
              f"pipeline detect {md(3, 'modea_detect')} modea_detect cam Dir Dd Status Flag",
              f"pipeline clear {pts('pt_clear')} pt_clear {bind}", f"pipeline lists {pts('pt_lists')} pt_lists {bind}",
              f"pipeline args {pts('pt_args')} pt_args {bind}",
              f"pipeline ptA {pts('pt_sample')} pt_sample {bind}", f"pipeline ptB {pts('pt_sample')} pt_sample {bind}",
              _push_line("camera", _push(scene, p, N, env_n, c_mix=c_mix)),
              _push_line("clear", _push(scene, p, N, env_n)), _push_line("lists", _push(scene, p, N, env_n, list_off=N)),
              _push_line("args", _push(scene, p, N, env_n)),
              _push_line("ptA", _push(scene, p, N, env_n, use_list=2, list_id=0, list_off=0)),
              _push_line("ptB", _push(scene, pfb, N, env_n, use_list=2, list_id=1, list_off=N)),
              "mark start", f"dispatch camera {g} 1 1", "mark camera", f"dispatch shade {g} 1 1", f"dispatch detect {g} 1 1",
              "mark modea", "dispatch clear 1 1 1", f"dispatch lists {g} 1 1", "dispatch args 1 1 1", "mark lists",
              f"loop {spp}", "indirect ptA Args 0", "endloop", "mark pt"]
        if nfb > 0:
            L += [f"loop {nfb}", "indirect ptB Args 12", "endloop", "mark fallback"]
        L += [f"repeat {repeat}", f"output Acc {td / 'acc.npy'} f4 {N} 4", f"output Color {td / 'col.npy'} f4 {N} 4",
              f"output Class {td / 'cls.npy'} u4 {N}", f"output Flag {td / 'flag.npy'} u4 {N}",
              f"output Alb {td / 'alb.npy'} f4 {N} 4", f"output Nrm {td / 'nrm.npy'} f4 {N} 4",
              f"output Stats {td / 'st.npy'} f4 {N} 4", f"output Cnt {td / 'cnt.npy'} u4 4", f"report {td / 'report.json'}"]
        (td / "graph.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
        r = subprocess.run([str(gpu.EXE), "graph", str(td / "graph.txt")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu graph: {r.stdout}\n{r.stderr}")
        rep = json.loads((td / "report.json").read_text(encoding="utf-8"))
        acc, col, cls = np.load(td / "acc.npy"), np.load(td / "col.npy")[:, :3].astype(np.float64), np.load(td / "cls.npy")
        flag = np.load(td / "flag.npy").astype(bool) & (cls == 0)
        alb, nrm, st, cnt = np.load(td / "alb.npy"), np.load(td / "nrm.npy"), np.load(td / "st.npy"), np.load(td / "cnt.npy")
    c = np.maximum(acc[:, 3:4], 1)
    traced = (cls != 0) | flag
    col[traced] = (acc[:, :3] / c)[traced]
    return dict(color=col, cls=cls, flags=flag, aov=(alb[:, :3] / c, nrm[:, :3] / c), stats=st, cnt=cnt, report=rep)


def cam_step(scene: Scene, cam, move, rot, frames=1, substeps=16, margin=0.05):
    """M33 на GPU: frames одинаковых шагов камеры (ядро cam_step). Возвращает (x, f, r, u) float32."""
    inc = write_header(scene)
    spv = gpu.compile_shader("pathtrace.slang", "cam_step", {**_defs(scene), "CAM_STEP": 1}, include=inc)
    push = [("f32", move[0]), ("f32", move[1]), ("f32", move[2]), ("f32", 0.0),
            ("f32", rot[0]), ("f32", rot[1]), ("f32", rot[2]), ("f32", margin), ("u32", substeps)]
    out, _ = gpu.run(spv, "cam_step", [("inout", pt_camera_buffer(scene, cam))], push=push, repeat=frames)
    return out[0][:4]


def snap_objects(scene: Scene):
    """DD-008 на GPU: ядро pt_snap проецирует центры объектов с on_M на M. Возвращает строки объектов после ядра."""
    inc = write_header(scene)
    spv = gpu.compile_shader("pathtrace.slang", "pt_snap", {**_defs(scene), "CAM_STEP": 1}, include=inc)
    buf = np.concatenate([np.zeros((OBJ_BASE, 4), np.float32), object_rows(scene)])
    push = [("f32", 0.0)] * 7 + [("f32", 0.0), ("u32", 16)]
    out, _ = gpu.run(spv, "pt_snap", [("inout", buf)], push=push)
    return out[0][OBJ_BASE:]


PT_BIND = ("cam Acc Acc2 Env Cube Stats Pix Alb Nrm Dir Dd Status Class Flag Cnt Args Frame AlbF NrmF Den Present Color")
I_SEED, I_USE_DEN, I_BGRA, I_EXPOSURE = 1, 20, 21, 22     # индексы push-констант PTParams (см. _push)


def app_graph(scene: Scene, cam, td, p: GPUPT = GPUPT(), nfb_move=8, speed=1.0, fmt="bc6h", hidden=False, vsync=False,
              bench=None, margin=0.05, det=None, k_grad=2.0, dumps=(), record=None, editor=None, edits=()):
    """Граф интерактивного приложения (DD-007): секции move / edit / static / post / snap и директивы app.
    td — каталог для файлов. bench = (файл сценария, json журнала) — режим замера. editor — dict(scene, work_dir, fov,
    env, python, script, args) для редактора сцен (DD-008); edits — [(кадр, строки объектов)] — горячие правки в
    сценарии замера (проверка: правка = холодный старт). Возвращает путь к графу."""
    from pathlib import Path
    from .gpu import cube_for_gpu
    from .gpu_modea import Detector
    td = Path(td)
    det = det or Detector()
    cube = cube_for_gpu(np.asarray(scene.mips[0]))
    N = cam.width * cam.height
    inc = write_header(scene)
    envb, env_n = env_buffer(scene)
    ptc = gpu.compile_shader("pathtrace.slang", "pt_camera", _defs(scene, 4), include=inc)
    pts = lambda e: gpu.compile_shader("pathtrace.slang", e, _defs(scene, 0), include=inc)
    cst = gpu.compile_shader("pathtrace.slang", "cam_step", {**_defs(scene), "CAM_STEP": 1}, include=inc)
    snp = gpu.compile_shader("pathtrace.slang", "pt_snap", {**_defs(scene), "CAM_STEP": 1}, include=inc)
    md = lambda k, e: gpu.compile_shader("modea.slang", e, {**_defs(scene), "KERNEL": k})
    g = gpu.groups_for(N)[0]
    pfb = GPUPT(**{**p.__dict__, "seed": p.seed + 7919})
    np.save(td / "cam.npy", pt_camera_buffer(scene, cam, k_grad, 0.0, det, 0, cube.shape[2]))
    np.save(td / "env.npy", envb)
    np.save(td / "cube.npy", gpu.f32(cube))
    enc = f" {gpu.compile_shader('bc6h_encode.slang', 'bc6h_encode')}" if fmt == "bc6h" else ""
    L = [f"buffer cam in {td / 'cam.npy'}", f"buffer Env in {td / 'env.npy'}", f"cube Cube {td / 'cube.npy'} {fmt} 16{enc}"]
    for name, b in [("Acc", 16 * N), ("Acc2", 16 * N), ("Stats", 16 * N), ("Pix", 8 * N), ("Alb", 16 * N), ("Nrm", 16 * N),
                    ("Dir", 16 * N), ("Dd", 64 * N), ("Status", 4 * N), ("Class", 4 * N), ("Flag", 4 * N), ("Cnt", 16),
                    ("Args", 32), ("Color", 16 * N), ("Present", 4 * N)]:
        L.append(f"buffer {name} zero {b}")
    for name in ["Frame", "AlbF", "NrmF", "Den"]:
        L.append(f"buffer {name} export {16 * N}")
    L += [f"pipeline camstep {cst} cam_step cam", f"pipeline snap {snp} pt_snap cam",
          f"pipeline camera {ptc} pt_camera {PT_BIND}",
          f"pipeline shade {md(2, 'modea_shade')} modea_shade cam Dir Dd Status Color Cube",
          f"pipeline detect {md(3, 'modea_detect')} modea_detect cam Dir Dd Status Flag"]
    background = scene.flat is not None or scene.meta.get("exit") == 1     # компактные пространства — без фона
    for name, entry in [("classall", "pt_classall"), ("clear", "pt_clear"), ("lists", "pt_lists"), ("args", "pt_args"), ("ptA", "pt_sample"),
                        ("ptB", "pt_sample"), ("reset", "pt_reset"), ("compose", "pt_compose"), ("present", "pt_present")]:
        L.append(f"pipeline {name} {pts(entry)} {entry} {PT_BIND}")
    cm = [("f32", 0)] * 4 + [("f32", 0)] * 3 + [("f32", margin), ("u32", 16)]
    L += [_push_line("camstep", cm), _push_line("snap", cm),
          _push_line("camera", _push(scene, p, N, env_n)), _push_line("clear", _push(scene, p, N, env_n)),
          _push_line("lists", _push(scene, p, N, env_n, list_off=N)), _push_line("args", _push(scene, p, N, env_n)),
          _push_line("ptA", _push(scene, p, N, env_n, use_list=2, list_id=0, list_off=0)),
          _push_line("ptB", _push(scene, pfb, N, env_n, use_list=2, list_id=1, list_off=N)),
          _push_line("reset", _push(scene, p, N, env_n)), _push_line("compose", _push(scene, p, N, env_n)),
          _push_line("classall", _push(scene, p, N, env_n)),
          _push_line("present", _push(scene, p, N, env_n, use_den=1, bgra=1))]
    # кадр со сбросом накопления: move — после шага камеры, edit — после правки объектов (камера на месте)
    body = ([f"dispatch camera {g} 1 1", "mark camera", f"dispatch shade {g} 1 1", f"dispatch detect {g} 1 1"] if background
            else [f"dispatch classall {g} 1 1", "mark camera"])
    body += ["dispatch clear 1 1 1", f"dispatch lists {g} 1 1",
             "dispatch args 1 1 1", f"dispatch reset {g} 1 1", "mark lists", "indirect ptA Args 0", "mark pt",
             f"loop {nfb_move}", "indirect ptB Args 12", "endloop", "mark fallback", f"dispatch compose {g} 1 1", "mark compose"]
    L += ["section move", "mark start", "dispatch camstep 1 1 1"] + body + ["section edit", "mark start"] + body
    L += ["section snap", "dispatch snap 1 1 1",
          "section static", "mark start", "indirect ptA Args 0", "mark pt", "indirect ptB Args 12", "mark fallback",
          f"dispatch compose {g} 1 1", "mark compose",
          "section post", f"dispatch present {g} 1 1",
          f"app {cam.width} {cam.height} Present", "denoise Frame AlbF NrmF Den", "campipe camstep",
          f"presentpipe present {I_USE_DEN} {I_BGRA} {I_EXPOSURE}", f"seedpipes {I_SEED} ptA ptB", f"speed {speed}",
          f"objrows cam {OBJ_BASE}"] + scene_directives(scene)
    if editor:
        L += editor_directives(scene, editor)
    for k, (fi, rows) in enumerate(edits):
        np.save(td / f"edit_rows_{k}.npy", gpu.f32(rows))
        L.append(f"edit {fi} {td / f'edit_rows_{k}.npy'}")
    if hidden:
        L.append("hidden")
    if vsync:
        L.append("vsync")
    if bench:
        L.append(f"bench {bench[0]} {bench[1]}")
    for fi, d in dumps:
        L.append(f"dump {fi} {d}")
    if record:
        L.append(f"record {record[0]} {record[1]}")
    path = td / "app_graph.txt"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def scene_directives(scene: Scene):
    """Плоские зоны (правило M34 в хосте) и имена объектов (DD-008)."""
    L = []
    fz = scene.flat
    if fz is not None:
        L.append(" ".join(["flat", str(len(fz.sheets))] + [_f(w) for w in fz.sheets] + [str(len(fz.centers)), _f(fz.rho0)]
                          + [_f(a) for c in fz.centers for a in c]))
    return L + [f"objname {k} {o.name.replace(' ', '_')}" for k, o in enumerate(scene.objs)]


def editor_directives(scene: Scene, ed):
    """Директивы редактора сцен (DD-008): файл сцены, многообразие, команда перегенерации графа."""
    from .scenefile import manifold_of
    kind, params = manifold_of(scene.shape)
    pv = lambda v: ",".join(_f(a) for a in np.atleast_1d(v))
    L = [f"editor {ed['scene']} {ed['work_dir']} {_f(ed['fov'])}",
         " ".join(["manifold", kind, ed.get("env") or "-"] + [f"{k}={pv(v)}" for k, v in params.items()])]
    if ed.get("python"):
        L.append(" ".join(["rebuild", ed["python"], ed["script"]] + list(ed.get("args", ()))))
    return L


def app_scene(name, mips=None):
    """Сцена и стартовая камера приложения по имени: two_wormholes, ellis, S3, spheritorus, torisphere, tiger."""
    from .camera import make_camera
    from .wormholes import ellis_point
    up = np.array([0, 0, 1.0, 0])
    if name == "two_wormholes":
        sc = PT.scene_two_wormholes(mips=mips)
        return sc, dict(x=np.array([0, -11.0, 0.5, 1.5]), f=np.array([0, 1.0, -0.12, 0]), up=up), 2.0
    if name == "ellis":
        sc = PT.scene_ellis(mips=mips)
        x, e = ellis_point(6.0)
        return sc, dict(x=x, f=-e, up=up), 2.0
    sc = PT.scene_compact_space(name)
    # S³: не в антиподе объекта (иначе все геодезические сходятся на нём и он закрывает весь фон)
    x0 = {"S3": [0.8, -0.35, 0.3, 0.38], "spheritorus": [1.0, 0, 0, 0.4], "torisphere": [1.4, 0, 0, 0],
          "tiger": [-1.5, 0, -0.8, 0]}[name]                                  # не в объекте (они при t = 2πk/7)
    from .shapes import project_points
    x = np.asarray(project_points(sc.shape, np.array([x0], float)))[0]
    return sc, dict(x=x, f=np.array([0, 1.0, 0, 0]), up=up), 0.5
