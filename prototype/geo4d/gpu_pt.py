"""Режим B на GPU (DD-005): описание сцены прототипа → сгенерированный заголовок Slang (объекты, материалы,
источники, плоские зоны), таблица выборки фона, запуск мегаядра pathtrace.slang через стенд geo4d.gpu."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from . import gpu
from .gpu_modea import camera_buffer
from .pathtrace import Scene, env_table

KIND = {"diffuse": 0, "metal": 1, "emit": 2}
GEN = gpu.GPU / "build" / "gen"
MAX_REPEAT = 256


def _f(x):
    return f"{float(x):.9g}"


def _expr(spec):
    kind = spec[0]
    if kind == "ball4":
        c, r = spec[1], spec[2]
        return f"jlength4(x - {_f(c[0])}, y - {_f(c[1])}, z - {_f(c[2])}, w - {_f(c[3])}) - {_f(r)}"
    if kind == "torus":
        c, R, r = spec[1], spec[2], spec[3]
        return (f"jlength3(jlength2(x - {_f(c[0])}, y - {_f(c[1])}) - {_f(R)}, z - {_f(c[2])}, w - {_f(c[3])})"
                f" - {_f(r)}")
    if kind == "plane":
        ax, off, sg = spec[1], spec[2], spec[3]
        return f"{_f(sg)} * ({'xyzw'[ax]} - {_f(off)})"
    raise ValueError(f"нет генератора для объекта {kind}")


def _arr(typ, name, rows, n):
    rows = rows or ["0"]
    return f"static const {typ} {name}[{max(n, 1)}] = {{ {', '.join(rows)} }};"


def scene_header(scene: Scene) -> str:
    objs = scene.objs
    n = len(objs)
    L = ["// Сгенерировано geo4d.gpu_pt.scene_header из описания сцены прототипа — не редактировать.",
         f"#define PT_NOBJ {n}", f"#define PT_NLIGHT {len(scene.lights)}"]
    fz = scene.flat
    if fz is not None:
        L += ["#define PT_HAS_FLAT 1", f"#define PT_NBALL {len(fz.centers)}", f"static const float PT_RHO0 = {_f(fz.rho0)};",
              _arr("float3", "PT_BALL_C", [f"float3({', '.join(_f(a) for a in c)})" for c in fz.centers], len(fz.centers))]
    else:
        L += ["#define PT_HAS_FLAT 0", "#define PT_NBALL 0", "static const float PT_RHO0 = 0;",
              "static const float3 PT_BALL_C[1] = { float3(0) };"]
    C, R, PL, MA, ME = [], [], [], [], []
    for o in objs:
        c = o.center if o.center is not None else np.zeros(4)
        C.append(f"float4({', '.join(_f(a) for a in c)})")
        R.append(_f(min(o.bound, 1e30)))
        if o.plane is not None:
            PL.append(f"float4(1, {o.plane[0]}, {_f(o.plane[1])}, {_f(o.plane[2])})")
        else:
            PL.append("float4(0, 0, 0, 0)")
        m = o.mat
        MA.append(f"float4({', '.join(_f(a) for a in m.albedo)}, {KIND[m.kind]})")
        ME.append(f"float4({', '.join(_f(a) for a in m.emit)}, {_f(m.rough)})")
    L += [_arr("float4", "OBJ_C", C, n), _arr("float", "OBJ_R", R, n), _arr("float4", "OBJ_PL", PL, n),
          _arr("float4", "MAT_A", MA, n), _arr("float4", "MAT_E", ME, n),
          _arr("int", "LIGHT_OBJ", [str(k) for k in scene.lights], len(scene.lights))]
    L.append("__generic<T : IReal> T obj_g(int k, T x, T y, T z, T w)\n{")
    for k, o in enumerate(objs):
        L.append(f"    if (k == {k}) return {_expr(o.spec)};")
    L.append("    return T.cst(1e30);\n}")
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


def _push(scene, p: GPUPT, n, env_n, use_list=0, c_mix=1.0):
    return [("u32", n), ("u32", p.seed), ("f32", p.tol_kappa), ("f32", p.tol_max), ("f32", p.c_hit),
            ("f32", p.k_cone), ("u32", p.max_depth), ("u32", p.rr_depth), ("u32", p.nee), ("u32", p.mis),
            ("f32", p.s_max), ("u32", p.n_max), ("f32", scene.meta.get("ell_max", 0.0)), ("u32", env_n),
            ("f32", p.env_scale * scene.env_scale), ("f32", p.k_lod), ("u32", use_list), ("f32", c_mix)]


def _defs(scene, nj=0):
    return {"SCENE": scene.meta["scene_id"], "EXIT": scene.meta["exit"], "KERNEL": 0, "NJ": nj}


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
        bufs = [("in", camera_buffer(cam, cube_n=cube.shape[2])), ("inout", acc), ("inout", acc2), ("in", envb),
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


def camera_pass(scene: Scene, cam, p: GPUPT = GPUPT(), c_mix=1.0):
    """Центральные лучи с полями Якоби и объектами: класс пикселя (0 — фон, 1 — объект, 2 — смешанный), d∞, ∂d∞."""
    inc = write_header(scene)
    spv = gpu.compile_shader("pathtrace.slang", "pt_camera", _defs(scene, nj=4), include=inc)
    N = cam.width * cam.height
    envb, env_n = env_buffer(scene)
    d1 = np.zeros((1, 4), np.float32)
    bufs = [("in", camera_buffer(cam)), ("in", d1), ("in", d1), ("in", envb), ("in", d1), ("out", ("f4", (N, 4))),
            ("in", np.zeros(1, np.uint32)), ("in", d1), ("in", d1), ("out", ("f4", (N, 4))), ("out", ("f4", (4 * N, 4))),
            ("out", ("u4", (N,))), ("out", ("u4", (N,)))]
    out, rep = gpu.run(spv, "pt_camera", bufs, push=_push(scene, p, N, env_n, c_mix=c_mix), groups=gpu.groups_for(N))
    return dict(cls=out[12], d=out[9], dd=out[10].reshape(N, 4, 4), status=out[11], stats=out[5], report=rep)


def render_frame(scene: Scene, cam, spp, p: GPUPT = GPUPT(), nfb=64, c_mix=1.0, det=None, k_grad=2.0, fmt="rgba32f"):
    """Кадр режима B (DD-005 п. 2): проход камеры → фон режимом A (SampleGrad, детектор M20) → пиксели объектов,
    смешанные и помеченные детектором — трассировщиком (spp и nfb сэмплов). Возвращает цвет (H·W, 3) и данные."""
    from .gpu import cube_for_gpu
    from .gpu_modea import Detector
    det = det or Detector()
    cube = cube_for_gpu(np.asarray(scene.mips[0]))
    N = cam.width * cam.height
    cp = camera_pass(scene, cam, p, c_mix)
    camb = camera_buffer(cam, k_grad, 0.0, det, 0, cube.shape[2])
    defs = lambda k: {"SCENE": scene.meta["scene_id"], "NJ": 0, "EXIT": scene.meta["exit"], "KERNEL": k}
    dd = gpu.f32(cp["dd"].reshape(4 * N, 4))
    st = cp["status"].astype(np.uint32)
    out, rep_s = gpu.run(gpu.compile_shader("modea.slang", "modea_shade", defs(2)), "modea_shade",
                         [("in", camb), ("in", cp["d"]), ("in", dd), ("in", st), ("out", ("f4", (N, 4))),
                          ("cube", (cube, fmt, 16))], groups=gpu.groups_for(N))
    col = out[4][:, :3].astype(np.float64)
    out, rep_d = gpu.run(gpu.compile_shader("modea.slang", "modea_detect", defs(3)), "modea_detect",
                         [("in", camb), ("in", cp["d"]), ("in", dd), ("in", st), ("out", ("u4", (N,)))],
                         groups=gpu.groups_for(N))
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
