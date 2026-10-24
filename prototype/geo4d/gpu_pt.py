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
    tol_max: float = 1e-3
    c_hit: float = 0.05          # ε_hit = c_hit·w (M26)
    k_cone: float = 0.0          # 0 — фон без мипов (мип 0), иначе доля раствора лепестка BSDF в конусе (M31)
    max_depth: int = 16
    rr_depth: int = 3
    nee: int = 1
    mis: int = 0                 # 0 — степенная эвристика, 1 — «только NEE» (проверка)
    s_max: float = 1000.0
    n_max: int = 4000
    env_scale: float = 1.0
    seed: int = 1


def render(scene: Scene, cam, spp, p: GPUPT = GPUPT(), fmt="rgba32f", repeat_split=1):
    """Средний цвет (H·W, 3), дисперсия среднего (H·W, 3), статистика по пикселям, отчёт стенда."""
    from .gpu import cube_for_gpu
    inc = write_header(scene)
    defs = {"SCENE": scene.meta["scene_id"], "EXIT": scene.meta["exit"], "KERNEL": 0, "NJ": 0}
    spv = gpu.compile_shader("pathtrace.slang", "pt_sample", defs, include=inc)
    npix = cam.width * cam.height
    envb, env_n = env_buffer(scene)
    cube = cube_for_gpu(np.asarray(scene.mips[0]))
    push = [("u32", npix), ("u32", p.seed), ("f32", p.tol_kappa), ("f32", p.tol_max), ("f32", p.c_hit),
            ("f32", p.k_cone), ("u32", p.max_depth), ("u32", p.rr_depth), ("u32", p.nee), ("u32", p.mis),
            ("f32", p.s_max), ("u32", p.n_max), ("f32", scene.meta.get("ell_max", 0.0)), ("u32", env_n),
            ("f32", p.env_scale * scene.env_scale)]
    bufs = [("in", camera_buffer(cam, cube_n=cube.shape[2])), ("out", ("f4", (npix,))), ("out", ("f4", (npix,))),
            ("in", envb), ("cube", (cube, fmt, 16)), ("out", ("f4", (npix,)))]
    out, rep = gpu.run(spv, "pt_sample", bufs, push=push, groups=gpu.groups_for(npix), repeat=spp)
    acc, acc2, stats = out[1], out[2], out[5]
    cnt = acc[:, 3:4]
    mean = acc[:, :3] / cnt
    var = np.maximum(acc2[:, :3] / cnt - mean ** 2, 0) * cnt / np.maximum(cnt - 1, 1) / cnt
    return mean, var, stats, rep
