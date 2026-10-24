"""Стенд GPU-порта (DD-004 п. 10): компиляция Slang → SPIR-V с кэшем, запуск `geo4d_gpu run`, обмен через .npy."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT / "gpu"
EXE = GPU / "build" / "geo4d_gpu.exe"
SPV_DIR = GPU / "build" / "spv"
SLANGC = Path(os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.328.1")) / "Bin" / "slangc.exe"

SCENE_IDS = {"hyperplane": 0, "S3": 1, "S2xR": 2, "S1xR2": 3, "ellipsoid": 4, "spheritorus": 5, "torisphere": 6,
             "tiger": 7, "ellis": 8, "compact_wh": 9, "two_compact_wh": 10}


def available():
    return EXE.exists() and SLANGC.exists()


def slangc_version():
    r = subprocess.run([str(SLANGC), "-version"], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def compile_shader(source, entry, defines=None, include=None):
    """Компилирует gpu/shaders/<source> с точкой входа entry; кэш по содержимому всех .slang, define-ам и файлам
    дополнительного каталога include (сгенерированные заголовки, например описание сцены режима B)."""
    defines = dict(defines or {})
    h = hashlib.sha256()
    dirs = [GPU / "shaders"] + ([Path(include)] if include else [])
    for d in dirs:
        for p in sorted(d.rglob("*.slang")):
            h.update(p.read_bytes())
    h.update(json.dumps([source, entry, sorted(defines.items())]).encode())
    SPV_DIR.mkdir(parents=True, exist_ok=True)
    out = SPV_DIR / f"{Path(source).stem}_{entry}_{h.hexdigest()[:16]}.spv"
    if not out.exists():
        cmd = [str(SLANGC), str(GPU / "shaders" / source), "-target", "spirv", "-profile", "spirv_1_6",
               "-entry", entry, "-fvk-use-entrypoint-name", "-O2", "-o", str(out)]
        cmd += [f"-D{k}={v}" for k, v in defines.items()]
        if include:
            cmd += ["-I", str(include)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"slangc: {r.stdout}\n{r.stderr}")
    return out


def run(spv, entry, buffers, push=(), groups=(1, 1, 1), repeat=1):
    """buffers — список по binding: ("in", массив) | ("out", (dtype, shape)) | ("inout", массив).
    push — список (тип, значение), тип ∈ f32 | u32 | i32. Возвращает (выходы по binding, отчёт)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lines = [f"spv {spv}", f"entry {entry}", f"groups {groups[0]} {groups[1]} {groups[2]}", f"repeat {repeat}",
                 f"report {td / 'report.json'}"]
        for t, v in push:
            lines.append(f"push {t} {float(v):.9g}" if t == "f32" else f"push {t} {int(v)}")   # не repr: np.float64(...)
        outs = {}
        for b, (mode, val) in enumerate(buffers):
            if mode == "cube":                   # ("cube", (массив (листы, 6, N, N, C), формат, анизотропия))
                arr, fmt, aniso = val
                np.save(td / f"cube{b}.npy", f32(arr))
                enc = f" {compile_shader('bc6h_encode.slang', 'bc6h_encode')}" if fmt == "bc6h" else ""
                lines.append(f"cube {b} {td / f'cube{b}.npy'} {fmt} {aniso}{enc}")
            elif mode == "in":
                np.save(td / f"in{b}.npy", np.ascontiguousarray(val))
                lines.append(f"buf {b} in {td / f'in{b}.npy'}")
            elif mode == "inout":
                np.save(td / f"in{b}.npy", np.ascontiguousarray(val))
                lines.append(f"buf {b} inout {td / f'in{b}.npy'} {td / f'out{b}.npy'}")
                outs[b] = td / f"out{b}.npy"
            else:
                dtype, shape = val
                lines.append(f"buf {b} out {td / f'out{b}.npy'} {dtype} " + " ".join(str(s) for s in shape))
                outs[b] = td / f"out{b}.npy"
        (td / "job.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        r = subprocess.run([str(EXE), "run", str(td / "job.txt")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu: {r.stdout}\n{r.stderr}")
        res = {b: np.load(p) for b, p in outs.items()}
        rep = json.loads((td / "report.json").read_text(encoding="utf-8"))
    return res, rep


def f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def groups_for(n, local=64):
    return ((n + local - 1) // local, 1, 1)


def trace(scene, X, V, E=None, tol=1e-6, h0=0.05, s_max=10.0, h_max=0.0, h_min=1e-7, n_max=100_000,
          escape_mode=0, escape_param=0.0, alpha=-1.0, xscale=0.0, fixed=False, nj=0, jacobi_jet="jet15",
          repeat=1):
    """Мегаядро trace.slang: лучи (N, 4) [+ поля (N, 2, 4, 4)] → x, v, E, статистика, статус, отчёт."""
    X, V = f32(X), f32(V)
    N = len(X)
    defs = {"SCENE": SCENE_IDS[scene], "NJ": nj, "JACOBI_JET": 1 if jacobi_jet == "hvp" else 0}
    spv = compile_shader("trace.slang", "trace", defs)
    Ebuf = f32(np.asarray(E).reshape(N * 8, 4)) if nj else np.zeros((1, 4), np.float32)
    push = [("u32", N), ("f32", tol), ("f32", h0), ("f32", s_max), ("f32", h_max), ("f32", h_min), ("u32", n_max),
            ("u32", escape_mode), ("f32", escape_param), ("f32", alpha), ("f32", xscale), ("u32", int(fixed))]
    bufs = [("inout", X), ("inout", V), ("inout", Ebuf), ("out", ("f4", (N, 4))), ("out", ("u4", (N,)))]
    out, rep = run(spv, "trace", bufs, push=push, groups=groups_for(N), repeat=repeat)
    res = dict(x=out[0], v=out[1], s=out[3][:, 0], nacc=out[3][:, 1], nrej=out[3][:, 2], npass=out[3][:, 3],
               status=out[4])
    if nj:
        res["E"] = out[2].reshape(N, 2, 4, 4)
    return res, rep


def cube_for_gpu(cube_bordered):
    """Кубмапа прототипа (листы, 6, N+2, N+2, 3) с рамкой → без рамки (листы, 6, N, N, 3) float32 для GPU
    (выборка куба в Vulkan бесшовная — рамка не нужна, DD-004 п. 8)."""
    return f32(np.asarray(cube_bordered)[:, :, 1:-1, 1:-1])


def denoise(color, albedo=None, normal=None, repeat=1):
    """Денойзер OptiX (geo4d_gpu denoise, DD-005 п. 5): color (H, W, 3|4), AOV той же формы. -> (H, W, 3), отчёт."""
    def rgba(a):
        a = np.asarray(a, np.float32)
        return np.ascontiguousarray(np.concatenate([a[..., :3], np.ones(a.shape[:-1] + (1,), np.float32)], -1))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        lines = [f"color {td / 'c.npy'}", f"output {td / 'o.npy'}", f"repeat {repeat}", f"report {td / 'r.json'}"]
        np.save(td / "c.npy", rgba(color))
        if albedo is not None:
            np.save(td / "a.npy", rgba(albedo))
            lines.append(f"albedo {td / 'a.npy'}")
        if normal is not None:
            np.save(td / "n.npy", rgba(normal))
            lines.append(f"normal {td / 'n.npy'}")
        (td / "job.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        r = subprocess.run([str(EXE), "denoise", str(td / "job.txt")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu denoise: {r.stdout}\n{r.stderr}")
        return np.load(td / "o.npy")[..., :3], json.loads((td / "r.json").read_text(encoding="utf-8"))
