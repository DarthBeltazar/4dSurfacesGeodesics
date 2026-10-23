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


def compile_shader(source, entry, defines=None):
    """Компилирует gpu/shaders/<source> с точкой входа entry; кэш по содержимому всех .slang и define-ам."""
    defines = dict(defines or {})
    h = hashlib.sha256()
    for p in sorted((GPU / "shaders").rglob("*.slang")):
        h.update(p.read_bytes())
    h.update(json.dumps([source, entry, sorted(defines.items())]).encode())
    SPV_DIR.mkdir(parents=True, exist_ok=True)
    out = SPV_DIR / f"{Path(source).stem}_{entry}_{h.hexdigest()[:16]}.spv"
    if not out.exists():
        cmd = [str(SLANGC), str(GPU / "shaders" / source), "-target", "spirv", "-profile", "spirv_1_6",
               "-entry", entry, "-fvk-use-entrypoint-name", "-O2", "-o", str(out)]
        cmd += [f"-D{k}={v}" for k, v in defines.items()]
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
            lines.append(f"push {t} {v!r}" if t == "f32" else f"push {t} {int(v)}")
        outs = {}
        for b, (mode, val) in enumerate(buffers):
            if mode == "in":
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
