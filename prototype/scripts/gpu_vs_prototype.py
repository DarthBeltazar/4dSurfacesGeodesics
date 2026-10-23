"""Этап 3: GPU-f32 против прототипа-f32 и прототипа-f64 на одних и тех же начальных состояниях (DD-004 п. 3).

Эталон траекторий — DOP853 f64 (rtol 1e-12); эталон полей Якоби — прототип f64, DOPRI5 tol 1e-12.
Критерий: e_gpu / e_p32 ≤ 4 на каждой фигуре (луч и поля); дополнительно — e_gpu против e_p64 при том же tol.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo4d import gpu
from geo4d.shapes import SHAPES, sample
from geo4d.camera import make_camera, jacobi_init
from geo4d.reference import dop853
from geo4d.results import new_run

HERE = Path(__file__).resolve().parent


def proto(scene, X, V, E, tol, s_max, nj, x64, method="bs32"):
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, scene=scene, X=X, V=V, E=E if E is not None else np.zeros(1), tol=tol, s_max=s_max, nj=nj,
                 method=method)
        env = dict(os.environ, GEO4D_X64="1" if x64 else "0")
        r = subprocess.run([sys.executable, str(HERE / "_proto_run.py"), str(inp), str(out)], env=env,
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        return dict(np.load(out))


def ang(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrays", type=int, default=64)
    ap.add_argument("--tols", default="1e-5,1e-6")
    a = ap.parse_args()
    tols = [float(t) for t in a.tols.split(",")]
    names = sorted(SHAPES)
    out = new_run(3, "gpu_vs_prototype", dict(vars(a), shapes=names, slangc=gpu.slangc_version(),
                                              L="min(10, 10ℓ_s)", jacobi_camera="fov 60°, 64 px, F = 3ℓ_s, U,V ∈ [−0.4, 0.4]"))
    rows = []
    for name in names:
        sh = SHAPES[name]
        L = min(10.0, 10.0 * sh.scale)
        X, V = sample(sh, a.nrays, seed=7)
        X32, V32 = X.astype(np.float32).astype(np.float64), V.astype(np.float32).astype(np.float64)
        xr, vr = dop853(sh, X32, V32, L)                       # эталон — из тех же float32-состояний
        # поля Якоби: камеры в точках X
        rng = np.random.default_rng(11)
        E0 = []
        D0 = []
        for x, v in zip(X32, V32):
            cam = make_camera(sh, x, v, rng.normal(size=4), fov=60, width=64, height=64, focus=3 * sh.scale)
            e, d = jacobi_init(cam, np.asarray(rng.uniform(-0.4, 0.4)), np.asarray(rng.uniform(-0.4, 0.4)))
            E0.append(e)
            D0.append(d)
        E0, D0 = np.array(E0), np.array(D0)
        E0_32 = E0.astype(np.float32).astype(np.float64)
        D0_32 = D0.astype(np.float32).astype(np.float64)
        Eref = proto(name, X32, D0_32, E0_32, 1e-12, L, 4, True, method="dopri5")
        for tol in tols:
            r = dict(shape=name, tol=tol, L=L)
            g, rep = gpu.trace(name, X32, V32, tol=tol, s_max=L, h0=0.1 * sh.scale)
            p32 = proto(name, X32, V32, None, tol, L, 0, False)
            p64 = proto(name, X32, V32, None, tol, L, 0, True)
            for lab, o in [("gpu", g), ("p32", p32), ("p64", p64)]:
                r[f"pos_{lab}"] = float(np.max(np.linalg.norm(np.asarray(o["x"], np.float64) - xr, axis=1)) / sh.scale)
                r[f"dir_{lab}"] = float(np.max(ang(np.asarray(o["v"], np.float64), vr)))
            r["passes_gpu"] = float(np.mean(g["npass"]))
            r["passes_p64"] = float(np.mean(p64["passes"]))
            r["regs_ray"] = rep["stats"].get("CS/Register Count")
            for jet in ["jet15", "hvp"]:
                gj, repj = gpu.trace(name, X32, D0_32, E=E0_32, tol=tol, s_max=L, h0=0.1 * sh.scale, nj=4, jacobi_jet=jet)
                eref = np.max(np.abs(Eref["E"]))
                r[f"E_gpu_{jet}"] = float(np.max(np.abs(gj["E"] - Eref["E"])) / eref)
                r[f"regs_{jet}"] = repj["stats"].get("CS/Register Count")
            pj = proto(name, X32, D0_32, E0_32, tol, L, 4, False)
            r["E_p32"] = float(np.max(np.abs(pj["E"] - Eref["E"])) / np.max(np.abs(Eref["E"])))
            pj64 = proto(name, X32, D0_32, E0_32, tol, L, 4, True)
            r["E_p64"] = float(np.max(np.abs(pj64["E"] - Eref["E"])) / np.max(np.abs(Eref["E"])))
            ratio = lambda q: max(r[f"pos_{q}"], r[f"dir_{q}"])
            r["ratio_ray"] = ratio("gpu") / max(ratio("p32"), 1e-30)
            r["ratio_E"] = max(r["E_gpu_jet15"], r["E_gpu_hvp"]) / max(r["E_p32"], 1e-30)
            rows.append(r)
            print(json.dumps(r), flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = ["| фигура | tol | луч: GPU / p32 / p64 (max поз./ℓ_s, напр.) | ratio луч | поля: GPU Jet15 / HVP / p32 / p64 | ratio поля | проходов GPU / p64 | регистры луч / Jet15 / HVP |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda q: f"{max(r['pos_' + q], r['dir_' + q]):.1e}"
        md.append(f"| {r['shape']} | {r['tol']:g} | {f('gpu')} / {f('p32')} / {f('p64')} | {r['ratio_ray']:.2f} | "
                  f"{r['E_gpu_jet15']:.1e} / {r['E_gpu_hvp']:.1e} / {r['E_p32']:.1e} / {r['E_p64']:.1e} | {r['ratio_E']:.2f} | "
                  f"{r['passes_gpu']:.0f} / {r['passes_p64']:.0f} | {r['regs_ray']} / {r['regs_jet15']} / {r['regs_hvp']} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
