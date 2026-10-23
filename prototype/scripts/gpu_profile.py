"""Этап 3 (DD-004 п. 9): профиль режима A на GPU — мс на кадр 1080p и 4K по проходам (timestamps, медиана),
проходов SDF на луч, доля фолбэка, регистры и расчётная занятость; варианты: мегаядро / wavefront (K), Jet15 / HVP."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d import gpu, gpu_modea as GM, wormholes as wh
from geo4d.shapes import SHAPES
from geo4d.camera import make_camera
from geo4d.cubemap import bake
from geo4d.render import background
from geo4d.results import new_run

REGS_PER_SM, MAX_WARPS = 65536, 48          # Ada (sm 8.9): регистровый файл SM и предел варп-слотов


def occupancy(regs):
    try:
        r = int(regs)
    except (TypeError, ValueError):
        return None
    alloc = ((r + 7) // 8) * 8                 # гранулярность выделения — 8 регистров на поток
    return min(MAX_WARPS, REGS_PER_SM // (alloc * 32)) / MAX_WARPS


def camera(scene, W, H):
    sh = SHAPES[scene]
    if scene == "ellis":
        x, e = wh.ellis_point(6.0)
        return sh, make_camera(sh, x, -e, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=W, height=H, focus=6.0), 20.0, 0.0
    p = sh.params
    cam = make_camera(sh, np.array([8.0, 0, 0, p["W"]]), [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=W,
                      height=H, focus=6.0)
    return sh, cam, p["rho0"], p["rho0"]


def median_runs(rep):
    keys = [k for k in rep["runs"][0] if k != "start"]
    return {k: float(np.median([r[k] for r in rep["runs"]])) / 1e6 for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=3e-3)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--nfb", type=int, default=64)
    a = ap.parse_args()
    out = new_run(3, "gpu_profile", dict(vars(a), slangc=gpu.slangc_version(), cube=512, fmt="bc6h",
                                         occupancy_model="Ada: 64K регистров/SM, 48 варпов/SM, гранулярность 8"))
    cube = gpu.cube_for_gpu(bake(background, 512))
    rows = []
    hist = {}
    for W, H in [(1920, 1080), (3840, 2160)]:
        for scene in ["ellis", "compact_wh"]:
            sh, cam, esc, flight = camera(scene, W, H)
            tol = a.kappa * cam.theta_pix
            # мегаядро (по проходам, без фолбэка-волны: фолбэк — поток на пиксель)
            mk = GM.render(scene, cam, cube, tol, esc, flight_r=flight, nfb=a.nfb, fmt="bc6h")
            ms = {k: float(np.median(v["times_ns"])) / 1e6 for k, v in mk["reports"].items()}
            rows.append(dict(res=f"{W}x{H}", scene=scene, variant="мегаядро, Jet15", ms=ms, total=sum(ms.values()),
                             fallback=float(mk["flags"].mean()), passes_mean=float(mk["stats"][:, 3].mean()),
                             regs={k: v["stats"].get("CS/Register Count") for k, v in mk["reports"].items()}))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            for jet in ["jet15", "hvp"]:
                for K in [8, 16, 32, 64]:
                    if jet == "hvp" and K != 16:
                        continue
                    cal = GM.render_wavefront(scene, cam, cube, tol, esc, K=K, flight_r=flight, nfb=0, fmt="bc6h",
                                              jacobi_jet=jet)
                    iters = int(cal["cnt"][2]) + 2
                    wfr = GM.render_wavefront(scene, cam, cube, tol, esc, K=K, iters=iters, flight_r=flight, nfb=a.nfb,
                                              fmt="bc6h", repeat=a.repeat, jacobi_jet=jet)
                    assert np.array_equal(wfr["status"] == 3, cal["status"] == 3), "усечение итераций изменило результат"
                    ms = median_runs(wfr["report"])
                    st = wfr["report"]["stats"]
                    regs = {k.split(":")[0]: v for k, v in st.items() if k.endswith("CS/Register Count")}
                    rows.append(dict(res=f"{W}x{H}", scene=scene, variant=f"wavefront K={K}, {jet}", ms=ms,
                                     total=sum(ms.values()), iters=iters, fallback=float(wfr["flags"].mean()),
                                     passes_mean=float(wfr["passes"].mean()), passes_p99=float(np.percentile(wfr["passes"], 99)),
                                     regs=regs, occ_integ=occupancy(regs.get("integ"))))
                    print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
                    if jet == "jet15" and K == 16:
                        hist[f"{scene} {W}x{H}"] = wfr["passes"]
    (out / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in hist.items():
        ax.hist(v, bins=np.logspace(1, 4, 60), histtype="step", label=k)
    ax.set_xscale("log")
    ax.set_xlabel("проходов Jet15 на луч (центральный луч режима A)")
    ax.set_ylabel("пикселей")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "passes_hist.png", dpi=120)
    md = ["| разрешение | сцена | вариант | init | интегр. | finish | оттенение | детектор | фолбэк | всего, мс | итераций | фолбэк, % | проходов (ср.) | регистры интегр. | занятость |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["ms"]
        if "integrate" in m:
            cells = [m.get("init", 0), m["integrate"], m.get("finish", 0), m.get("shade", 0), m.get("detect", 0), m.get("fallback", 0)]
            reg = r["regs"].get("integ")
            occ = f"{r['occ_integ'] * 100:.0f} %" if r.get("occ_integ") is not None else "—"
        else:
            cells = [0, m["central"], 0, m["shade"], m["detect"], m.get("fallback", 0)]
            reg = r["regs"].get("central")
            occ = f"{occupancy(reg) * 100:.0f} %" if occupancy(reg) is not None else "—"
        md.append(f"| {r['res']} | {r['scene']} | {r['variant']} | " + " | ".join(f"{c:.2f}" for c in cells) +
                  f" | **{r['total']:.1f}** | {r.get('iters', '—')} | {r['fallback'] * 100:.1f} | {r['passes_mean']:.0f} | {reg} | {occ} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
