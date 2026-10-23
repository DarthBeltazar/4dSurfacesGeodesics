"""Этап 3 (DD-004 п. 2): κ и α во float32 на GPU. Первичные лучи кадра 1080p (fov 90°, θ_pix ≈ 1e-3),
случайная выборка пикселей; эталон — прототип f64, DOPRI5 tol 1e-12. Метрика — p99 ошибки направления выхода в θ_pix.
Правило (заранее): наименьшая стоимость при p99 ≤ 0.05 θ_pix; если недостижимо — лучшая достижимая точность."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d import gpu, wormholes as wh
from geo4d.shapes import SHAPES
from geo4d.camera import make_camera, pixel_uv, pixel_dirs
from geo4d.integrators import make_integrator, Config
from geo4d.results import new_run

KAPPAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
ALPHAS = [0.0, 0.1, 1.0]


def scene_setup(name, W, H):
    sh = SHAPES[name]
    if name == "ellis":
        x_cam, e_ell = wh.ellis_point(6.0)
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=W, height=H)
        return sh, cam, None, 1, 20.0, wh.ellis_escape(20.0), wh.ellis_exit
    p = sh.params
    cam = make_camera(sh, np.array([8.0, 0, 0, p["W"]]), [-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=W, height=H)
    return sh, cam, p["rho0"], 2, p["rho0"], wh.compact_escape(p["rho0"]), wh.compact_exit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npix", type=int, default=4096)
    a = ap.parse_args()
    out = new_run(3, "gpu_kappa_alpha", dict(vars(a), kappas=KAPPAS, alphas=ALPHAS, res="1920×1080", fov=90,
                                             reference="прототип f64 DOPRI5 tol 1e-12", slangc=gpu.slangc_version(),
                                             rule="min стоимость при p99 ≤ 0.05 θ_pix"))
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axs, ["ellis", "compact_wh"]):
        sh, cam, flight_r, esc_mode, esc_param, esc, exf = scene_setup(name, 1920, 1080)
        U, V = (t.ravel() for t in pixel_uv(cam))
        rng = np.random.default_rng(0)
        pix = rng.choice(len(U), a.npix, replace=False)
        D = pixel_dirs(cam, U[pix], V[pix])
        X = np.broadcast_to(cam.x, D.shape).copy()
        if flight_r is not None:
            X, hit = wh.flat_flight_to_sphere(cam.x, D, np.zeros(3), flight_r)
            X, D, pix = X[hit], D[hit], pix[hit]
        X32, D32 = X.astype(np.float32).astype(np.float64), D.astype(np.float32).astype(np.float64)
        ref = make_integrator(sh, Config(method="dopri5", tol=1e-12, s_max=1e4, n_max=10 ** 6), escape=esc)(X32, D32)
        dref, sref, _ = exf(ref["x"], ref["v"])
        ok_ref = np.asarray(ref["status"]) == 2
        for alpha_f in ALPHAS:
            errs, costs = [], []
            for kap in KAPPAS:
                tol = kap * cam.theta_pix
                g, rep = gpu.trace(name, X32, D32, tol=tol, s_max=1e4, h0=0.1, n_max=4000, escape_mode=esc_mode,
                                   escape_param=esc_param, alpha=alpha_f / sh.scale)
                dg, sg, _ = exf(jnp.asarray(g["x"], jnp.float64), jnp.asarray(g["v"], jnp.float64))
                ok = ok_ref & (g["status"] == 2) & (np.asarray(sg) == np.asarray(sref))
                e = 2 * np.arcsin(np.clip(np.linalg.norm(np.asarray(dg)[ok] - np.asarray(dref)[ok], axis=1) / 2, 0, 1)) / cam.theta_pix
                r = dict(scene=name, alpha_factor=alpha_f, kappa=kap, tol=tol, p99_pix=float(np.percentile(e, 99)),
                         max_pix=float(e.max()), passes=float(np.mean(g["npass"])), frac_ok=float(ok.mean()),
                         frac_nmax=float(np.mean(g["status"] == 3)), time_ms=float(np.median(rep["times_ns"]) / 1e6))
                rows.append(r)
                print(json.dumps(r), flush=True)
                errs.append(r["p99_pix"])
                costs.append(r["passes"])
            ax.loglog(costs, errs, "o-", ms=3, label=f"α = {alpha_f:g}/ℓ_s")
        ax.axhline(0.05, color="gray", ls=":")
        ax.set_title(f"{name}: GPU f32, 1080p, fov 90°")
        ax.set_xlabel("проходов SDF на луч")
        ax.set_ylabel("p99 ошибки направления, θ_pix")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "kappa_alpha.png", dpi=120)
    (out / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = ["| сцена | α·ℓ_s | κ | p99, θ_pix | max, θ_pix | проходов | на потолке шагов |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scene']} | {r['alpha_factor']:g} | {r['kappa']:g} | {r['p99_pix']:.3g} | {r['max_pix']:.3g} | "
                  f"{r['passes']:.0f} | {r['frac_nmax']:.4f} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
