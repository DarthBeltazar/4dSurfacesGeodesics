"""Этап 1: стоимость первичных лучей рендера при равной точности выходного направления (в долях пикселя).

Сцены: Эллис (камера на ℓ = 6) и компактная червоточина (камера в плоской зоне, аналитический пролёт до шара ρ₀).
Эталон — DOPRI5 с tol = 1e-12. Метрика — 99-й перцентиль угла между выходными направлениями (в θ_pix)
и доля пикселей с другим листом выхода. Стоимость — средние проходы SDF на пиксель.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d.shapes import SHAPES
from geo4d.integrators import Config, DONE_ESCAPE
from geo4d import wormholes as wh
from geo4d.render import pinhole_dirs, render_wormhole
from geo4d.results import new_run
from benchmark import cost_at

METHODS = [("RK4 (фикс.)", dict(method="rk4"), "h"),
           ("BS32 stale", dict(method="bs32", fsal="stale"), "tol"),
           ("BS32 reeval", dict(method="bs32", fsal="reeval"), "tol"),
           ("DOPRI5", dict(method="dopri5"), "tol")]
TARGETS = [0.1, 0.01]   # p99 ошибки в θ_pix


def scene(name, res, fov):
    sh = SHAPES[name]
    if name == "ellis":
        x_cam, e_ell = wh.ellis_point(6.0, sh.params["b"])
        dirs, th = pinhole_dirs(res, res, fov, -e_ell, np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]))
        return sh, x_cam, dirs, th, None, wh.ellis_escape(20.0), wh.ellis_exit
    p = sh.params
    x_cam = np.array([8.0, 0.0, 0.0, p["W"]])
    dirs, th = pinhole_dirs(res, res, fov, np.array([-1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]))
    X, hit = wh.flat_flight_to_sphere(x_cam, dirs, np.zeros(3), p["rho0"])
    return sh, X, dirs, th, hit, wh.compact_escape(p["rho0"]), wh.compact_exit


def ang(a, b):
    return 2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--fov", type=float, default=90.0)
    a = ap.parse_args()
    sweeps = dict(h=np.geomspace(0.5, 0.01, 9), tol=np.geomspace(1e-2, 1e-9, 12))
    out = new_run(1, "render_benchmark", dict(vars(a), targets_p99_theta_pix=TARGETS,
                                              sweeps={k: v.tolist() for k, v in sweeps.items()},
                                              methods={m: kw for m, kw, _ in METHODS}, reference="dopri5 tol=1e-12"))
    rows, table = [], {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axs, ["ellis", "compact_wh"]):
        sh, X, dirs, th, act, esc, ex = scene(name, a.res, a.fov)
        ref = render_wormhole(sh, X, dirs, Config(method="dopri5", tol=1e-12, s_max=1e4, n_max=20000), esc, ex, active=act)
        ok = ref[3] == DONE_ESCAPE
        n_act = int(np.sum(act)) if act is not None else len(dirs)
        for label, kw, par in METHODS:
            errs, costs = [], []
            for val in sweeps[par]:
                cfg = Config(s_max=1e4, n_max=20000, **({"h": val * sh.scale, "h_max": np.inf} if par == "h"
                                                         else {"tol": val, "h": 0.1 * sh.scale}), **kw)
                col, steps, nsdf, status, d, sheet = render_wormhole(sh, X, dirs, cfg, esc, ex, active=act)
                m = ok & (status == DONE_ESCAPE)
                e = ang(d[m], ref[4][m]) / th
                wrong = float(np.mean(sheet[m] != ref[5][m]))
                p99 = float(np.percentile(e, 99))
                c = float(np.sum(nsdf) / n_act)          # на интегрируемый пиксель
                errs.append(max(p99, 1e-12))
                costs.append(c)
                rows.append(dict(scene=name, method=label, param=par, value=val, p99_err_pix=p99,
                                 max_err_pix=float(np.max(e)), wrong_sheet_frac=wrong, nsdf_per_pixel=c,
                                 frac_not_escaped=float(np.mean(status != DONE_ESCAPE))))
            for t in TARGETS:
                table[(name, label, t)] = cost_at(errs, costs, t)
            ax.loglog(costs, errs, "o-", ms=3, label=label)
            print(name, label, [f"{table[(name, label, t)]:.0f}" for t in TARGETS], flush=True)
        for t in TARGETS:
            ax.axhline(t, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name}: первичные лучи {a.res}², fov {a.fov:g}°")
        ax.set_xlabel("проходов SDF на интегрируемый пиксель")
        ax.set_ylabel("p99 ошибки направления, θ_pix")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "render_work_precision.png", dpi=130)
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = []
    for t in TARGETS:
        md.append(f"\n### p99 ошибки ≤ {t:g} θ_pix: проходов SDF на интегрируемый пиксель\n")
        md.append("| метод | Эллис | компактная |")
        md.append("|---|---|---|")
        for label, _, _ in METHODS:
            md.append(f"| {label} | " + " | ".join(
                "вне развёртки" if np.isnan(table[(s, label, t)]) else f"{table[(s, label, t)]:.0f}"
                for s in ["ellis", "compact_wh"]) + " |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
