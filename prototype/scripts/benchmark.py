"""Этап 1: вызовы SDF (проходы Jet7) на луч при равной угловой точности. Таблица метод × фигура.

Для каждого метода перебирается шаг (фиксированные схемы) или tol (адаптивные); ошибка — максимум по лучам
угла между v(L) и эталонным v_ref(L) (DOP853, rtol = 1e-12). Стоимость при целевой ошибке ε —
лог-лог интерполяция между соседними точками развёртки.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d.shapes import SHAPES, sample
from geo4d.integrators import make_integrator, Config
from geo4d.reference import dop853
from geo4d.results import new_run
from convergence import angle

METHODS = [  # (метка, параметры Config, перебираемый параметр)
    ("naive (фикс.)", dict(method="naive"), "h"),
    ("RK4 (фикс.)", dict(method="rk4"), "h"),
    ("BS32 stale", dict(method="bs32", fsal="stale"), "tol"),
    ("BS32 reeval", dict(method="bs32", fsal="reeval"), "tol"),
    ("DOPRI5", dict(method="dopri5"), "tol"),
]
TARGETS = [1e-3, 1e-5, 1e-7]


def cost_at(errs, costs, target):
    """Минимальная стоимость, при которой ошибка ≤ target (лог-лог интерполяция по развёртке)."""
    order = np.argsort(costs)
    e, c = np.asarray(errs)[order], np.asarray(costs)[order]
    for i in range(len(c)):
        if e[i] <= target:
            if i == 0:
                return c[0]
            t = (np.log(target) - np.log(e[i - 1])) / (np.log(e[i]) - np.log(e[i - 1]))
            return float(np.exp(np.log(c[i - 1]) + t * (np.log(c[i]) - np.log(c[i - 1]))))
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="S3,ellipsoid,tiger,ellis")
    ap.add_argument("--nrays", type=int, default=32)
    a = ap.parse_args()
    shapes = a.shapes.split(",")
    sweeps = dict(h=np.geomspace(0.5, 1e-3, 13), h_naive=np.geomspace(0.1, 1e-4, 10),
                  tol=np.geomspace(1e-3, 1e-12, 10))
    out = new_run(1, "benchmark", dict(shapes=shapes, nrays=a.nrays, targets=TARGETS,
                                       path_length="min(10, 10·ℓ_s)", error="max угол(v, v_ref) по лучам",
                                       sweeps={k: v.tolist() for k, v in sweeps.items()},
                                       methods={m: kw for m, kw, _ in METHODS}))
    rows, table = [], {}
    fig, axs = plt.subplots(1, len(shapes), figsize=(4.2 * len(shapes), 3.8), squeeze=False)
    for ax, name in zip(axs[0], shapes):
        sh = SHAPES[name]
        L = min(10.0, 10.0 * sh.scale)
        X, V = sample(sh, a.nrays, seed=7)
        xr, vr = dop853(sh, X, V, L)
        for label, kw, par in METHODS:
            vals = sweeps["h_naive" if kw["method"] == "naive" else par]
            errs, costs = [], []
            for val in vals:
                if par == "h":
                    cfg = Config(h=val * sh.scale, s_max=L, h_max=np.inf, n_max=10 ** 7, **kw)
                else:
                    cfg = Config(tol=val, h=0.1 * sh.scale, s_max=L, n_max=10 ** 7, **kw)
                o = make_integrator(sh, cfg)(jnp.asarray(X), jnp.asarray(V))
                e = float(np.max(angle(o["v"], vr)))
                ep = float(np.max(np.linalg.norm(np.asarray(o["x"]) - xr, axis=1)) / sh.scale)
                c = float(np.mean(o["nsdf"]))
                errs.append(max(e, 1e-16))
                costs.append(c)
                rows.append(dict(shape=name, method=label, param=par, value=val, angle_err=e, pos_err=ep,
                                 nsdf_mean=c, nsdf_max=float(np.max(o["nsdf"])),
                                 rejects_mean=float(np.mean(o["nrej"]))))
            for t in TARGETS:
                table[(name, label, t)] = cost_at(errs, costs, t)
            ax.loglog(costs, errs, "o-", ms=3, label=label)
            print(name, label, [f"{table[(name, label, t)]:.0f}" for t in TARGETS], flush=True)
        for t in TARGETS:
            ax.axhline(t, color="gray", lw=0.5, ls=":")
        ax.set_title(f"{name}, L = {L:g}")
        ax.set_xlabel("проходов SDF на луч")
        ax.set_ylabel("max угловая ошибка, рад")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "work_precision.png", dpi=130)
    with open(out / "sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = []
    for t in TARGETS:
        md.append(f"\n### Угловая ошибка ≤ {t:g} рад: проходов SDF на луч (среднее)\n")
        md.append("| метод | " + " | ".join(shapes) + " |")
        md.append("|---" * (len(shapes) + 1) + "|")
        for label, _, _ in METHODS:
            cells = []
            for s in shapes:
                v = table[(s, label, t)]
                cells.append("вне развёртки" if np.isnan(v) else f"{v:.0f}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "table.json").write_text(json.dumps({f"{k[0]}|{k[1]}|{k[2]:g}": v for k, v in table.items()},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
