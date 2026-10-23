"""Этап 1: график сходимости при фиксированном шаге — наблюдаемый порядок (BS32 ≈ 3).

Ошибка = max по лучам |x(L) − x_ref(L)|/ℓ_s + угол(v, v_ref); эталон — точное решение (S³, S²×R) или DOP853.
"""
import argparse
import csv
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

METHODS = [  # (метка, параметры Config)
    ("naive", dict(method="naive")),
    ("rk4", dict(method="rk4")),
    ("bs32 stale", dict(method="bs32", fixed=True, fsal="stale")),
    ("bs32 reeval", dict(method="bs32", fixed=True, fsal="reeval")),
    ("dopri5", dict(method="dopri5", fixed=True)),
]


def angle(a, b):
    a, b = np.asarray(a), np.asarray(b)
    c = np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    s = np.linalg.norm(a / np.linalg.norm(a, axis=-1, keepdims=True) - b / np.linalg.norm(b, axis=-1, keepdims=True), axis=-1)
    return np.where(c > 0.9, 2 * np.arcsin(np.clip(s / 2, 0, 1)), np.arccos(np.clip(c, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="S3,S2xR,ellipsoid,ellis")
    ap.add_argument("--nrays", type=int, default=16)
    ap.add_argument("--L", type=float, default=10.0)
    a = ap.parse_args()
    shapes = a.shapes.split(",")
    hs = np.geomspace(0.4, 0.004, 12)
    out = new_run(1, "convergence", dict(shapes=shapes, nrays=a.nrays, L=a.L, hs=hs.tolist(),
                                         methods=[m for m, _ in METHODS]))
    rows = []
    fig, axs = plt.subplots(1, len(shapes), figsize=(4.2 * len(shapes), 3.8), squeeze=False)
    for ax, name in zip(axs[0], shapes):
        sh = SHAPES[name]
        X, V = sample(sh, a.nrays, seed=0)
        if sh.exact is not None:
            xr, vr = (np.asarray(t) for t in sh.exact(X, V, np.full(len(X), a.L)))
        else:
            xr, vr = dop853(sh, X, V, a.L)
        for label, kw in METHODS:
            errs = []
            for h in hs * sh.scale:
                o = make_integrator(sh, Config(h=h, s_max=a.L, h_max=np.inf, n_max=10 ** 7, **kw))(
                    jnp.asarray(X), jnp.asarray(V))
                e = np.max(np.linalg.norm(np.asarray(o["x"]) - xr, axis=1) / sh.scale + angle(o["v"], vr))
                errs.append(e)
                rows.append(dict(shape=name, method=label, h=h, err=e, nsdf=float(np.mean(o["nsdf"]))))
            errs = np.array(errs)
            ok = (errs > 1e-9) & (errs < 0.3)        # асимптотическая область: выше пола эталона ~1e-11
            p = np.polyfit(np.log(hs[ok]), np.log(errs[ok]), 1)[0] if ok.sum() >= 3 else np.nan
            for r in rows:
                if r["shape"] == name and r["method"] == label:
                    r["order"] = p
            ax.loglog(hs * sh.scale, errs, "o-", ms=3, label=f"{label} (p≈{p:.2f})")
            print(f"{name:10s} {label:12s} order {p:.2f}")
        ax.set_title(name)
        ax.set_xlabel("h")
        ax.set_ylabel("ошибка при s = L")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=130)
    with open(out / "convergence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
