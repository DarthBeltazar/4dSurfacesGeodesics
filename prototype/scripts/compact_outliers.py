"""Этап 1 (ADR-0002): разбор пикселей с максимальной ошибкой направления на компактной червоточине."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from render_benchmark import scene, ang
from geo4d.integrators import Config, make_integrator
from geo4d.render import render_wormhole
from geo4d.results import new_run

TOLS = [1e-2, 5.34e-4, 2.85e-5]


def main():
    out = new_run(1, "compact_outliers", dict(res=96, fov=90.0, tols=TOLS, top=6, reference="dopri5 tol=1e-12"))
    sh, X, dirs, th, act, esc, ex = scene("compact_wh", 96, 90.0)
    ref = render_wormhole(sh, X, dirs, Config(method="dopri5", tol=1e-12, s_max=1e4, n_max=20000), esc, ex, active=act)
    rec_run = make_integrator(sh, Config(method="dopri5", tol=1e-12, s_max=1e4, n_max=20000, record=20000), escape=esc)
    res = []
    for tol in TOLS:
        r = render_wormhole(sh, X, dirs, Config(tol=tol, h=0.1, s_max=1e4, n_max=20000), esc, ex, active=act)
        e = ang(r[4], ref[4]) / th
        idx = np.argsort(e)[::-1][:6]
        L = np.linalg.norm(np.cross(X[idx, :3], dirs[idx, :3]), axis=1)
        o = rec_run(X[idx], dirs[idx])
        rec, n = np.asarray(o["rec"]), np.asarray(o["nacc"])
        rmin = [float(np.min(np.linalg.norm(rr[:k + 1, :3], axis=1))) for rr, k in zip(rec, n)]
        res.append(dict(tol=tol, frac_err_gt_0p1_pix=float(np.mean(e[act] > 0.1)), n_active=int(act.sum()),
                        top=[dict(pixel=int(j), err_pix=float(e[j]), L=float(L[i]), rho_min=rmin[i],
                                  sheet_ref=int(ref[5][j]), sheet=int(r[5][j])) for i, j in enumerate(idx)]))
        print(res[-1]["tol"], res[-1]["frac_err_gt_0p1_pix"], res[-1]["top"][0], flush=True)
    (out / "outliers.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
