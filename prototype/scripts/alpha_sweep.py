"""Этап 1 (ADR-0001): влияние параметра Баумгарта α на точность и стоимость BS32."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax.numpy as jnp

from geo4d.shapes import SHAPES, sample
from geo4d.integrators import make_integrator, Config
from geo4d.reference import dop853
from geo4d.results import new_run
from convergence import angle

SHAPE_NAMES = ["S3", "ellipsoid", "tiger", "ellis", "compact_wh"]
ALPHA_FACTORS = [0.0, 0.1, 1.0, 10.0]   # α = factor / ℓ_s
TOLS = [1e-4, 1e-6, 1e-8]


def main():
    out = new_run(1, "alpha_sweep", dict(shapes=SHAPE_NAMES, alpha_factors=ALPHA_FACTORS, tols=TOLS, nrays=32))
    res = []
    for name in SHAPE_NAMES:
        sh = SHAPES[name]
        L = min(10.0, 10.0 * sh.scale)
        X, V = sample(sh, 32, seed=11)
        xr, vr = dop853(sh, X, V, L)
        for k in ALPHA_FACTORS:
            for tol in TOLS:
                o = make_integrator(sh, Config(tol=tol, s_max=L, alpha=k / sh.scale))(jnp.asarray(X), jnp.asarray(V))
                r = dict(shape=name, alpha_factor=k, tol=tol, angle_err=float(np.max(angle(o["v"], vr))),
                         nsdf_mean=float(np.mean(o["nsdf"])), rejects_mean=float(np.mean(o["nrej"])))
                res.append(r)
                print(r, flush=True)
    (out / "alpha_sweep.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    lines = ["| фигура | tol | " + " | ".join(f"α={k:g}/ℓ_s: ошибка / SDF" for k in ALPHA_FACTORS) + " |",
             "|---|---|" + "---|" * len(ALPHA_FACTORS)]
    for name in SHAPE_NAMES:
        for tol in TOLS:
            cells = [next(f"{r['angle_err']:.1e} / {r['nsdf_mean']:.0f}" for r in res
                          if r["shape"] == name and r["tol"] == tol and r["alpha_factor"] == k) for k in ALPHA_FACTORS]
            lines.append(f"| {name} | {tol:g} | " + " | ".join(cells) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
