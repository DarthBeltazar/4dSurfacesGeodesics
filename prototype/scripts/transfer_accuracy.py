"""Этап 2: точность матрицы переноса против конечных разностей по фигурам (tol 1e-10 и рабочий κ·θ_pix),
и стоимость прохода Jet15 относительно Jet7."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from geo4d.shapes import SHAPES, sample
from geo4d.integrators import Config
from geo4d.jet import seed7, seed15
from geo4d.transfer import shape_of, cameras, run_case
from geo4d.results import new_run

CASES = sorted(SHAPES) + ["S3/mult", "ellipsoid/mult", "tiger/fexp", "ellis/mult"]
NCAM = 6
KAPPA = 1e-3


def jet_cost(sh, n=65536, reps=20):
    X, V = sample(sh, 64, seed=0)
    X = jnp.asarray(np.resize(X, (n, 4)))
    V = jnp.asarray(np.resize(V, (n, 4)))
    f7 = jax.jit(jax.vmap(lambda x, v: sh.f(seed7(x, v)).c))
    f15 = jax.jit(jax.vmap(lambda x: sh.f(seed15(x)).H.sum()))
    f0 = jax.jit(jax.vmap(sh.f_vec))
    out = {}
    for name, fn, args in [("scalar", f0, (X,)), ("jet7", f7, (X, V)), ("jet15", f15, (X,))]:
        fn(*args).block_until_ready()
        t = time.perf_counter()
        for _ in range(reps):
            fn(*args).block_until_ready()
        out[name] = (time.perf_counter() - t) / reps / n * 1e9      # нс на точку
    return out


def main():
    out = new_run(2, "transfer_accuracy", dict(cases=CASES, ncam=NCAM, kappa=KAPPA, camera="fov 60°, 64 px, F = 3ℓ_s",
                                               L="min(10, 10ℓ_s)", reference="DOPRI5 tol 1e-13, eps_pix 1e-3, eps_ap 1e-4·ℓ_s"))
    rows = []
    for name in CASES:
        sh = shape_of(name)
        L = min(10.0, 10.0 * sh.scale)
        cams = cameras(sh, NCAM, seed=2)
        th = cams[0][0].theta_pix
        r = dict(case=name)
        for label, cfg in [("tol1e-10", Config(tol=1e-10)), ("render", Config(tol=KAPPA * th))]:
            res = [run_case(sh, c, U, V, L, cfg, Config(method="dopri5", tol=1e-13)) for c, U, V in cams]
            r[f"max_rel_{label}"] = float(max(e for e, _ in res))
            r["min_ratio"] = float(min(q for _, q in res))
        if "/" not in name:
            r["cost_ns"] = jet_cost(sh)
        rows.append(r)
        print(r, flush=True)
    (out / "transfer_accuracy.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = ["| фигура | max отн. ошибка, tol 1e-10 | max отн. ошибка, tol = 1e-3·θ_pix | min ratio | Jet7/скаляр | Jet15/скаляр |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        c = r.get("cost_ns")
        cc = f"{c['jet7'] / c['scalar']:.1f} | {c['jet15'] / c['scalar']:.1f}" if c else "— | —"
        md.append(f"| {r['case']} | {r['max_rel_tol1e-10']:.1e} | {r['max_rel_render']:.1e} | {r['min_ratio']:.3f} | {cc} |")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
