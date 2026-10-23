"""Этап 1: полный угол φ Эллиса (и компактной червоточины) против квадратуры для 50 прицельных параметров."""
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

from geo4d.shapes import SHAPES
from geo4d.integrators import make_integrator, Config
from geo4d import wormholes as wh, reference as ref
from geo4d.results import new_run

TOLS = [1e-4, 1e-6, 1e-8, 1e-10]
ELL_C, ELL_MAX, RHO_C = 10.0, 30.0, 5.0


def main():
    Ls_e = np.r_[np.linspace(0.05, 0.9, 25), np.linspace(1.1, 4.0, 25)]
    Ls_c = np.r_[np.linspace(0.05, 0.9, 25), np.linspace(1.1, 2.9, 25)]
    out = new_run(1, "deflection", dict(tols=TOLS, ellis=dict(ell_c=ELL_C, ell_max=ELL_MAX, L=Ls_e.tolist()),
                                        compact=dict(rho_c=RHO_C, L=Ls_c.tolist(), **SHAPES["compact_wh"].params)))
    rows, summary = [], {}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for scene, Ls in [("ellis", Ls_e), ("compact_wh", Ls_c)]:
        sh = SHAPES[scene]
        p = sh.params
        if scene == "ellis":
            X, V = wh.ellis_rays(ELL_C, Ls)
            esc, ex = wh.ellis_escape(ELL_MAX), wh.ellis_exit
            Q = [ref.ellis_phi_total(L, ELL_C) for L in Ls]
        else:
            X, V = wh.compact_rays(RHO_C, Ls, **p)
            esc, ex = wh.compact_escape(p["rho0"]), wh.compact_exit
            Q = [ref.compact_phi_total(L, RHO_C, **p) for L in Ls]
        q = np.array([v for v, _ in Q])
        for tol in TOLS:
            o = make_integrator(sh, Config(tol=tol, s_max=500.0), escape=esc, aux=wh.phi_increment, naux=1)(
                jnp.asarray(X), jnp.asarray(V))
            _, sheet, dphi = ex(o["x"], o["v"])
            err = np.asarray(o["aux"][:, 0] + dphi) - q
            sheet_ok = bool(np.all(np.asarray(sheet) == [s for _, s in Q]))
            summary[f"{scene}|{tol:g}"] = dict(max_abs_err=float(np.max(np.abs(err))), sheet_ok=sheet_ok,
                                               nsdf_mean=float(np.mean(o["nsdf"])))
            for L, e, ph, n in zip(Ls, err, q, np.asarray(o["nsdf"])):
                rows.append(dict(scene=scene, tol=tol, L=L, phi_quad=ph, err=e, nsdf=int(n)))
            ax = axs[0] if scene == "ellis" else axs[1]
            ax.semilogy(Ls, np.abs(err) + 1e-17, "o-", ms=2, label=f"tol={tol:g}")
        axs[2].loglog(TOLS, [summary[f"{scene}|{t:g}"]["max_abs_err"] for t in TOLS], "o-", label=scene)
        print(scene, {k: v for k, v in summary.items() if k.startswith(scene)})
    for ax, t in zip(axs[:2], ["Эллис: |φ − φ_квадр|", "компактная: |φ − φ_квадр|"]):
        ax.set_title(t)
        ax.set_xlabel("L")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axs[2].loglog(TOLS, 100 * np.array(TOLS), "k:", label="100·tol")
    axs[2].set_title("max ошибка по 50 L")
    axs[2].set_xlabel("tol")
    axs[2].grid(True, which="both", alpha=0.3)
    axs[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "deflection.png", dpi=130)
    with open(out / "deflection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
