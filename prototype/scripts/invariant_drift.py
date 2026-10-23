"""Этап 1 (U3): графики дрейфа инвариантов вдоль длины пути на всех фигурах."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d.shapes import SHAPES, sample, moments
from geo4d.integrators import make_integrator, Config
from geo4d.results import new_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--L", type=float, default=50.0, help="длина пути в единицах ℓ_s")
    ap.add_argument("--nrays", type=int, default=8)
    a = ap.parse_args()
    names = sorted(SHAPES)
    out = new_run(1, "invariant_drift", dict(vars(a), shapes=names))
    fig, axs = plt.subplots(2, 5, figsize=(20, 7))
    summary = {}
    for ax, name in zip(axs.flat, names):
        sh = SHAPES[name]
        X, V = sample(sh, a.nrays, seed=3)
        L = a.L * sh.scale
        o = make_integrator(sh, Config(tol=a.tol, s_max=L, record=20000))(jnp.asarray(X), jnp.asarray(V))
        rec, n = np.asarray(o["rec"]), np.asarray(o["nacc"])
        assert np.all(n < 20000)
        s_all, f_all, v_all, m_all, j_all = [], [], [], [], []
        for r, k in zip(rec, n):
            tr = r[: k + 1]
            x, v, s = tr[:, :4], tr[:, 4:8], tr[:, 8]
            s_all.append(s)
            f_all.append(np.abs(np.asarray(jax.vmap(sh.f_vec)(x))))
            v_all.append(np.abs(np.linalg.norm(v, axis=1) - 1))
            if sh.planes:
                Lm = np.asarray(moments(x, v, sh.planes))
                m_all.append(np.max(np.abs(Lm - Lm[0]), axis=1))
            if sh.invariants is not None:
                I = np.asarray(sh.invariants(x, v))[:, 0]
                j_all.append(np.abs(I / I[0] - 1))
        for i in range(len(s_all)):
            kw = dict(lw=0.6, alpha=0.7)
            ax.semilogy(s_all[i], f_all[i] + 1e-18, color="C0", label="|f|" if i == 0 else None, **kw)
            ax.semilogy(s_all[i], v_all[i] + 1e-18, color="C1", label="||v|−1|" if i == 0 else None, **kw)
            if m_all:
                ax.semilogy(s_all[i], m_all[i] + 1e-18, color="C2", label="max|ΔLᵢⱼ|" if i == 0 else None, **kw)
            if j_all:
                ax.semilogy(s_all[i], j_all[i] + 1e-18, color="C3", label="Иоахимсталь (отн.)" if i == 0 else None, **kw)
        summary[name] = dict(L=L, max_f=float(max(map(np.max, f_all))), max_dv=float(max(map(np.max, v_all))),
                             max_dL=float(max(map(np.max, m_all))) if m_all else None,
                             max_joach=float(max(map(np.max, j_all))) if j_all else None,
                             steps_mean=float(np.mean(n)))
        ax.set_title(f"{name}, tol={a.tol:g}")
        ax.set_xlabel("s")
        ax.set_ylim(1e-18, 1e-2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        print(name, summary[name], flush=True)
    fig.tight_layout()
    fig.savefig(out / "drift.png", dpi=110)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
