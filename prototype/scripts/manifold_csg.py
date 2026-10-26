"""M38 (черн.): CSG многообразий гладким минимумом с компактным носителем — проверка до кода окна.

Для «плоскость ∪ S³» (перебор k/2d) и «слой − 4-шар»:
- min |∇f| на M (особые точки при пережатии шейки) и радиус шейки (профиль по ρ = |q|, M осесимметрично);
- κ_max = max |собств. значение S| на M (S = P H P / |∇f|) → ℓ_s = 1/κ_max;
- носитель: вне шара |q| < r_c на M f ≡ w-листу (vᵀHv = 0), все точки с |S| > 1e-9 — внутри;
- геодезические через шейку: BS32 с hcap (r_c, ℓ_s) против DOP853, и тот же BS32 с ℓ_s = R (наивные метаданные).
Пишет results/stage-5/<дата>_manifold_csg/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from geo4d.integrators import Config, make_integrator
from geo4d.reference import dop853
from geo4d.results import new_run
from geo4d.shapes import Shape, plane_sphere_union, sample, slab_minus_ball

R, D = 3.0, 0.5
RATIOS = [1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0]          # k / 2d
N_PTS = 6000


def surface_stats(sh, r_c, seed=0):
    P, _ = sample(sh, N_PTS, seed=seed)
    res = np.abs(np.asarray(jax.vmap(sh.f_vec)(jnp.asarray(P))))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(jnp.asarray(P)))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(jnp.asarray(P)))
    ok = res < 1e-9
    P, G, H = P[ok], G[ok], H[ok]
    g = np.linalg.norm(G, axis=1)
    n = G / g[:, None]
    Pr = np.eye(4)[None] - n[:, :, None] * n[:, None, :]
    S = Pr @ H @ Pr / g[:, None, None]
    kap = np.abs(np.linalg.eigvalsh(S)).max(axis=1)
    q = np.linalg.norm(P[:, :3], axis=1)
    curved = kap > 1e-9
    return dict(n_on_M=int(ok.sum()), n_failed=int((~ok).sum()), min_grad=float(g.min()),
                kappa_max=float(kap.max()), ell_s=float(1 / kap.max()),
                max_q_curved=float(q[curved].max()) if curved.any() else 0.0,
                max_kappa_outside_carrier=float(kap[q >= r_c].max()) if (q >= r_c).any() else 0.0,
                n_outside_carrier=int((q >= r_c).sum()))


def neck_radius(f_vec, w_lo, w_hi, rho_max):
    """Минимум по w ∈ [w_lo, w_hi] наименьшего ρ > 0 с f(ρ, 0, 0, w) = 0 (на оси f < 0 — внутри шейки)."""
    ws = np.linspace(w_lo, w_hi, 201)
    best = np.inf
    rho = np.linspace(0, rho_max, 4001)
    fj = jax.jit(jax.vmap(f_vec))
    for w in ws:
        X = jnp.asarray(np.c_[rho, np.zeros((len(rho), 2)), np.full(len(rho), w)])
        fv = np.asarray(fj(X))
        if fv[0] >= 0:
            return 0.0                                     # ось вне — шейки нет (или разрыв)
        i = np.argmax(fv > 0)
        if fv[i] <= 0:
            continue
        best = min(best, rho[i])
    return float(best)


def derive(sh, ell_s, r_c):
    """Выведенные метаданные: ℓ_s = 1/κ_max, шаг вне носителя — до шара r_c, внутри — ℓ_s/4."""
    sh.scale = ell_s
    sh.hcap = lambda x: jnp.maximum(jnp.linalg.norm(x[:3]) - r_c, 0.25 * ell_s)


def geodesics(sh, naive_scale, r_c, n=12, L=40.0, tol=1e-7):
    """Лучи с плоскости из x = −(r_c + 3) на шейку с разными прицельными параметрами (в плоскости xy)."""
    x0 = -(r_c + 3.0)
    b = np.linspace(0.0, 1.2 * sh.params.get("k", 1.0), n)
    X = np.c_[np.full(n, x0), b, np.zeros(n), np.zeros(n)]
    V = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    xr, _ = dop853(sh, X, V, L)
    out = {}
    for name, s in [("derived", sh), ("naive", Shape(sh.name, sh.f, naive_scale, sh.sample_raw))]:
        p = make_integrator(s, Config(tol=tol, s_max=L))(jnp.asarray(X), jnp.asarray(V))
        e = np.linalg.norm(np.asarray(p["x"]) - xr, axis=1)
        out[name] = dict(err_max=float(e.max()), err_med=float(np.median(e)), nsdf_mean=float(np.mean(p["nsdf"])),
                         status=np.asarray(p["status"]).tolist())
    out["into_sphere"] = int(np.sum(xr[:, 3] > 0.5 * sh.params.get("d", 0.0) + 1e-3))
    return out


def main():
    cfg = dict(R=R, d=D, ratios=RATIOS, n_pts=N_PTS, csmin_c=0.25,
               slab=dict(W=1.5, B=2.0, k=1.0), geodesic=dict(L=40.0, tol=1e-7, n=12))
    out_dir = new_run(5, "manifold_csg", cfg)
    rows = []
    for r in RATIOS:
        k = r * 2 * D
        sh = plane_sphere_union(R, D, k)
        r_c = R + 1.25 * k
        st = surface_stats(sh, r_c)
        st.update(ratio=r, k=k, neck=neck_radius(sh.f_vec, -0.25 * k, D + 0.25 * k, R))
        derive(sh, st["ell_s"], r_c)
        st["geodesics"] = geodesics(sh, R, r_c)
        rows.append(st)
        g = st["geodesics"]
        print(f"k/2d={r:4.2f} k={k:.2f} neck={st['neck']:.3f} min|∇f|={st['min_grad']:.3e} κmax={st['kappa_max']:.3f} "
              f"q_curved≤{st['max_q_curved']:.2f} (r_c={r_c:.2f}) κ_out={st['max_kappa_outside_carrier']:.1e} | "
              f"err derived={g['derived']['err_max']:.2e} naive={g['naive']['err_max']:.2e} "
              f"nsdf {g['derived']['nsdf_mean']:.0f}/{g['naive']['nsdf_mean']:.0f} into S³: {g["into_sphere"]}/12",
              flush=True)
    s = cfg["slab"]
    sh = slab_minus_ball(**s)
    r_c = s["B"] + s["k"]
    st = surface_stats(sh, r_c)
    st.update(neck=neck_radius(sh.f_vec, -0.2, 0.2, 2 * s["B"]))
    derive(sh, st["ell_s"], r_c)
    st["geodesics"] = geodesics(sh, s["B"], r_c)
    print("slab−ball:", json.dumps({k: v for k, v in st.items() if k != "geodesics"}),
          {k: v["err_max"] for k, v in st["geodesics"].items() if isinstance(v, dict)})
    (out_dir / "results.json").write_text(json.dumps(dict(plane_sphere=rows, slab_ball=st), indent=2), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    sys.exit(main())
