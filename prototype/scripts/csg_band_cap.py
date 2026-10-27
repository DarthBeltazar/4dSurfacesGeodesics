"""M39 (черн.): предел шага у CSG многообразий по расстоянию до полосы склейки (DD-014, «вложенные носители»).

Сейчас внутри шара носителя сферы шаг ≤ ℓ_s/4 (ℓ_s = 0.5/κ_max — кромка шейки) по всему шару, включая тело S³.
Вне полосы |a − b| < k узла CSG f совпадает с куском; |∇(a − b)| ≤ L_a + L_b ⇒ до полосы не ближе (|a − b| − k)/L.
Новый предел: max(d_шара, min(¼ℓ_кусков, max((|a − b| − k)/L, ¼ℓ_s))). Сравнение со старым на лучах камеры сцены
bubble (плоскость ∪ S³, R = 3, d = 0.5, k = 2) и на «слое − шаре»: проходы SDF и ошибка против DOP853.
Пишет results/stage-5/<дата>_csg_band_cap/."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import jax
import jax.numpy as jnp

from geo4d.integrators import Config, make_integrator
from geo4d.reference import dop853
from geo4d.results import new_run
from geo4d.shapes import Shape, plane_sphere_union, sample, slab_minus_ball

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifold_csg import surface_stats  # noqa: E402

SAFETY = 0.5                      # ℓ_s = 0.5/κ_max, как в окне (kCsgScaleSafety)
LEVELS = (1.0, 0.5, 0.25)        # уровни полосы |t| < f·k для «graded»
TOLS = (1e-7, 3e-6)               # 3e-6 ≈ κ·θ_pix окна в 1080p (ADR-0004: κ = 3e-3, θ_pix ≈ 1.2e-3)


def shell_scales(sh, absT, k, n=40000):
    """ℓ_j = SAFETY/κ_max по выборке на M в оболочках |t| ∈ [k_{j+1}, k_j) уровней LEVELS·k (последняя — [0, k_J))."""
    P, _ = sample(sh, n, seed=0)
    Pd = jnp.asarray(P)
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    g = np.linalg.norm(G, axis=1)
    nn = G / g[:, None]
    Pr = np.eye(4)[None] - nn[:, :, None] * nn[:, None, :]
    kap = np.abs(np.linalg.eigvalsh(Pr @ H @ Pr / g[:, None, None])).max(axis=1)
    u = np.asarray(jax.vmap(absT)(Pd))
    ks = [f * k for f in LEVELS] + [0.0]
    return [SAFETY / max(kap[(u >= ks[j + 1]) & (u < ks[j])].max(initial=1e-9), 1e-9) for j in range(len(LEVELS))]


def caps(sh, r_c, ell_s, absT, k, L, ell_piece):
    """Предел шага: old — ¼ℓ_s в шаре носителя; new — M39 (полоса |t| < k); graded — M39 с уровнями внутри полосы."""
    old = lambda x: jnp.maximum(jnp.linalg.norm(x[:3]) - r_c, 0.25 * ell_s)
    new = lambda x: jnp.maximum(jnp.linalg.norm(x[:3]) - r_c,
                                jnp.minimum(0.25 * ell_piece, jnp.maximum((absT(x) - k) / L, 0.25 * ell_s)))
    ells = [max(e, ell_s) for e in shell_scales(sh, absT, k)]
    ells[-1] = ell_s                                  # самая внутренняя оболочка — кромка: ℓ_s по всей выборке

    def graded(x):
        t = absT(x)
        h = 0.25 * ell_piece
        for f, e in zip(LEVELS, ells):
            h = jnp.minimum(h, jnp.maximum((t - f * k) / L, 0.25 * e))
        return jnp.maximum(jnp.linalg.norm(x[:3]) - r_c, h)
    return dict(old=old, new=new, graded=graded), ells


def run(sh, X, V, L_path, ell_s, hc, alpha_scale, tols):
    """Эталон DOP853 — с max_step: без него на плоскости шаг растёт и перепрыгивает полосу склейки."""
    t = time.time()
    path = dop853(sh, X, V, L_path, alpha=0.1 / alpha_scale, t_eval=np.linspace(0.0, L_path, 801), max_step=0.05)
    xr = path[:, -1, :4]
    print(f"  DOP853 {time.time() - t:.0f} с", flush=True)
    out = {}
    for (name, h), tol in [(c, t) for t in tols for c in hc.items()]:
        s = Shape(sh.name, sh.f, ell_s, sh.sample_raw, params=sh.params, hcap=h)
        p = make_integrator(s, Config(tol=tol, s_max=L_path))(jnp.asarray(X), jnp.asarray(V))
        e = np.linalg.norm(np.asarray(p["x"]) - xr, axis=1)
        out[f"{name}_tol{tol:.0e}"] = dict(err_max=float(e.max()), err_med=float(np.median(e)), nsdf_mean=float(np.mean(p["nsdf"])),
                         nsdf=np.asarray(p["nsdf"]).tolist(), err=e.tolist(), status=np.asarray(p["status"]).tolist())
    out["max_w"] = np.abs(path[:, :, 3]).max(axis=1).tolist()
    return out


def report(name, r, ell_s):
    print(f"{name}: ℓ_s={ell_s:.3f}, ℓ уровней {np.round(r['ell_levels'], 3).tolist()}" + (f", в S³ {r['crossed']}" if "crossed" in r else ""), flush=True)
    for key, v in r.items():
        if isinstance(v, dict):
            print(f"  {key}: err max {v['err_max']:.2e} med {v['err_med']:.2e}, nsdf {v['nsdf_mean']:.0f}", flush=True)


def main():
    cfg = dict(bubble=dict(R=3.0, d=0.5, k=2.0, cam=[0.0, -12.0, 0.5, 0.0], fan_deg=20.0, n=24),
               slab=dict(W=1.5, B=2.0, k=1.0, n=12), L=40.0, tols=TOLS, levels=LEVELS, safety=SAFETY, ref_max_step=0.05)
    out_dir = new_run(5, "csg_band_cap", cfg)
    res = {}

    b = cfg["bubble"]
    R, d, k = b["R"], b["d"], b["k"]
    sh = plane_sphere_union(R, d, k)
    r_c = R + 1.25 * k
    st = surface_stats(sh, r_c)
    ell_s = SAFETY / st["kappa_max"]
    C = R + d
    absT = lambda x: jnp.abs(x[3] - (jnp.linalg.norm(x - jnp.array([0.0, 0.0, 0.0, C])) - R))   # a = w, b = сфера
    hc, ells = caps(sh, r_c, ell_s, absT, k, 2.0, R)
    # лучи камеры: веер в плоскости xy и вверх по z (камера смотрит вдоль +y на шейку)
    rng = np.random.default_rng(1)
    th = np.deg2rad(b["fan_deg"]) * (2 * rng.random((b["n"], 2)) - 1)
    V = np.c_[np.sin(th[:, 0]), np.cos(th[:, 0]) * np.cos(th[:, 1]), np.cos(th[:, 0]) * np.sin(th[:, 1]), np.zeros(b["n"])]
    X = np.tile(b["cam"], (b["n"], 1))
    r = run(sh, X, V, cfg["L"], ell_s, hc, R, TOLS)
    r.update(kappa_max=st["kappa_max"], ell_s=ell_s, ell_levels=ells, crossed=int(np.sum(np.array(r["max_w"]) > 1.0)))
    res["bubble"] = r
    report("bubble", r, ell_s)

    s = cfg["slab"]
    W, B, k = s["W"], s["B"], s["k"]
    sh = slab_minus_ball(W, B, k)
    r_c = B + k
    st = surface_stats(sh, r_c)
    ell_s = SAFETY / st["kappa_max"]
    # csmax(a, b): a = (w² − W²)/(2W), b = B − |x|; |∇a| = |w|/W ≤ 1.25 при шаге ≤ W/4 у M (|w| ≤ W)
    absT = lambda x: jnp.abs((x[3] ** 2 - W * W) * (0.5 / W) - (B - jnp.linalg.norm(x)))
    hc, ells = caps(sh, r_c, ell_s, absT, k, 2.25, min(W, B))
    x0 = -(r_c + 3.0)
    bb = np.linspace(0.0, 1.2 * k, s["n"])
    X = np.c_[np.full(s["n"], x0), bb, np.zeros(s["n"]), np.full(s["n"], W)]
    V = np.tile([1.0, 0.0, 0.0, 0.0], (s["n"], 1))
    r = run(sh, X, V, cfg["L"], ell_s, hc, B, TOLS)
    r.update(kappa_max=st["kappa_max"], ell_s=ell_s, ell_levels=ells)
    res["slab_ball"] = r
    report("slab−ball", r, ell_s)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("→", out_dir)


if __name__ == "__main__":
    sys.exit(main())
