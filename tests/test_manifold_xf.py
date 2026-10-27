"""M40 (черн.): метаданные многообразия при преобразованиях xf.* (DD-015) — проверка в прототипе до окна.

f′(p) = s·f(T⁻¹p), T(p) = c + s·Q(p − c) + t (Q — поворот в плоскости из осей x, y, z). Метаданные по M40: листы
w + t_w (масштаб — около c_w), центры шаров T(c), радиусы шаров, k, ℓ_s, полосы M39 — ×s; L та же. Критерий:
BS32 с выведенным hcap против DOP853 — ошибка не больше 2·s·(ошибки исходной фигуры на прообразах лучей)."""
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d.integrators import Config, make_integrator
from geo4d.reference import dop853
from geo4d.shapes import Shape, plane_sphere_union, sample, slab_minus_ball

R, D, K = 3.0, 0.5, 2.0
SL = dict(W=1.5, B=2.0, k=1.0)


def xf(angle_deg, s, c, t):
    """T и T⁻¹ (поворот в плоскости xy на angle около c, масштаб s около c, затем сдвиг t)."""
    a = np.deg2rad(angle_deg)
    Q = np.eye(4)
    Q[0, 0] = Q[1, 1] = np.cos(a)
    Q[1, 0], Q[0, 1] = np.sin(a), -np.sin(a)
    c, t = np.asarray(c, float), np.asarray(t, float)
    fwd = lambda p: c + s * (Q @ (p - c)) + t
    inv_ops = lambda p: tuple(c[k] + sum(float(Q[j, k]) * (p[j] - c[j] - t[j]) for j in range(4)) / s for k in range(4))
    return fwd, inv_ops, Q


def base(name):
    """Исходная фигура, её метаданные (центр шара носителя 4D, r_c, ℓ_s, |t| узла, L, масштаб кусков, k, листы)."""
    if name == "bubble":
        sh = plane_sphere_union(R, D, K)
        C = np.array([0.0, 0.0, 0.0, R + D])
        absT = lambda x: jnp.abs(x[3] - (jnp.linalg.norm(x - jnp.asarray(C)) - R))
        return sh, dict(c=C, r_c=R + 1.25 * K, ell=0.097, absT=absT, L=2.0, ell_p=R, k=K, sheets=[0.0])
    sh = slab_minus_ball(**SL)
    W, B, k = SL["W"], SL["B"], SL["k"]
    absT = lambda x: jnp.abs((x[3] ** 2 - W * W) * (0.5 / W) + jnp.linalg.norm(x) - B)
    return sh, dict(c=np.zeros(4), r_c=B + 1.25 * k, ell=0.1, absT=absT, L=2.25, ell_p=min(W, B), k=k, sheets=[W, -W])


def hcap_from(m):
    cq = jnp.asarray(m["c"][:3])
    return lambda x: jnp.maximum(jnp.linalg.norm(x[:3] - cq) - m["r_c"],
                                 jnp.minimum(0.25 * m["ell_p"], jnp.maximum((m["absT"](x) - m["k"]) / m["L"], 0.25 * m["ell"])))


@pytest.mark.parametrize("name", ["bubble", "slab"])
@pytest.mark.parametrize("angle,s,t", [(0.0, 1.0, [1.0, -2.0, 0.5, 0.7]), (30.0, 1.0, [0.0, 0.0, 0.0, 0.0]),
                                       (25.0, 1.5, [0.5, 1.0, 0.0, -0.4])])
def test_transformed_metadata_geodesics(name, angle, s, t):
    sh, m = base(name)
    ctr = np.array([0.3, -0.2, 0.1, 0.4])
    fwd, inv, Q = xf(angle, s, ctr, t)
    # M40: метаданные преобразованной фигуры — из исходных, без знания f
    absT0 = m["absT"]
    Qi = jnp.asarray(Q.T)
    m2 = dict(c=fwd(m["c"]), r_c=s * m["r_c"], ell=s * m["ell"], L=m["L"], ell_p=s * m["ell_p"], k=s * m["k"],
              absT=lambda x: s * absT0(jnp.asarray(ctr) + Qi @ (x - jnp.asarray(ctr) - jnp.asarray(t)) / s),
              sheets=[ctr[3] + s * (w - ctr[3]) + t[3] for w in m["sheets"]])
    f0 = sh.f
    sh2 = Shape(name + "/xf", lambda p: s * f0(inv(p)), m2["ell"], lambda rng, n: np.array([fwd(p) for p in sh.sample_raw(rng, n)]),
                hcap=hcap_from(m2))
    sh1 = Shape(name, sh.f, m["ell"], sh.sample_raw, hcap=hcap_from(m))
    X, V = sample(sh1, 16, seed=3)
    n = 8                                         # и лучи с листа в шейку
    X = np.r_[X, np.c_[np.full(n, -(m["r_c"] + 1.0)), np.linspace(0, 2.0, n), np.zeros(n), np.full(n, m["sheets"][0])]]
    V = np.r_[V, np.tile([1.0, 0, 0, 0], (n, 1))]
    X2 = np.array([fwd(x) for x in X])
    V2 = V @ Q.T
    # листы преобразованной фигуры — на выведенных высотах (вдали от шара f′ = ±(w − w′) с точностью до множителя)
    far = np.array([fwd(np.array([m["c"][0] + 4 * m["r_c"], 0.0, 0.0, w])) for w in m["sheets"]])
    assert np.allclose(far[:, 3], m2["sheets"]) and np.allclose([float(sh2.f_vec(jnp.asarray(p))) for p in far], 0, atol=1e-12)
    L, tol = 12.0, 1e-6
    errs = []
    for shp, Xs, Vs, Ls in [(sh1, X, V, L), (sh2, X2, V2, s * L)]:
        xr, _ = dop853(shp, Xs, Vs, Ls, alpha=0.1 / m["ell_p"], max_step=0.05 * s)
        p = make_integrator(shp, Config(tol=tol, s_max=Ls))(jnp.asarray(Xs), jnp.asarray(Vs))
        assert np.all(np.asarray(p["status"]) == 1)
        errs.append(np.max(np.linalg.norm(np.asarray(p["x"]) - xr, axis=1)))
    assert errs[1] <= 2 * s * errs[0] + 1e-7, errs
