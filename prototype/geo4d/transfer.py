"""Матрица переноса луча (M17): поля Якоби → производные точки и направления выхода."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .jet import seed15
from .jacobi import shape_operator
from .camera import jacobi_init, transverse_basis, make_camera, pixel_dirs, lens_ray
from .shapes import SHAPES, sample, reparam
from .integrators import make_integrator


def ambient_dv(f, x, v, J):
    """M17: ∂v = J′ − ⟨SJ, v⟩n — нормальная поправка (возвращается слагаемое −⟨SJ, v⟩n). x, v: (4,), J: (k, 4)."""
    Jt = f(seed15(x))
    S, n = shape_operator(Jt.g, Jt.H)
    return -(J @ (S @ v))[:, None] * n


def state_derivatives(f, x, v, E):
    """Производные (∂x, ∂v) по каждому полю в объемлющих координатах: (k, 4), (k, 4)."""
    return E[0], E[1] + ambient_dv(f, x, v, E[0])


def exit_dir_derivatives(exit_fn, f, x, v, E):
    """M17: ∂d∞ по каждому полю — jvp функции выхода вдоль (J, J′ − ⟨SJ,v⟩n). Возвращает d∞ (3,), ∂d∞ (k, 3)."""
    dx, dv = state_derivatives(f, x, v, E)
    d = lambda xx, vv: exit_fn(xx, vv)[0]
    d0 = d(x, v)
    dd = jax.vmap(lambda a, b: jax.jvp(d, (x, v), (a, b))[1])(dx, dv)
    return d0, dd


def transfer_matrix_4x4(f, x, v, E):
    """M17: 4×4 (∂x⊥ по b₁,b₂; ∂v⊥ по b₁,b₂) × (ap_r, ap_u, pix_r, pix_u) в точке x, v."""
    x, v = np.asarray(x), np.asarray(v)
    dx, dv = (np.asarray(t) for t in state_derivatives(f, jnp.asarray(x), jnp.asarray(v), jnp.asarray(E)))
    g = np.asarray(jax.grad(lambda p: f(tuple(p[k] for k in range(4))))(jnp.asarray(x)))
    B = transverse_basis(v, g / np.linalg.norm(g))
    return np.concatenate([B @ dx.T, B @ dv.T], 0)


# ---- Проверка против конечных разностей (тест и скрипт отчёта) -----------------------------------

EPS_PIX = 1e-3        # сдвиг пикселя для разностей, пиксели
EPS_AP = 1e-4         # сдвиг апертуры, в единицах ℓ_s


def shape_of(name):
    """Фигура по имени; «база/вид» — вариант U5 (reparam)."""
    if "/" in name:
        base, kind = name.split("/")
        return reparam(SHAPES[base], kind)
    return SHAPES[name]


def cameras(sh, n, seed):
    X, V = sample(sh, n, seed=seed)
    rng = np.random.default_rng(seed + 1000)          # отдельный поток: не совпадать с сэмплером точек
    cams = []
    for x, v in zip(X, V):
        cam = make_camera(sh, x, v, rng.normal(size=4), fov=60.0, width=64, height=64, focus=3 * sh.scale)
        U, Vv = rng.uniform(-0.4, 0.4, 2)
        cams.append((cam, U, Vv))
    return cams


def neighbours(sh, cam, U, V):
    """Начальные состояния 8 соседей (±ε по каждому из 4 входов) и шаги ε (в единицах входа)."""
    dpix = 2 * cam.tf / cam.width
    xs, vs = [], []
    for k in range(4):
        for sgn in (+1, -1):
            if k < 2:
                a = sgn * EPS_AP * sh.scale
                x, v = lens_ray(sh, cam, U, V, a if k == 0 else 0.0, a if k == 1 else 0.0)
            else:
                dU, dV = (sgn * EPS_PIX * dpix, 0.0) if k == 2 else (0.0, sgn * EPS_PIX * dpix)
                x, v = cam.x, pixel_dirs(cam, np.asarray(U + dU), np.asarray(V + dV))
            xs.append(x)
            vs.append(v)
    eps = np.array([EPS_AP * sh.scale] * 2 + [EPS_PIX] * 2)
    return np.array(xs), np.array(vs), eps


def central_fd(Y, eps):
    Y = Y.reshape(4, 2, -1)
    return (Y[:, 0] - Y[:, 1]) / (2 * eps[:, None])


def run_case(sh, cam, U, V, L, cfg_c, cfg_n):
    """Поля Якоби (cfg_c) против центральных разностей соседей (cfg_n) при длине пути L.
    Возвращает (max относительная ошибка столбца, ratio = det позиционного блока пикс. полей / плоский)."""
    E0, d0 = jacobi_init(cam, np.asarray(U), np.asarray(V))
    oc = make_integrator(sh, cfg_c.with_(s_max=L, jacobi=4))(cam.x[None], d0[None], E0[None])
    xc, vc, E = (np.asarray(oc[k][0]) for k in ("x", "v", "E"))
    Xn, Vn, eps = neighbours(sh, cam, U, V)
    on = make_integrator(sh, cfg_n.with_(s_max=L))(Xn, Vn)
    fd_x = central_fd(np.asarray(on["x"]), eps)
    fd_v = central_fd(np.asarray(on["v"]), eps)
    dx, dv = (np.asarray(t) for t in state_derivatives(sh.f, jnp.asarray(xc), jnp.asarray(vc), jnp.asarray(E)))
    M = transfer_matrix_4x4(sh.f, xc, vc, E)
    # диагностика близости к каустике: позиционный блок пиксельных полей против плоского L·J′(0)
    det = abs(np.linalg.det(M[:2, 2:]))
    flat = L ** 2 * abs(np.linalg.det(transfer_matrix_4x4(sh.f, cam.x, d0, E0)[2:, 2:]))
    rel = max(np.max(np.linalg.norm(dx - fd_x, axis=1) / np.linalg.norm(np.c_[fd_x, fd_v], axis=1)),
              np.max(np.linalg.norm(dv - fd_v, axis=1) / np.linalg.norm(np.c_[fd_x, fd_v], axis=1)))
    return rel, det / flat
