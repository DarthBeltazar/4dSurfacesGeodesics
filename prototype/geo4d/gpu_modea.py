"""Режим A на GPU (DD-004): оркестровка ядер modea.slang через стенд geo4d.gpu (проверка корректности;
замеры кадра — отдельной подкомандой хоста)."""
from __future__ import annotations

import numpy as np

from . import gpu
from .modea import Detector, SIGMA_PIX

EXIT = {"ellis": 1, "compact_wh": 2}


def camera_buffer(cam, k_grad=1.0, flight_r=0.0, det: Detector = Detector()):
    """7 float4 для Cam (modea.slang)."""
    return gpu.f32(np.array([
        cam.x, cam.f, cam.r, cam.u,
        [cam.tf, cam.width, cam.height, cam.focus],
        [cam.aperture, flight_r, k_grad, SIGMA_PIX],
        [cam.theta_pix, det.det_lo, det.det_hi, det.tau]]))


def _params(n, tol, s_max, n_max, escape_mode, escape_param, h0=0.1):
    return [("u32", n), ("f32", tol), ("f32", h0), ("f32", s_max), ("f32", 0.0), ("f32", 1e-7), ("u32", n_max),
            ("u32", escape_mode), ("f32", escape_param), ("f32", -1.0), ("f32", 0.0), ("u32", 0)]


def render(scene, cam, cube, tol, escape_param, k_grad=1.0, flight_r=0.0, det: Detector = Detector(),
           nfb=256, seed=3, fmt="rgba16f", aniso=16, n_max=4000, s_max=1e4, fallback=True):
    """cube — кубмапа без рамки (листы, 6, N, N, 3). Возвращает цвета (H·W, 3), флаги, промежуточные данные и отчёты."""
    ex = EXIT[scene]
    sid = gpu.SCENE_IDS[scene]
    N = cam.width * cam.height
    camb = camera_buffer(cam, k_grad, flight_r, det)
    esc_mode = 1 if scene == "ellis" else 2
    spv = gpu.compile_shader("modea.slang", "modea_central", {"SCENE": sid, "NJ": 4, "EXIT": ex, "KERNEL": 1})
    out, rep_c = gpu.run(spv, "modea_central",
                         [("in", camb), ("out", ("f4", (N, 4))), ("out", ("f4", (4 * N, 4))), ("out", ("f4", (N, 4))),
                          ("out", ("u4", (N,)))],
                         push=_params(N, tol, s_max, n_max, esc_mode, escape_param), groups=gpu.groups_for(N))
    d, dd, stats, status = out[1], out[2].reshape(N, 4, 4), out[3], out[4]
    spv = gpu.compile_shader("modea.slang", "modea_shade", {"SCENE": sid, "NJ": 0, "EXIT": ex, "KERNEL": 2})
    out, rep_s = gpu.run(spv, "modea_shade",
                         [("in", camb), ("in", d), ("in", gpu.f32(dd.reshape(4 * N, 4))), ("in", status.astype(np.uint32)),
                          ("out", ("f4", (N, 4))), ("cube", (cube, fmt, aniso))], groups=gpu.groups_for(N))
    col = out[4][:, :3].astype(np.float64)
    spv = gpu.compile_shader("modea.slang", "modea_detect", {"SCENE": sid, "NJ": 0, "EXIT": ex, "KERNEL": 3})
    out, rep_d = gpu.run(spv, "modea_detect",
                         [("in", camb), ("in", d), ("in", gpu.f32(dd.reshape(4 * N, 4))), ("in", status.astype(np.uint32)),
                          ("out", ("u4", (N,)))], groups=gpu.groups_for(N))
    flags = out[4].astype(bool)
    res = dict(color_nofb=col.copy(), flags=flags, d=d, dd=dd, stats=stats, status=status,
               reports=dict(central=rep_c, shade=rep_s, detect=rep_d))
    if fallback and flags.any():
        pix = np.nonzero(flags)[0].astype(np.uint32)
        spv = gpu.compile_shader("modea.slang", "modea_fallback", {"SCENE": sid, "NJ": 0, "EXIT": ex, "KERNEL": 4})
        out, rep_f = gpu.run(spv, "modea_fallback",
                             [("in", camb), ("in", np.array([[len(pix), nfb, seed, 0]], np.uint32)), ("in", pix),
                              ("out", ("f4", (len(pix), 4))), ("cube", (cube, fmt, aniso))],
                             push=_params(len(pix), tol, s_max, n_max, esc_mode, escape_param), groups=gpu.groups_for(len(pix)))
        col[pix] = out[3][:, :3]
        res["reports"]["fallback"] = rep_f
    res["color"] = col
    return res
