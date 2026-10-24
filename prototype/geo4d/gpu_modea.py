"""Режим A на GPU (DD-004): оркестровка ядер modea.slang через стенд geo4d.gpu (проверка корректности;
замеры кадра — отдельной подкомандой хоста)."""
from __future__ import annotations

import numpy as np

from . import gpu
from .modea import Detector, SIGMA_PIX

EXIT = {"ellis": 1, "compact_wh": 2}


def camera_buffer(cam, k_grad=2.0, flight_r=0.0, det: Detector = Detector(), filter_mode=0, cube_n=512):
    """8 float4 для Cam (modea.slang). filter_mode: 0 — SampleGrad (k_grad), 1 — программный фильтр M19."""
    return gpu.f32(np.array([
        cam.x, cam.f, cam.r, cam.u,
        [cam.tf, cam.width, cam.height, cam.focus],
        [cam.aperture, flight_r, k_grad, SIGMA_PIX],
        [cam.theta_pix, det.det_lo, min(det.det_hi, 1e30), min(det.tau, 1e30)],
        [filter_mode, cube_n, 0, 0]]))


def _params(n, tol, s_max, n_max, escape_mode, escape_param, h0=0.1):
    return [("u32", n), ("f32", tol), ("f32", h0), ("f32", s_max), ("f32", 0.0), ("f32", 1e-7), ("u32", n_max),
            ("u32", escape_mode), ("f32", escape_param), ("f32", -1.0), ("f32", 0.0), ("u32", 0)]


def render(scene, cam, cube, tol, escape_param, k_grad=2.0, flight_r=0.0, det: Detector = Detector(),
           nfb=256, seed=3, fmt="rgba16f", aniso=16, n_max=4000, s_max=1e4, fallback=True, filter_mode=0):
    """cube — кубмапа без рамки (листы, 6, N, N, 3). Возвращает цвета (H·W, 3), флаги, промежуточные данные и отчёты."""
    ex = EXIT[scene]
    sid = gpu.SCENE_IDS[scene]
    N = cam.width * cam.height
    camb = camera_buffer(cam, k_grad, flight_r, det, filter_mode, cube.shape[2])
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


def _push_line(pipe, items):
    out = [f"push {pipe}"]
    for t, v in items:
        out.append("iter" if t == "iter" else (f"{t} {float(v):.9g}" if t == "f32" else f"{t} {int(v)}"))
    return " ".join(out)


def render_wavefront(scene, cam, cube, tol, escape_param, K=16, iters=None, k_grad=2.0, flight_r=0.0,
                     det: Detector = Detector(), nfb=256, seed=3, fmt="rgba16f", aniso=16, n_max=4000, s_max=1e4,
                     filter_mode=0, repeat=1, jacobi_jet="jet15"):
    """Весь кадр режима A одним командным буфером (DD-004 п. 5): init → iters × (args → integrate K шагов, компакция)
    → finish → оттенение → детектор → очередь фолбэка → фолбэк (косвенно). Возвращает цвета, флаги, статусы, отчёт."""
    import subprocess
    import tempfile
    from pathlib import Path
    import json
    ex, sid = EXIT[scene], gpu.SCENE_IDS[scene]
    N = cam.width * cam.height
    iters = iters or (n_max + K - 1) // K + 1
    esc_mode = 1 if scene == "ellis" else 2
    jj = 1 if jacobi_jet == "hvp" else 0
    wf = lambda k, e: gpu.compile_shader("wavefront.slang", e, {"SCENE": sid, "NJ": 4, "EXIT": ex, "KERNEL": k, "JACOBI_JET": jj})
    md = lambda k, e, extra=None: gpu.compile_shader("modea.slang", e, {"SCENE": sid, "NJ": 0, "EXIT": ex, "KERNEL": k, **(extra or {})})
    params = _params(N, tol, s_max, n_max, esc_mode, escape_param)
    wpush = params + [("iter", 0), ("u32", K), ("u32", nfb), ("u32", seed)]
    wbind = "cam X V E SH St Q0 Q1 Cnt Args Dir Dd Status Flag FbHdr FbList"
    g = gpu.groups_for(N)[0]
    with tempfile.TemporaryDirectory() as td:
        td = gpu.QPath(td)
        np.save(td / "cam.npy", camera_buffer(cam, k_grad, flight_r, det, filter_mode, cube.shape[2]))
        np.save(td / "cube.npy", gpu.f32(cube))
        enc = f" {gpu.compile_shader('bc6h_encode.slang', 'bc6h_encode')}" if fmt == "bc6h" else ""
        L = [f"buffer cam in {td / 'cam.npy'}", f"cube Cube {td / 'cube.npy'} {fmt} {aniso}{enc}"]
        for name, bytes_ in [("X", 16 * N), ("V", 16 * N), ("E", 128 * N), ("SH", 16 * N), ("St", 8 * N), ("Q0", 4 * N),
                             ("Q1", 4 * N), ("Cnt", 16), ("Args", 64), ("Dir", 16 * N), ("Dd", 64 * N), ("Status", 4 * N),
                             ("Flag", 4 * N), ("FbHdr", 16), ("FbList", 4 * N), ("Color", 16 * N),
                             ("Partial", 16 * max(1, int(0.25 * N)) * ((nfb + 7) // 8))]:   # ≤ 25 % пикселей на фолбэке
            L.append(f"buffer {name} zero {bytes_}")
        L += [f"pipeline init {wf(10, 'wf_init')} wf_init {wbind}", f"pipeline args {wf(11, 'wf_args')} wf_args {wbind}",
              f"pipeline integ {wf(12, 'wf_integrate')} wf_integrate {wbind}",
              f"pipeline finish {wf(13, 'wf_finish')} wf_finish {wbind}",
              f"pipeline fbq {wf(14, 'wf_fbqueue')} wf_fbqueue {wbind}", f"pipeline fbargs {wf(14, 'wf_fbargs')} wf_fbargs {wbind}",
              f"pipeline shade {md(2, 'modea_shade')} modea_shade cam Dir Dd Status Color Cube",
              f"pipeline detect {md(3, 'modea_detect')} modea_detect cam Dir Dd Status Flag",
              f"pipeline fallback {md(4, 'modea_fallback', {'FB_CHUNK': 8})} modea_fallback cam FbHdr FbList Color Cube Partial",
              f"pipeline fbreduce {md(4, 'modea_fbreduce', {'FB_CHUNK': 8})} modea_fbreduce cam FbHdr FbList Color Cube Partial"]
        for p in ["init", "args", "integ", "finish", "fbq", "fbargs"]:
            L.append(_push_line(p, wpush))
        L.append(_push_line("fallback", params))
        L.append(_push_line("fbreduce", params))
        L += ["mark start", f"dispatch init {g} 1 1", "mark init", f"loop {iters}", "dispatch args 1 1 1",
              "indirect integ Args 0", "endloop", "mark integrate", f"dispatch finish {g} 1 1", "mark finish",
              f"dispatch shade {g} 1 1", "mark shade", f"dispatch detect {g} 1 1", "mark detect"]
        if nfb > 0:
            L += [f"dispatch fbq {g} 1 1", "dispatch fbargs 1 1 1", "indirect fallback Args 12",
                  "indirect fbreduce Args 24", "mark fallback"]
        L += [f"repeat {repeat}", f"output Color {td / 'color.npy'} f4 {N} 4", f"output Flag {td / 'flag.npy'} u4 {N}",
              f"output Status {td / 'status.npy'} u4 {N}", f"output St {td / 'st.npy'} u4 {N} 2",
              f"output Cnt {td / 'cnt.npy'} u4 4", f"report {td / 'report.json'}"]
        (td / "graph.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
        r = subprocess.run([str(gpu.EXE), "graph", str(td / "graph.txt")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"geo4d_gpu graph: {r.stdout}\n{r.stderr}")
        rep = json.loads((td / "report.json").read_text(encoding="utf-8"))
        return dict(color=np.load(td / "color.npy")[:, :3].astype(np.float64), flags=np.load(td / "flag.npy").astype(bool),
                    status=np.load(td / "status.npy"), passes=np.load(td / "st.npy")[:, 0], cnt=np.load(td / "cnt.npy"),
                    report=rep, iters=iters)
