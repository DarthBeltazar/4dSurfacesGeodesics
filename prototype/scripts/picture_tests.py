"""Этап 2: картинки тестов S³ (M21) и S¹×R² (M22) для отчёта. Сами проверки — tests/test_picture_*.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d.shapes import SHAPES
from geo4d.integrators import Config
from geo4d.camera import make_camera, pixel_uv, pixel_dirs
from geo4d.markers import ball_markers, cylinder_intrinsic_markers, trace_markers, shade, MARKER_COLORS, NO_HIT
from geo4d.results import new_run
from test_picture_s1r2 import BALLS, euclid_trace, S_MAX

RES = 160


def main():
    out = new_run(2, "picture_tests", dict(res=RES, s3=dict(R=1.0, markers="d=1.0 a=0.3 (красный), d=2.2 a=0.25 сбоку"),
                                           s1r2=dict(R=1.0, balls=BALLS.tolist(), s_max=S_MAX)))
    stats = {}
    # --- S³: вперёд и назад -------------------------------------------------------------------
    sh = SHAPES["S3"]
    x0 = np.array([0, 0, 0, 1.0])
    F, RT, UP = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2]
    d1, a1 = 1.0, 0.3
    m1 = np.cos(d1) * x0 + np.sin(d1) * F
    side = np.cos(0.5) * F + np.sin(0.5) * RT
    m2 = np.cos(2.2) * x0 + np.sin(2.2) * side
    g = ball_markers([m1, m2], [2 * np.sin(a1 / 2), 2 * np.sin(0.25 / 2)])
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, fwd, title in [(axs[0], F, "S³: вперёд"), (axs[1], -F, "S³: назад (второе изображение)")]:
        cam = make_camera(sh, x0, fwd, UP, RT if fwd[0] > 0 else -RT, fov=100, width=RES, height=RES)
        U, V = pixel_uv(cam)
        D = pixel_dirs(cam, U, V).reshape(-1, 4)
        ids, s, st, steps = trace_markers(sh, np.broadcast_to(x0, D.shape), D, g,
                                          Config(tol=1e-3 * cam.theta_pix, s_max=2 * np.pi - 1e-3, n_max=20000), 1e-6)
        ax.imshow(shade(ids, st).reshape(RES, RES, 3), extent=[-cam.tf, cam.tf, -cam.tf, cam.tf])
        alpha = np.arcsin(np.sin(a1) / np.sin(d1))
        cx = 0.0
        th = np.linspace(0, 2 * np.pi, 200)
        rr = np.tan(alpha)
        ax.plot(cx + rr * np.cos(th), rr * np.sin(th), "w--", lw=0.8, label=f"M21: α = {np.degrees(alpha):.2f}°")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        stats[title] = dict(steps_mean=float(steps.mean()), frac_hit=float((ids >= 0).mean()))
    fig.tight_layout()
    fig.savefig(out / "s3_front_back.png", dpi=110)
    # --- S¹×R²: геодезический против евклидова --------------------------------------------------
    sh = SHAPES["S1xR2"]
    x0 = np.array([1.0, 0, 0, 0])
    cam = make_camera(sh, x0, [0, np.cos(0.3), np.sin(0.3), 0.05], [0, 0, 0, 1.0], fov=90, width=RES, height=RES)
    U, V = pixel_uv(cam)
    D = pixel_dirs(cam, U, V).reshape(-1, 4)
    g = cylinder_intrinsic_markers(1.0, BALLS[:, :3], BALLS[:, 3])
    ids, s, st, steps = trace_markers(sh, np.broadcast_to(x0, D.shape), D, g,
                                      Config(tol=1e-3 * cam.theta_pix, s_max=S_MAX, n_max=50000), 1e-7)
    eid, et, margin = euclid_trace(np.zeros(3), np.stack([D[:, 1], D[:, 2], D[:, 3]], 1), BALLS, S_MAX)
    shade_d = lambda i, t: np.where((i >= 0)[:, None], MARKER_COLORS[np.maximum(i, 0) % 6] * np.exp(-0.06 * np.nan_to_num(t, posinf=0))[:, None], NO_HIT)
    img_g = shade_d(ids, s).reshape(RES, RES, 3)
    img_e = shade_d(eid, et).reshape(RES, RES, 3)
    mism = (ids != eid).reshape(RES, RES)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_g)
    axs[0].set_title("S¹×R²: геодезический трассировщик в R⁴")
    axs[1].imshow(img_e)
    axs[1].set_title("евклидов, развёртка с периодом 2πR")
    axs[2].imshow(mism, cmap="gray")
    axs[2].set_title(f"расхождение id: {mism.sum()} пикс. (силуэт < 1e-6: {(margin < 1e-6).sum()})")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "s1r2_vs_euclid.png", dpi=110)
    hit = (eid >= 0) & (ids == eid)
    stats["S1xR2"] = dict(id_mismatch=int(mism.sum()), silhouette_px=int((margin < 1e-6).sum()),
                          max_dt=float(np.max(np.abs(s[hit] - et[hit]))), steps_mean=float(steps.mean()))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
