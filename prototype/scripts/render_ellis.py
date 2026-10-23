"""Этап 1: картинка сцены Эллиса и тепловая карта числа шагов / проходов SDF на пиксель."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d.shapes import SHAPES
from geo4d.integrators import Config, DONE_NMAX
from geo4d import wormholes as wh
from geo4d.render import pinhole_dirs, render_wormhole, tonemap
from geo4d.results import new_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--ell-cam", type=float, default=6.0)
    ap.add_argument("--ell-max", type=float, default=20.0)
    ap.add_argument("--kappa", type=float, default=1e-3, help="tol = κ·θ_pix (ADR-0002)")
    ap.add_argument("--n-max", type=int, default=4000)
    ap.add_argument("--name", default="ellis_render")
    a = ap.parse_args()

    sh = SHAPES["ellis"]
    b = sh.params["b"]
    x_cam, e_ell = wh.ellis_point(a.ell_cam, b)
    dirs, th_pix = pinhole_dirs(a.res, a.res, a.fov, -e_ell, np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0]))
    tol = a.kappa * th_pix
    cfg = Config(method="bs32", tol=tol, h=0.1 * b, s_max=1e4, n_max=a.n_max)
    out = new_run(1, a.name, dict(vars(a), tol=tol, theta_pix=th_pix, b=b, cfg=cfg.__dict__))

    t0 = time.time()
    col, steps, nsdf, status, _, _ = render_wormhole(sh, x_cam, dirs, cfg, wh.ellis_escape(a.ell_max, b),
                                               lambda x, v: wh.ellis_exit(x, v, b))
    dt = time.time() - t0
    R = a.res
    plt.imsave(out / "ellis.png", tonemap(col.reshape(R, R, 3)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, data, title in [(axs[0], steps, "попыток шага на пиксель"), (axs[1], nsdf, "проходов SDF на пиксель")]:
        im = ax.imshow(data.reshape(R, R), cmap="inferno", norm=matplotlib.colors.LogNorm())
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "ellis_steps_heatmap.png", dpi=120)
    stats = dict(time_s=dt, rays=int(len(dirs)), tol=tol, theta_pix=th_pix,
                 steps_mean=float(steps.mean()), steps_median=float(np.median(steps)), steps_max=int(steps.max()),
                 nsdf_mean=float(nsdf.mean()), nsdf_median=float(np.median(nsdf)), nsdf_p99=float(np.percentile(nsdf, 99)),
                 frac_step_cap=float(np.mean(status == DONE_NMAX)))
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    np.save(out / "nsdf.npy", nsdf.reshape(R, R).astype(np.int32))
    print(json.dumps(stats, indent=2))
    print(out)


if __name__ == "__main__":
    main()
