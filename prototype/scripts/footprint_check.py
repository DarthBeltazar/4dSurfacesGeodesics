"""Этап 2: проверка следа режима A без текстуры — эмпирическая ковариация направлений выхода стохастических
лучей пикселя (гаусс пикселя + гаусс апертуры) против Σ из полей Якоби (M18)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo4d.shapes import SHAPES
from geo4d import wormholes as wh
from geo4d.camera import make_camera
from geo4d.cubemap import bake, build_mips
from geo4d.integrators import Config, DONE_ESCAPE
from geo4d import modea as MA
from geo4d.results import new_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--npix", type=int, default=200)
    ap.add_argument("--spp", type=int, default=4096)
    a = ap.parse_args()
    configs = dict(pinhole=dict(aperture=0.0, focus=6.0), dof=dict(aperture=0.3, focus=6.0),
                   dof_small=dict(aperture=0.05, focus=6.0))
    out = new_run(2, "footprint_check", dict(vars(a), configs=configs, scene="ellis, камера ℓ = 6, fov 90°",
                                             metric="‖C_emp − Σ‖_F / ‖Σ‖_F; смещение среднего в σ₊"))
    sh = SHAPES["ellis"]
    mips = build_mips(bake(lambda d, s: d * 0, 8))           # текстура не используется
    x_cam, e_ell = wh.ellis_point(6.0)
    esc, ex = wh.ellis_escape(20.0), wh.ellis_exit
    summary = {}
    for name, cc in configs.items():
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=a.res, height=a.res, **cc)
        cfg = Config(tol=1e-3 * cam.theta_pix, s_max=1e4, n_max=4000)
        tr = MA.trace_central(sh, cam, cfg, esc, ex)
        flags, _ = MA.detector_flags(cam, tr, MA.Detector())
        rng = np.random.default_rng(0)
        pix = rng.choice(np.nonzero(~flags)[0], a.npix, replace=False)
        D, S, St = MA.stochastic_pixels(sh, cam, pix, a.spp, cfg, esc, ex, mips, seed=5, return_dirs=True)
        a1, s1, a2, s2 = MA.footprint(tr["dd"][pix], cam.aperture)
        Sig = s1[:, None, None] ** 2 * np.einsum("ni,nj->nij", a1, a1) + s2[:, None, None] ** 2 * np.einsum("ni,nj->nij", a2, a2)
        rel, bias, theory = [], [], []
        for k in range(len(pix)):
            ok = (St[k] == DONE_ESCAPE) & (S[k] == tr["sheet"][pix[k]])
            d0 = tr["d"][pix[k]]
            dev = D[k][ok] - d0
            dev = dev - (dev @ d0)[:, None] * d0                       # касательная плоскость к сфере направлений
            C = np.cov(dev.T)
            rel.append(np.linalg.norm(C - Sig[k]) / np.linalg.norm(Sig[k]))
            bias.append(np.linalg.norm(dev.mean(0)) / s1[k])
            theory.append(np.sqrt(2.0 / ok.sum()))                    # стат. ошибка оценки ковариации
        rel, bias = np.array(rel), np.array(bias)
        summary[name] = dict(rel_cov_median=float(np.median(rel)), rel_cov_p95=float(np.percentile(rel, 95)),
                             rel_cov_max=float(rel.max()), stat_noise_expected=float(np.median(theory)),
                             mean_bias_median_sigma=float(np.median(bias)), mean_bias_p95_sigma=float(np.percentile(bias, 95)),
                             sigma_major_px_median=float(np.median(s1) / cam.theta_pix))
        print(name, summary[name], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
