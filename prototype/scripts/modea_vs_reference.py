"""Этап 2: режим A (Якоби + фильтр кубмапы + детектор + фолбэк) против эталона 1024 spp на Эллисе.

Эталон — две независимые половины по 512 spp (шумовой пол = PSNR между половинами). Фолбэк-сэмплы считаются
один раз для всех пикселей (N_fb spp), и развёртка порогов детектора комбинирует их без перетрассировки.
PSNR — на тонмапленных значениях [0, 1].
"""
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
from geo4d import wormholes as wh
from geo4d.camera import make_camera
from geo4d.cubemap import bake, build_mips
from geo4d.render import background
from geo4d.integrators import Config
from geo4d import modea as MA
from geo4d.results import new_run

SWEEP_TAU = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, np.inf]
SWEEP_DET = [(0.5, 2.0), (0.25, 4.0), (0.1, 10.0), (0.0, np.inf)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--spp", type=int, default=1024)
    ap.add_argument("--nfb", type=int, default=256)
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--configs", default="pinhole,dof")
    a = ap.parse_args()
    configs = dict(pinhole=dict(aperture=0.0, focus=6.0), dof=dict(aperture=0.3, focus=6.0))
    configs = {k: configs[k] for k in a.configs.split(",")}
    out = new_run(2, "modea_vs_reference", dict(vars(a), configs=configs, sigma_pix=MA.SIGMA_PIX,
                                                default_detector=MA.Detector().__dict__,
                                                sweep_tau=[str(t) for t in SWEEP_TAU], sweep_det=[list(map(str, d)) for d in SWEEP_DET],
                                                scene="ellis, камера ℓ = 6, fov 90°", psnr="тонмапленные [0, 1]"))
    sh = SHAPES["ellis"]
    t0 = time.time()
    mips = build_mips(bake(background, a.cube))
    print(f"кубмапа {a.cube}: {time.time() - t0:.1f} с", flush=True)
    x_cam, e_ell = wh.ellis_point(6.0)
    esc, ex = wh.ellis_escape(20.0), wh.ellis_exit
    summary = {}
    for name, cc in configs.items():
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=a.res, height=a.res, **cc)
        cfg = Config(tol=a.kappa * cam.theta_pix, s_max=1e4, n_max=4000)
        allpix = np.arange(a.res * a.res)
        t0 = time.time()
        tr = MA.trace_central(sh, cam, cfg, esc, ex)
        col_a, L, n = MA.shade_mode_a(tr, mips, cam.aperture)
        t_modea = time.time() - t0
        col_exact, _, _ = MA.shade_mode_a(tr, mips, cam.aperture, exact_filter=True)
        t0 = time.time()
        half = a.spp // 2
        ref1 = MA.stochastic_pixels(sh, cam, allpix, half, cfg, esc, ex, mips, seed=1)
        ref2 = MA.stochastic_pixels(sh, cam, allpix, half, cfg, esc, ex, mips, seed=2)
        t_ref = time.time() - t0
        ref = 0.5 * (ref1 + ref2)
        fb_s = MA.stochastic_pixels(sh, cam, allpix, a.nfb, cfg, esc, ex, mips, seed=3, return_samples=True)
        fb, fb64 = fb_s.mean(1), fb_s[:, :64].mean(1)
        extra = {}
        if cam.aperture > 0:
            disk1 = MA.stochastic_pixels(sh, cam, allpix, 256, cfg, esc, ex, mips, seed=4, aperture="disk")
            disk2 = MA.stochastic_pixels(sh, cam, allpix, 256, cfg, esc, ex, mips, seed=6, aperture="disk")
            disk = 0.5 * (disk1 + disk2)
            extra["psnr_modeA_vs_disk512"] = MA.psnr(col_a, disk)
            extra["psnr_modeA_vs_gauss_ref_same512"] = MA.psnr(col_a, ref1)
            extra["psnr_disk_halves"] = MA.psnr(disk1, disk2)
        flags, diag = MA.detector_flags(cam, tr, MA.Detector())
        comb = np.where(flags[:, None], fb, col_a)
        floor_halves = MA.psnr(ref1, ref2)
        s = dict(theta_pix=cam.theta_pix, tol=cfg.tol, time_modeA_s=t_modea, time_ref_s=t_ref,
                 nsdf15_mean=float(tr["nsdf15"].mean()), mip_level_mean=float(L.mean()), taps_mean=float(n.mean()),
                 psnr_modeA_no_fallback=MA.psnr(col_a, ref), psnr_modeA_default=MA.psnr(comb, ref),
                 psnr_fb_only=MA.psnr(fb, ref), psnr_fb64_only=MA.psnr(fb64, ref),
                 psnr_modeA_default_fb64=MA.psnr(np.where(flags[:, None], fb64, col_a), ref),
                 psnr_modeA_exact_filter_no_fallback=MA.psnr(col_exact, ref),
                 psnr_modeA_exact_filter_default=MA.psnr(np.where(flags[:, None], fb, col_exact), ref),
                 psnr_modeA_exact_filter_unflagged=MA.psnr(col_exact[~flags], ref[~flags]),
                 psnr_ref_halves_unflagged=MA.psnr(ref1[~flags], ref2[~flags]),
                 psnr_ref_floor_est_unflagged=MA.psnr(ref1[~flags], ref2[~flags]) + 10 * np.log10(4),
                 fallback_frac_default=float(flags.mean()),
                 psnr_ref_halves=floor_halves, psnr_ref_floor_est=floor_halves + 10 * np.log10(4),
                 psnr_modeA_unflagged=MA.psnr(col_a[~flags], ref[~flags]), **extra)
        sweep = []
        for lo, hi in SWEEP_DET:
            for tau in SWEEP_TAU:
                fl, _ = MA.detector_flags(cam, tr, MA.Detector(det_lo=lo, det_hi=hi, tau=tau))
                img = np.where(fl[:, None], fb, col_a)
                img_x = np.where(fl[:, None], fb, col_exact)
                sweep.append(dict(det_lo=lo, det_hi=str(hi), tau=str(tau), fallback_frac=float(fl.mean()),
                                  psnr=MA.psnr(img, ref), psnr_exact_filter=MA.psnr(img_x, ref)))
        s["sweep"] = sweep
        summary[name] = s
        print(name, json.dumps({k: v for k, v in s.items() if k != "sweep"}, indent=1), flush=True)
        R = a.res
        tm = lambda c: MA.tonemap01(c.reshape(R, R, 3))
        err = np.abs(MA.tonemap01(comb) - MA.tonemap01(ref)).max(-1).reshape(R, R)
        np.save(out / f"{name}_images.npy", np.stack([col_a, col_exact, comb, ref1, ref2, fb]).astype(np.float32))
        fig, axs = plt.subplots(1, 5, figsize=(22, 4.6))
        for ax, img, t in [(axs[0], tm(col_a), "режим A (без фолбэка)"), (axs[1], tm(comb), "режим A + фолбэк"),
                           (axs[2], tm(ref), f"эталон {a.spp} spp")]:
            ax.imshow(img)
            ax.set_title(t)
            ax.axis("off")
        im = axs[3].imshow(err, cmap="magma", vmin=0, vmax=0.1)
        axs[3].set_title("|A+фолбэк − эталон| (тонмап)")
        axs[3].axis("off")
        fig.colorbar(im, ax=axs[3], fraction=0.046)
        axs[4].imshow(flags.reshape(R, R), cmap="gray")
        axs[4].set_title(f"фолбэк: {flags.mean() * 100:.1f} % пикселей")
        axs[4].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"{name}_compare.png", dpi=110)
        plt.imsave(out / f"{name}_modeA.png", tm(comb))
        plt.imsave(out / f"{name}_reference.png", tm(ref))
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for lo, hi in SWEEP_DET:
            pts = [(r["fallback_frac"], r["psnr"]) for r in sweep if r["det_lo"] == lo]
            ax.plot(*zip(*pts), "o-", ms=3, label=f"мипы, |det| ∈ [{lo}, {hi}]")
            pts = [(r["fallback_frac"], r["psnr_exact_filter"]) for r in sweep if r["det_lo"] == lo]
            ax.plot(*zip(*pts), "s--", ms=3, label=f"точный фильтр, |det| ∈ [{lo}, {hi}]")
        ax.axhline(floor_halves + 10 * np.log10(4), color="gray", ls=":", label="оценка шумового пола эталона")
        ax.set_xlabel("доля пикселей на фолбэке")
        ax.set_ylabel("PSNR, дБ")
        ax.set_title(f"{name}: развёртка порога τ невязки")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{name}_detector_sweep.png", dpi=120)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
