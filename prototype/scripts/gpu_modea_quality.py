"""Этап 3 (DD-004 п. 8): качество режима A на GPU против эталона 1024 spp этапа 2 (тот же кадр 128², та же
процедурная кубмапа 512). Калибровка k для SampleGrad, BC6H против RGBA16F, режим детектора.
Отдельно — против изображения режима A прототипа с точным гауссовым фильтром (разделяет ошибку фильтра)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from geo4d import gpu, gpu_modea as GM, wormholes as wh
from geo4d import modea as MA
from geo4d.shapes import SHAPES
from geo4d.camera import make_camera
from geo4d.cubemap import bake
from geo4d.render import background
from geo4d.results import new_run, ROOT

STAGE2 = ROOT / "results" / "stage-2" / "2026-10-24_modea_vs_reference"
KS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
FORMATS = ["rgba16f", "bc6h"]
DETECTORS = {"по умолчанию (τ=2, |det|∈[0.1,10])": MA.Detector(),
             "только смена листа": MA.Detector(det_lo=0.0, det_hi=1e30, tau=1e30)}


def main():
    cfgs = json.loads((STAGE2 / "config.json").read_text(encoding="utf-8"))["configs"]
    out = new_run(3, "gpu_modea_quality", dict(ks=KS, formats=FORMATS, detectors=list(DETECTORS), stage2_configs=cfgs,
                                               reference=str(STAGE2.relative_to(ROOT)), nfb=256, kappa=1e-3,
                                               slangc=gpu.slangc_version()))
    sh = SHAPES["ellis"]
    cube = gpu.cube_for_gpu(bake(background, 512))
    x_cam, e_ell = wh.ellis_point(6.0)
    rows = []
    for name, cc in cfgs.items():
        imgs = np.load(STAGE2 / f"{name}_images.npy")        # [mode A мипы, точный фильтр, A+фолбэк, ref1, ref2, fb]
        ref, exact = 0.5 * (imgs[3] + imgs[4]), imgs[1]
        floor = MA.psnr(imgs[3], imgs[4]) + 10 * np.log10(4)
        cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=128, height=128, **cc)
        tol = 1e-3 * cam.theta_pix
        for fmt in FORMATS:
            for k in KS + ["M19"]:
                for dname, det in DETECTORS.items():
                    fm = 1 if k == "M19" else 0                     # M19 — программный фильтр (k не используется)
                    r = GM.render("ellis", cam, cube, tol, 20.0, k_grad=2.0 if fm else k, det=det, nfb=256, fmt=fmt,
                                  filter_mode=fm)
                    row = dict(config=name, format=fmt, k=k, detector=dname,
                               psnr_ref_nofb=MA.psnr(r["color_nofb"], ref), psnr_ref=MA.psnr(r["color"], ref),
                               psnr_vs_exact_filter=MA.psnr(r["color_nofb"], exact), fallback_frac=float(r["flags"].mean()),
                               ref_floor=floor, proto_mips_nofb=MA.psnr(imgs[0], ref), proto_exact_nofb=MA.psnr(exact, ref))
                    rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    # стоимость оттенения в 1080p: аппаратный SampleGrad против программного M19
    cam = make_camera(sh, x_cam, -e_ell, [0, 0, 1, 0], [0, 1, 0, 0], fov=90, width=1920, height=1080, focus=6.0,
                      aperture=0.3)
    shade_ms = {}
    for fm in [0, 1]:
        r = GM.render("ellis", cam, cube, 3e-3 * cam.theta_pix, 20.0, filter_mode=fm, fallback=False, fmt="bc6h")
        shade_ms["M19" if fm else "SampleGrad"] = float(np.median(r["reports"]["shade"]["times_ns"]) / 1e6)
    (out / "shade_1080p_ms.json").write_text(json.dumps(shade_ms, indent=1), encoding="utf-8")
    print("оттенение 1080p, мс:", shade_ms)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["| конфигурация | формат | k | детектор | PSNR без фолбэка | PSNR с фолбэком | против точного фильтра | фолбэк |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['config']} | {r['format']} | {r['k']} | {r['detector']} | {r['psnr_ref_nofb']:.2f} | "
                  f"{r['psnr_ref']:.2f} | {r['psnr_vs_exact_filter']:.2f} | {r['fallback_frac'] * 100:.2f} % |")
    md.append("")
    for name in cfgs:
        r0 = next(r for r in rows if r["config"] == name)
        md.append(f"{name}: прототип (мипы) {r0['proto_mips_nofb']:.2f} дБ, прототип (точный фильтр) "
                  f"{r0['proto_exact_nofb']:.2f} дБ, шумовой пол эталона {r0['ref_floor']:.2f} дБ.")
    (out / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(out)


if __name__ == "__main__":
    main()
