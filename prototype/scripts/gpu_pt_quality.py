"""Этап 4 (DD-005, план проверки п. 9–10): качество режима B на GPU, сцена 3, 480×270, эталон — трассировщик
4096 spp при tol 1e-4 без роста допуска и мипов.
1. Смещение от роста допуска с конусом (M30) и мипов фона по конусу (M31): PSNR, z среднего по кадру, проходы, мс.
2. Денойзер OptiX: PSNR шумного и денойзенного кадра при 1/4/16/64 spp; без AOV / +альбедо / +альбедо+нормаль.
3. Кадр целиком (фон — режим A) против эталона."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo4d import gpu, gpu_pt as G, pathtrace as PT
from geo4d.camera import make_camera
from geo4d.modea import tonemap01, psnr
from geo4d.results import new_run

W, H, NREF = 480, 270, 4096
REF = dict(tol_kappa=1e-3, tol_max=1e-4, k_cone=0.0, k_lod=0.0)


def main():
    out = new_run(4, "gpu_pt_quality", dict(W=W, H=H, nref=NREF, ref=REF, default=G.GPUPT().__dict__, cube=256,
                                            slangc=gpu.slangc_version()))
    sc = PT.scene_two_wormholes(mips=PT.procedural_env(256))
    cam = make_camera(sc.shape, np.array([0, -11.0, 0.5, 1.5]), np.array([0, 1.0, -0.12, 0]), np.array([0, 0, 1.0, 0]),
                      fov=75, width=W, height=H)
    ref, vref, _, _ = G.render(sc, cam, NREF, G.GPUPT(**REF, seed=1))
    ref2, vref2, _, _ = G.render(sc, cam, NREF, G.GPUPT(**REF, seed=2))
    res = dict(noise_floor_psnr=psnr(ref, ref2))
    np.save(out / "reference.npy", ref.astype(np.float32))
    sweep = []
    for kc, kl in [(0, 0), (0.02, 0), (0.05, 0), (0.1, 0), (0.25, 0), (1.0, 0), (0.05, 1), (0.25, 1)]:
        m, v, st, rep = G.render(sc, cam, NREF, G.GPUPT(tol_kappa=3e-3, tol_max=1e-2, k_cone=kc, k_lod=kl, seed=3))
        ms = float(np.median([t / 1e6 for r in rep for t in r["times_ns"]]))
        sweep.append(dict(k_cone=kc, k_lod=kl, psnr=psnr(m, ref), img_z=float((m.mean() - ref.mean()) / np.sqrt((v + vref).sum() / m.size ** 2)),
                          rel=float(m.mean() / ref.mean() - 1), var_ratio=float(v.mean() / vref.mean()),
                          passes=float((st[:, 1] % 1e6).sum() / (W * H * NREF)), ms_per_spp=ms))
        print(json.dumps(sweep[-1]), flush=True)
    res["cone_sweep"] = sweep
    den = []
    for spp in [1, 4, 16, 64]:
        m, v, st, _ = G.render(sc, cam, spp, G.GPUPT(seed=100 + spp))
        alb, nrm = G.render.aov
        row = dict(spp=spp, noisy=psnr(m, ref))
        for name, aa, nn in [("color", None, None), ("albedo", alb, None), ("albedo_normal", alb, nrm)]:
            d, rep = gpu.denoise(m.reshape(H, W, 3), None if aa is None else aa.reshape(H, W, 3),
                                 None if nn is None else nn.reshape(H, W, 3), repeat=3)
            row[name] = psnr(d.reshape(-1, 3), ref)
            if name == "albedo_normal" and spp in (1, 16):
                plt.imsave(out / f"denoised_{spp}spp.png", tonemap01(d))
                plt.imsave(out / f"noisy_{spp}spp.png", tonemap01(m.reshape(H, W, 3)))
        den.append(row)
        print(json.dumps(row), flush=True)
    res["denoise"] = den
    fr = G.render_frame(sc, cam, 64, G.GPUPT(seed=7), nfb=64)
    bg = fr["cls"] == 0
    res["frame"] = dict(bg_psnr=psnr(fr["color"][bg], ref[bg]), frac_bg=float(bg.mean()), frac_fb=float(fr["flags"].mean()))
    plt.imsave(out / "reference.png", tonemap01(ref.reshape(H, W, 3)))
    plt.imsave(out / "classes.png", fr["cls"].reshape(H, W), cmap="viridis", vmin=0, vmax=2)
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = [f"Шумовой пол эталона (два прогона по {NREF} spp): {res['noise_floor_psnr']:.1f} дБ", "",
         "| k_cone | k_lod | PSNR, дБ | z среднего | отн. смещение | дисперсия / эталон | проходов на путь | мс на сэмпл |",
         "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['k_cone']} | {r['k_lod']} | {r['psnr']:.1f} | {r['img_z']:.1f} | {r['rel']:+.4f} | {r['var_ratio']:.3f} | "
          f"{r['passes']:.0f} | {r['ms_per_spp']:.2f} |" for r in sweep]
    L += ["", "| spp | шумный | денойз: цвет | + альбедо | + альбедо + нормаль |", "|---|---|---|---|---|"]
    L += [f"| {r['spp']} | {r['noisy']:.1f} | {r['color']:.1f} | {r['albedo']:.1f} | {r['albedo_normal']:.1f} |" for r in den]
    L += ["", f"Кадр (64 spp, фон — режим A): фон {100 * res['frame']['frac_bg']:.0f} % пикселей, PSNR фона против эталона "
          f"{res['frame']['bg_psnr']:.1f} дБ; на фолбэке {100 * res['frame']['frac_fb']:.1f} %."]
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
