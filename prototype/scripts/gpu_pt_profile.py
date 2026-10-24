"""Этап 4 (DD-005, план проверки п. 11): профиль кадра режима B на GPU — 1080p и 4K, «интерактив» (1 spp, фолбэк 8)
и «финал» (64 spp, фолбэк 64) + денойзер OptiX. Время — метки GPU по проходам (медиана; у трассировщика — сумма
медиан сэмплов), без пересылок хост↔GPU стенда. Доли классов пикселей, проходов Jet7 на путь, регистры."""
import argparse
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
from geo4d.modea import tonemap01
from geo4d.wormholes import ellis_point
from geo4d.results import new_run

REGS_PER_SM, MAX_WARPS = 65536, 48


def occupancy(regs):
    alloc = ((int(regs) + 7) // 8) * 8
    return min(MAX_WARPS, REGS_PER_SM // (alloc * 32)) / MAX_WARPS


def setup(name, W, H, mips):
    up = np.array([0, 0, 1.0, 0])
    if name == "two_wormholes":
        sc = PT.scene_two_wormholes(mips=mips)
        cam = make_camera(sc.shape, np.array([0, -11.0, 0.5, 1.5]), np.array([0, 1.0, -0.12, 0]), up, fov=75, width=W,
                          height=H)
    else:
        sc = PT.scene_ellis(mips=mips)
        x, e = ellis_point(6.0)
        cam = make_camera(sc.shape, x, -e, up, fov=75, width=W, height=H)
    return sc, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--fmt", default="bc6h")
    a = ap.parse_args()
    p = G.GPUPT()
    out = new_run(4, "gpu_pt_profile", dict(vars(a), gpupt=p.__dict__, slangc=gpu.slangc_version(),
                                            modes=dict(interactive=dict(spp=1, nfb=8), final=dict(spp=64, nfb=64))))
    mips = PT.procedural_env(a.cube)
    rows = []
    for name in ["two_wormholes", "ellis"]:
        for W, H in [(1920, 1080), (3840, 2160)]:
            sc, cam = setup(name, W, H, mips)
            for mode, spp, nfb in [("interactive", 1, 8), ("final", 64, 64)]:
                fr = G.render_frame(sc, cam, spp, p, nfb=nfb, fmt=a.fmt)
                t = fr["times"]
                pt_ms = float(np.median(t["pt"])) * spp if t["pt"] else 0.0
                fb_ms = float(np.median(t["fallback"])) * nfb if t["fallback"] else 0.0
                alb, nrm = fr["aov"]
                den, drep = gpu.denoise(fr["color"].reshape(H, W, 3), alb.reshape(H, W, 3), nrm.reshape(H, W, 3), repeat=5)
                dn_ms = float(np.median(drep["times_ms"]))
                cls = fr["cls"]
                bg = cls == 0
                final = den.reshape(-1, 3).copy()
                final[bg & ~fr["flags"]] = fr["color"][bg & ~fr["flags"]]     # фон режима A — точный, не из денойзера
                st = fr["stats"]
                npath = max((cls != 0).sum() * spp, 1)
                regs_pt = fr["reports"]["pt"]["stats"].get("CS/Register Count", "?")
                regs_cam = fr["reports"]["camera"]["stats"].get("CS/Register Count", "?")
                row = dict(scene=name, res=f"{W}x{H}", mode=mode, spp=spp, nfb=nfb,
                           camera_ms=float(np.median(t["camera"])), shade_ms=float(np.median(t["shade"])),
                           detect_ms=float(np.median(t["detect"])), pt_ms=pt_ms, fallback_ms=fb_ms, denoise_ms=dn_ms,
                           frac_bg=float(bg.mean()), frac_obj=float((cls == 1).mean()), frac_mixed=float((cls == 2).mean()),
                           frac_fb=float(fr["flags"].mean()), passes_per_path=float((st[cls != 0, 1] % 1e6).sum() / npath),
                           flat_steps_per_path=float(st[cls != 0, 2].sum() / npath),
                           capped_frac=float(st[cls != 0, 0].sum() / npath), regs_pt=regs_pt, regs_camera=regs_cam,
                           occ_pt=occupancy(regs_pt), occ_camera=occupancy(regs_cam))
                row["total_ms"] = sum(row[k] for k in ["camera_ms", "shade_ms", "detect_ms", "pt_ms", "fallback_ms", "denoise_ms"])
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if W == 1920:
                    plt.imsave(out / f"{name}_{mode}_1080p.png", tonemap01(final.reshape(H, W, 3)))
                    plt.imsave(out / f"{name}_{mode}_1080p_noisy.png", tonemap01(fr["color"].reshape(H, W, 3)))
    (out / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    hdr = "| сцена | разрешение | режим | камера | фон (оттен.+дет.) | трассировщик | фолбэк | денойзер | **всего, мс** | фон / объект / смеш. / фолбэк, % | проходов Jet7 на путь | регистры PT / камера |"
    L = [hdr, "|" + "---|" * 12]
    for r in rows:
        L.append(f"| {r['scene']} | {r['res']} | {r['mode']} ({r['spp']} spp, фолбэк {r['nfb']}) | {r['camera_ms']:.1f} | "
                 f"{r['shade_ms'] + r['detect_ms']:.2f} | {r['pt_ms']:.1f} | {r['fallback_ms']:.1f} | {r['denoise_ms']:.1f} | "
                 f"**{r['total_ms']:.1f}** | {100 * r['frac_bg']:.0f} / {100 * r['frac_obj']:.0f} / {100 * r['frac_mixed']:.1f} / "
                 f"{100 * r['frac_fb']:.1f} | {r['passes_per_path']:.0f} | {r['regs_pt']} / {r['regs_camera']} |")
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
