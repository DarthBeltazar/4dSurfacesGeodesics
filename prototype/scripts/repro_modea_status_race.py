"""Воспроизведение известной проблемы (docs/known-issues.md, «Режим A: пурпурный пиксель изредка»): одна и та же
секция кадра move (две горловины, 320×180) в режиме graph, N запусков; печатает запуски, в которых буферы кадра
отличаются от первого, и номера пикселей. Ожидается: в ~10–40 % запусков отличается только Color в пикселях 8192
и/или 24576 (поток 0 групп 128 и 384): modea_shade видит Status = 4, хотя pt_camera записал 2.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python prototype/scripts/repro_modea_status_race.py [N] [WxH]"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "gpu" / "build" / ("geo4d_gpu.exe" if sys.platform == "win32" else "geo4d_gpu")
W, H = 320, 180


def main():
    global W, H
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    if len(sys.argv) > 2:
        W, H = (int(v) for v in sys.argv[2].split("x"))
    N = W * H
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r = subprocess.run([str(EXE), "export-graph", str(ROOT / "scenes" / "two_wormholes.json"), str(td / "g"), "--res",
                            f"{W}x{H}", "--compile"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out, sec = [], None
        for line in (td / "g" / "app_graph.txt").read_text(encoding="utf-8").splitlines():
            w = line.split()
            if not w:
                continue
            if w[0] == "section":
                sec = w[1]
            elif sec is None and w[0] in ("buffer", "cube", "pipeline", "push"):
                out.append(f"buffer {w[1]} zero {w[3]}" if w[0] == "buffer" and w[2] == "export" else line)
            elif sec == "move":
                out.append(line)
        names = {"Class": ("u4", f"{N}"), "Status": ("u4", f"{N}"), "Flag": ("u4", f"{N}"), "Color": ("f4", f"{N} 4"),
                 "Dir": ("f4", f"{N} 4")}
        ref, bad = None, 0
        for k in range(n):
            o = td / f"r{k}"
            o.mkdir()
            g = out + [f"output {b} {o / (b + '.npy')} {t} {s}" for b, (t, s) in names.items()]
            (o / "graph.txt").write_text("\n".join(g) + "\n", encoding="utf-8")
            r = subprocess.run([str(EXE), "graph", str(o / "graph.txt")], capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            cur = {b: np.load(o / f"{b}.npy").reshape(N, -1) for b in names}
            if ref is None:
                ref = cur
                continue
            diff = {b: np.flatnonzero(np.any(cur[b] != ref[b], axis=1)).tolist() for b in names}
            diff = {b: v for b, v in diff.items() if v}
            if diff:
                bad += 1
                print(k, diff, "Color:", [cur["Color"][i].tolist() for i in diff.get("Color", [])[:2]])
        print(f"отличаются: {bad} из {n - 1}")


if __name__ == "__main__":
    main()
