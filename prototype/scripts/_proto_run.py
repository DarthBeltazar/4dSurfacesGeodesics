"""Вспомогательный запуск прототипа в отдельном процессе (точность задаётся GEO4D_X64 до импорта geo4d).
Использование: python _proto_run.py <вход.npz> <выход.npz>; во входе: scene, X, V, E (опц.), tol, s_max, nj."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import geo4d  # noqa: F401  (точность)
from geo4d.shapes import SHAPES
from geo4d.integrators import make_integrator, Config


def main():
    d = np.load(sys.argv[1], allow_pickle=True)
    sh = SHAPES[str(d["scene"])]
    nj = int(d["nj"])
    cfg = Config(method=str(d["method"]), tol=float(d["tol"]), s_max=float(d["s_max"]), jacobi=nj, n_max=10 ** 6)
    run = make_integrator(sh, cfg)
    o = run(d["X"], d["V"], d["E"]) if nj else run(d["X"], d["V"])
    out = {k: np.asarray(o[k]) for k in ("x", "v", "s", "status")}
    out["passes"] = np.asarray(o["nsdf15"] if nj else o["nsdf"])
    if nj:
        out["E"] = np.asarray(o["E"])
    np.savez(sys.argv[2], **out)


if __name__ == "__main__":
    main()
