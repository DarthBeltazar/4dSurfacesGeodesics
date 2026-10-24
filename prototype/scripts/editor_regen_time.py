"""DD-008: время структурной перегенерации графа окна (то, что ждёт выход из режима редактирования после добавления
объекта): `app.py --scene <json> --graph-only` с новой структурой объектов, как её запускает хост.

    python prototype/scripts/editor_regen_time.py --res 1280x720 --runs 3"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo4d import scenefile as SF
from geo4d.results import new_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="two_wormholes")
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--cube", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    out = new_run(5, "editor_regen_time", vars(a))
    app = Path(__file__).resolve().parent / "app.py"
    times = []
    for i in range(a.runs):
        d = json.loads(SF.resolve(a.scene).read_text(encoding="utf-8"))
        d["objects"] += [dict(name=f"x{k}", kind="ball4", center=[1.0, 2.0 + k, 0.0, 1.5], r=0.3)
                         for k in range(i + 1)]                       # каждый прогон — новая структура (без кэша slangc)
        SF.save(out / f"scene_{i}.json", d)
        t0 = time.perf_counter()
        subprocess.run([sys.executable, str(app), "--scene", str(out / f"scene_{i}.json"), "--res", a.res,
                        "--cube", str(a.cube), "--graph-only", str(out / f"graph_{i}")], check=True,
                       capture_output=True)
        times.append(time.perf_counter() - t0)
    res = dict(seconds=times)
    (out / "times.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)


if __name__ == "__main__":
    main()
