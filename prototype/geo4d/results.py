"""Каталоги воспроизводимых результатов: results/stage-N/YYYY-MM-DD_<имя>/ (DD-006)."""
from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or None
    except Exception:
        return None


def meta():
    libs = {}
    for name in ("jax", "jaxlib", "numpy", "scipy", "matplotlib"):
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    import jax
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        os=platform.platform(), cpu=platform.processor(), python=sys.version.split()[0],
        gpu=_gpu(), jax_backend=jax.default_backend(), jax_x64=bool(jax.config.read("jax_enable_x64")),
        libs=libs)


def new_run(stage: int, name: str, config: dict) -> Path:
    """Создаёт каталог запуска с config.json, meta.json, cmd.txt. Повторный запуск в тот же день перезаписывает."""
    d = ROOT / "results" / f"stage-{stage}" / f"{datetime.date.today().isoformat()}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    m = meta()
    (d / "meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    exe = Path(sys.executable)
    try:
        exe = exe.relative_to(ROOT)
    except ValueError:
        pass
    script = Path(sys.argv[0]).resolve()
    try:
        script = script.relative_to(ROOT)
    except ValueError:
        pass
    (d / "cmd.txt").write_text(" ".join([exe.as_posix(), script.as_posix(), *sys.argv[1:]]) + "\n", encoding="utf-8")
    if m["git_dirty"]:
        print(f"ВНИМАНИЕ: рабочее дерево не закоммичено — числа из {d} нельзя ссылать в документах", file=sys.stderr)
    return d
