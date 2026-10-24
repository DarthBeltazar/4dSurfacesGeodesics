"""Сборка переносимого архива редактора сцен для Windows (релиз): хост geo4d_gpu.exe с SDL3 и рантаймом MSVC, slangc
из Vulkan SDK, Python-часть окна (geo4d, app.py), шейдеры, сцены, заранее скомпилированные SPIR-V для всех сцен
scenes/*.json (быстрый и динамический графы) и run_editor.cmd (при первом запуске создаёт .venv и ставит
requirements.txt). Нужен свежий gpu\\build.cmd (профиль release).

    python prototype/scripts/package_windows.py --version 0.1.0-alpha
Результат: dist/geo4d-editor-<версия>-win64.zip (каталог dist/ не в git)."""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SDK = Path(os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.328.1"))
VS_REDIST = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC")
CRT = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
SLANG = ["slangc.exe", "slang.dll", "slang-glslang.dll", "slang-glsl-module.dll", "slang-rt.dll"]

RUN_CMD = r"""@echo off
rem geo4d scene editor: first run creates .venv (Python 3.12) and installs requirements.txt, then starts the window.
rem Arguments go to app.py, e.g.: run_editor.cmd --scene S3 --res 1920x1080 --fullscreen
rem (ASCII only: cmd.exe reads .cmd files in the OEM code page)
setlocal
set "ROOT=%~dp0"
set "GEO4D_SLANGC=%ROOT%slang\slangc.exe"
set PYTHONIOENCODING=utf-8
if not exist "%ROOT%.venv\Scripts\python.exe" (
    echo First run: creating the Python environment in .venv, this takes a few minutes...
    py -3.12 -m venv "%ROOT%.venv" 2>nul || python -m venv "%ROOT%.venv"
    if not exist "%ROOT%.venv\Scripts\python.exe" (
        echo Python 3.12 was not found. Install it from https://www.python.org/downloads/ and run again.
        pause
        exit /b 1
    )
    "%ROOT%.venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo pip install failed.
        rmdir /s /q "%ROOT%.venv"
        pause
        exit /b 1
    )
)
"%ROOT%.venv\Scripts\python.exe" "%ROOT%prototype\scripts\app.py" %*
if errorlevel 1 pause
"""

README = """geo4d — редактор сцен искривлённых пространств (червоточины, S³, T³ …), Windows x64, {version}
Исходники и описание: https://github.com/DarthBeltazar/4dSurfacesGeodesics

Требования
  * GPU с Vulkan 1.3 и свежий драйвер (проверено: NVIDIA RTX 4080 SUPER, драйвер 610.88). Денойзер OptiX — только
    NVIDIA; без него окно работает без денойзера.
  * Python 3.12 (https://www.python.org/downloads/), «py» или «python» в PATH. Vulkan SDK не нужен — slangc в slang\\.
  * ~1.5 ГБ на диске для .venv (JAX, NumPy, SciPy, Matplotlib), интернет при первом запуске.

Запуск
  run_editor.cmd                                   две червоточины, 1280×720
  run_editor.cmd --scene S3 --fullscreen           трёхмерная сфера на весь экран
  run_editor.cmd --scene my_scene.json --res 1920x1080
  Первый запуск: создаётся .venv и ставятся пакеты (несколько минут), строится фон (~10 с).

Управление
  W/S, A/D, Q/E — полёт (Shift — ×4); ЛКМ по кадру — взгляд мышью (Esc или ЛКМ — отпустить); стрелки — взгляд;
  C/Z — крен; F1 — панели; Tab — режим редактирования; F11 — полноэкранный режим; R — денойзер; Esc — выход.

Панели
  «Рендер» — экспозиция, скорость, разрешение рендера; «Многообразие» — вид пространства и параметры («Применить» —
  сразу, в режиме редактирования); «Файл» — открыть/сохранить сцену (scenes\\*.json); «Объекты» — шары, торы,
  плоскости, материалы, источники света. Новая форма/число объектов или другое пространство — сразу, в режиме
  редактирования (кадр медленнее); при выходе из него (Tab) быстрые шейдеры компилируются за 1–10 с.

Альфа-версия: проверено на одной машине (Windows 11, RTX 4080 SUPER). Сообщения об ошибках — в issues на GitHub.

Сторонние компоненты: Slang (slangc, Apache-2.0 WITH LLVM-exception, https://github.com/shader-slang/slang),
SDL3 (zlib), Dear ImGui (MIT), volk (MIT), Vulkan Memory Allocator (MIT), рантайм Microsoft Visual C++
(распространяемые файлы Visual Studio 2022). Лицензия проекта — LICENSE (MIT).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0-alpha")
    ap.add_argument("--no-spv", action="store_true", help="без заранее скомпилированных шейдеров")
    a = ap.parse_args()
    name = f"geo4d-editor-{a.version}-win64"
    out = ROOT / "dist" / name
    if out.exists():
        shutil.rmtree(out)
    (out / "gpu" / "build").mkdir(parents=True)
    exe = ROOT / "gpu" / "build" / "geo4d_gpu.exe"
    for f in [exe, ROOT / "gpu" / "build" / "SDL3.dll"]:
        shutil.copy2(f, out / "gpu" / "build")
    crt = sorted(VS_REDIST.glob("*/x64/Microsoft.VC143.CRT"))[-1]
    (out / "slang").mkdir()
    for f in CRT:                                     # рядом с exe и рядом с slangc (поиск DLL — каталог программы)
        shutil.copy2(crt / f, out / "gpu" / "build")
        shutil.copy2(crt / f, out / "slang")
    for f in SLANG:
        shutil.copy2(SDK / "Bin" / f, out / "slang")
    shutil.copytree(ROOT / "gpu" / "shaders", out / "gpu" / "shaders")
    shutil.copytree(ROOT / "prototype" / "geo4d", out / "prototype" / "geo4d",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (out / "prototype" / "scripts").mkdir()
    shutil.copy2(ROOT / "prototype" / "scripts" / "app.py", out / "prototype" / "scripts")
    shutil.copytree(ROOT / "scenes", out / "scenes")
    for f in ["requirements.txt", "LICENSE"]:
        shutil.copy2(ROOT / f, out)
    (out / "run_editor.cmd").write_bytes(RUN_CMD.replace("\n", "\r\n").encode("ascii"))
    (out / "README.txt").write_bytes(README.format(version=a.version).replace("\n", "\r\n").encode("utf-8-sig"))
    if not a.no_spv:          # шейдеры всех сцен — в кэш архива (ключ кэша — содержимое, не пути)
        env = dict(os.environ, GEO4D_SLANGC=str(out / "slang" / "slangc.exe"), PYTHONIOENCODING="utf-8")
        env.pop("GEO4D_GPU_EXE", None)
        tmp = out / "_graphs"
        for sc in sorted((out / "scenes").glob("*.json")):
            subprocess.run([sys.executable, str(out / "prototype" / "scripts" / "app.py"), "--scene", sc.stem,
                            "--cube", "64", "--graph-only", str(tmp / sc.stem)], check=True, env=env,
                           capture_output=True)
            print("шейдеры:", sc.stem)
        shutil.rmtree(tmp)
        for d in ["gen", "cache"]:                    # заголовки — генерируются заново; фон 64 — не тот, что в окне
            shutil.rmtree(out / "gpu" / "build" / d, ignore_errors=True)
        for p in (out / "prototype").rglob("__pycache__"):
            shutil.rmtree(p)
    zp = ROOT / "dist" / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(out))
    print(zp, f"{zp.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
