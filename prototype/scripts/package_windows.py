"""Windows standalone ZIP (DD-010): Python is used only to prepare the release.

Build gpu\\build.cmd first, then:
    .venv/Scripts/python prototype/scripts/package_windows.py --version 0.2.0-alpha
The packaged executable validates every scene using its bundled Slang DLLs.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VS_REDIST = Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Redist\MSVC")
CRT = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
DLLS = ["SDL3.dll", "slang.dll", "slang-glslang.dll", "slang-glsl-module.dll", "slang-rt.dll"]

RUN_CMD = r'''@echo off
rem Standalone editor. Example: run_editor.cmd scenes\bubble.g4d --res 1920x1080
setlocal
"%~dp0geo4d_gpu.exe" %*
if errorlevel 1 (
    pause
    exit /b 1
)
'''

README = """geo4d — редактор искривлённых пространств, Windows x64, {version}
Исходники этой сборки: https://github.com/DarthBeltazar/4dSurfacesGeodesics/tree/{commit}

Распакуйте архив и запустите run_editor.cmd или geo4d_gpu.exe.
Python, Vulkan SDK, установка пакетов и интернет для запуска не нужны.
Нужны Windows x64, GPU с Vulkan 1.3 и свежий драйвер. Денойзер OptiX — только NVIDIA;
без него приложение продолжает работать. AMD и Intel не проверены.

Примеры (аргумент сцены — путь, а не --scene):
  run_editor.cmd
  run_editor.cmd scenes\\S3.json --fullscreen
  run_editor.cmd scenes\\bubble.g4d --res 1920x1080

W/S, A/D, Q/E — полёт (Shift — быстрее); ЛКМ по кадру и мышь, стрелки — взгляд;
C/Z — крен; средняя кнопка по объекту — выбрать ноду; F1 — панели;
F11 — полноэкранный режим; R — денойзер; Esc — отпустить мышь / выход.

Панель «Граф»: ПКМ по холсту — добавить ноду; соединяйте порты, меняйте параметры
в инспекторе. Delete — удалить; Ctrl+Z / Ctrl+Y — отменить / повторить.
Сцена сохраняется в .g4d; прежние .json открываются с переводом в граф.
Структурная правка включает предпросмотр, быстрые шейдеры собираются в фоне.
Флажок P — предпросмотр. Текстуры и PBR-материал пока не реализованы.

Первый запуск компилирует шейдеры. Кэш: %LOCALAPPDATA%\\geo4d\\cache;
файлы программы не меняются. Не разносите exe, DLL, shaders/, nodes/ и scenes/.
Альфа: проверено на Windows с RTX 4080 SUPER. Изредка возможна пурпурная точка
в фоне (известная проблема режима A). Полный список ограничений — docs/known-issues.md
в исходниках. Ошибки: https://github.com/DarthBeltazar/4dSurfacesGeodesics/issues

Лицензия: GNU GPL v3 или более поздняя (LICENSE). Сторонние компоненты:
SDL3 (zlib), Dear ImGui и imgui-node-editor (MIT), nlohmann/json (MIT),
volk и Vulkan Memory Allocator (MIT), Slang (Apache-2.0 WITH LLVM-exception),
заголовки NVIDIA OptiX (NVIDIA SDK), распространяемые файлы Microsoft Visual C++.

Дополнительное разрешение по разделу 7 GNU GPL v3: если вы изменяете эту программу или любое основанное на ней
произведение, связывая или объединяя его с NVIDIA OptiX SDK (заголовки optix-dev) или драйверными библиотеками
NVIDIA CUDA (или их изменёнными версиями), содержащими части, на которые распространяются условия их лицензий,
правообладатели этой программы дают вам дополнительное разрешение распространять полученное произведение.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.2.0-alpha")
    ap.add_argument("--no-spv", action="store_true", help="пропустить проверку компиляции шейдеров всех сцен")
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", a.version):
        ap.error("invalid version")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        ap.error("commit tracked changes before packaging: the source link must match the build")
    name = f"geo4d-editor-{a.version}-win64"
    out = ROOT / "dist" / name
    # Verify the recursive-removal target stays inside the release directory.
    if out.resolve().parent != (ROOT / "dist").resolve():
        raise ValueError("unsafe output directory")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    build = ROOT / "gpu" / "build"
    for f in ["geo4d_gpu.exe", *DLLS]:
        shutil.copy2(build / f, out)
    crt = sorted(VS_REDIST.glob("*/x64/Microsoft.VC143.CRT"))[-1]
    for f in CRT:
        shutil.copy2(crt / f, out)
    for source, dest in [(ROOT / "gpu" / "shaders", "shaders"), (ROOT / "nodes", "nodes"),
                         (ROOT / "scenes", "scenes")]:
        shutil.copytree(source, out / dest)
    shutil.copy2(ROOT / "LICENSE", out)
    shutil.copytree(ROOT / "docs" / "licenses", out / "licenses")
    (out / "run_editor.cmd").write_bytes(RUN_CMD.replace("\n", "\r\n").encode("ascii"))
    (out / "README.txt").write_bytes(README.format(version=a.version, commit=commit).replace("\n", "\r\n").encode("utf-8-sig"))
    (out / "build.json").write_text(json.dumps({"version": a.version, "commit": commit}, indent=2) + "\n", encoding="utf-8")
    if not a.no_spv:
        validation = out / "_validation"
        env = {k: v for k, v in os.environ.items() if not k.startswith("GEO4D_")}
        env["GEO4D_CACHE_DIR"] = str(validation / "cache")
        scenes = sorted((out / "scenes").glob("*.json")) + sorted((out / "scenes").glob("*.g4d"))
        for sc in scenes:
            r = subprocess.run([str(out / "geo4d_gpu.exe"), "export-graph", str(sc),
                                str(validation / sc.name), "--cube", "64", "--compile"],
                               cwd=out, env=env, capture_output=True, text=True, encoding="utf-8")
            if r.returncode:
                raise RuntimeError(f"{sc.name}: {r.stdout}\n{r.stderr}")
            print("шейдеры:", sc.name, flush=True)
        shutil.rmtree(validation)
    zp = ROOT / "dist" / f"{name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(name) / f.relative_to(out))
    digest = hashlib.sha256(zp.read_bytes()).hexdigest()
    zp.with_suffix(".zip.sha256").write_text(f"{digest}  {zp.name}\n", encoding="ascii")
    print(zp, f"{zp.stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
