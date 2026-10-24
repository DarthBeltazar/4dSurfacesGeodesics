"""Обёртка geo4d_gpu для проверки Linux-сборки из WSL: переписывает пути Windows в job/graph-файле на /mnt/<диск>/…
и запускает ~/geo4d-linux/geo4d_gpu в WSL. Переменная GEO4D_WSL_ENV — доп. окружение (например SDL_VIDEO_DRIVER=offscreen)."""
import os
import re
import subprocess
import sys

def tr(s):
    return re.sub(r"\b([A-Za-z]):[\\/]([^\s\"]*)", lambda m: "/mnt/" + m.group(1).lower() + "/" + m.group(2).replace("\\", "/"), s)

args = sys.argv[1:]
if len(args) >= 2:
    src = args[1]
    txt = open(src, encoding="utf-8").read()
    dst = src + ".wsl"
    open(dst, "w", encoding="utf-8", newline="\n").write(tr(txt))
    args[1] = tr(dst)
env = os.environ.get("GEO4D_WSL_ENV", "").split()
cmd = ["wsl", "-d", "Ubuntu", "--", "env"] + env + ["/home/alexandrgeorgiev/geo4d-linux/geo4d_gpu"] + args
r = subprocess.run(cmd, capture_output=True)
sys.stdout.buffer.write(r.stdout); sys.stdout.flush()
sys.stderr.buffer.write(r.stderr); sys.stderr.flush()   # UTF-8 как есть
sys.exit(r.returncode)
