"""Генерирует gpu/shaders/bc6h_core.slang из vendored compress.hlsl: функции кодирования без глобальных ресурсов
и без CSMain (они заменены своей точкой входа в bc6h_encode.slang)."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]  # gpu/
src = (root / "third_party" / "bc6h" / "compress.hlsl").read_text(encoding="utf-8").splitlines()
start_res = next(i for i, l in enumerate(src) if l.startswith("Texture2D SrcTexture"))
end_cb = next(i for i in range(start_res, len(src)) if src[i].startswith("};"))
cs = next(i for i, l in enumerate(src) if l.startswith("[numthreads(8, 8, 1)]"))
body = src[:start_res] + src[end_cb + 1:cs]
# HLSL (fxc) неявно приводит float3 → uint3 в f16tof32; Slang требует явного приведения (семантика та же)
n_fix = sum(l.count("f16tof32(clamp(") for l in body)
body = [l.replace("f16tof32(clamp(", "f16tof32((uint3)clamp(") for l in body]
assert n_fix == 4, n_fix
header = [
    "// Функции кодирования BC6H из GPURealTimeBC6H (K. Narkowicz, MIT, см. gpu/third_party/bc6h/LICENSE,",
    "// коммит в UPSTREAM_COMMIT.txt). Сгенерировано из compress.hlsl: удалены глобальные ресурсы",
    "// (SrcTexture, OutputTexture, PointSampler, MainCB) и точка входа CSMain; в 4 вызовах f16tof32 явное",
    "// приведение (uint3) вместо неявного (HLSL); остальное без изменений.",
    "#ifndef QUALITY",
    "#define QUALITY 1",
    "#endif",
]
(root / "shaders" / "bc6h_core.slang").write_text("\n".join(header + body) + "\n", encoding="utf-8")
print("ok", len(body))
