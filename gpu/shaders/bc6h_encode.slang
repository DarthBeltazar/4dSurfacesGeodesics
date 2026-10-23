// Кодирование уровня кубмапы в BC6H (UFLOAT) на GPU (DD-004 п. 8): поток на блок 4×4.
// Буферы: 0 тексели уровня (float4, [слой][y][x]), 1 блоки (uint4, [слой][by][bx]).
#include "bc6h_core.slang"

struct PC { uint size; uint layers; };     // сторона уровня в текселях, число слоёв (листы × 6)
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gTexels;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> gBlocks;

[shader("compute")] [numthreads(8, 8, 1)]
void bc6h_encode(uint3 id : SV_DispatchThreadID)
{
    uint nb = (pc.size + 3) / 4;
    uint layer = id.z;
    if (id.x >= nb || id.y >= nb || layer >= pc.layers) return;
    float3 texels[16];
    [ForceUnroll] for (int k = 0; k < 16; k++)
    {
        uint x = min(id.x * 4 + uint(k % 4), pc.size - 1), y = min(id.y * 4 + uint(k / 4), pc.size - 1);
        texels[k] = max(gTexels[(layer * pc.size + y) * pc.size + x].xyz, 0.0);
    }
    uint4 block = uint4(0, 0, 0, 0);
    float blockMSLE = 0.0f;
    EncodeP1(block, blockMSLE, texels);
#if ENCODE_P2
    float bestScore = EvaluateP2Pattern(0, texels);
    uint bestPattern = 0;
    for (uint p = 1; p < 32; ++p)
    {
        float score = EvaluateP2Pattern(p, texels);
        if (score < bestScore) { bestPattern = p; bestScore = score; }
    }
    EncodeP2Pattern(block, blockMSLE, bestPattern, texels);
#endif
    gBlocks[(layer * nb + id.y) * nb + id.x] = block;
}
