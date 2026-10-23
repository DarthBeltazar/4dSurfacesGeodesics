// U1 на GPU: Jet7, HVP-jet и Jet15 в точках P с направлениями W (сравнение с прототипом float64).
// Выход на точку — 8 float4: (f, ∇f·w, wᵀHw, 0), ∇f, H·w (HVP), (f_hvp, b_hvp, f_15, 0), H (4 строки).
#include "scenes.slang"

struct PC { uint n; };
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gP;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gW;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gOut;

[shader("compute")] [numthreads(64, 1, 1)]
void jets_u1(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pc.n) return;
    float4 p = gP[i], w = gW[i];
    Jet7 s7[4];
    JetHVP sh[4];
    Jet15 s15[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        float4 e = float4(0); e[k] = 1.0;
        s7[k] = Jet7(p[k], w[k], 0, e);              // M4: (x_k, w_k, 0, e_k)
        sh[k] = JetHVP(p[k], w[k], e, float4(0));    // M23: (x_k, w_k, e_k, 0)
        s15[k] = Jet15(p[k], e, float4x4(0));        // M5: (x_k, e_k, 0)
    }
    Jet7 j7 = sdf(s7[0], s7[1], s7[2], s7[3]);
    JetHVP jh = sdf(sh[0], sh[1], sh[2], sh[3]);
    Jet15 j15 = sdf(s15[0], s15[1], s15[2], s15[3]);
    gOut[8 * i + 0] = float4(j7.a, j7.b, 2 * j7.c, 0);
    gOut[8 * i + 1] = j7.d;
    gOut[8 * i + 2] = jh.m;
    gOut[8 * i + 3] = float4(jh.a, jh.b, j15.a, 0);
    [ForceUnroll] for (int r = 0; r < 4; r++) gOut[8 * i + 4 + r] = j15.H[r];
}
