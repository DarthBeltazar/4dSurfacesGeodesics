// Ядро «trace» (мегаядро, поток на луч): сравнение с прототипом по фигурам (DD-004 п. 3).
// Буферы: 0 x (float4), 1 v (float4), 2 E (8 float4 на луч: J₀..J₃, J′₀..J′₃; при NJ = 0 не читается),
//         3 статистика (s, nacc, nrej, проходов jet), 4 статус (uint).
#include "integrator.slang"

[[vk::push_constant]] ConstantBuffer<Params> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gX;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gV;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gE;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gStats;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> gStatus;

[shader("compute")] [numthreads(64, 1, 1)]
void trace(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pc.n) return;
    RayState st;
    st.x = gX[i];
    st.v = gV[i];
#if NJ > 0
    [ForceUnroll] for (int k = 0; k < NJ; k++) { st.E.J[k] = gE[8 * i + k]; st.E.Jp[k] = gE[8 * i + 4 + k]; }
#endif
    st.s = 0; st.h = 0; st.nacc = 0; st.nrej = 0; st.npass = 0; st.status = RUNNING;
    trace_ray(st, pc);
    gX[i] = st.x;
    gV[i] = st.v;
#if NJ > 0
    [ForceUnroll] for (int k = 0; k < NJ; k++) { gE[8 * i + k] = st.E.J[k]; gE[8 * i + 4 + k] = st.E.Jp[k]; }
#endif
    gStats[i] = float4(st.s, float(st.nacc), float(st.nrej), float(st.npass));
    gStatus[i] = st.status;
}
