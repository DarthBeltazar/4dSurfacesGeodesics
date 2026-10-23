// Проверка раскладки кубмапы (M19) и SampleGrad: выборка в направлениях с заданными градиентами.
// Буферы: 0 направления (xyz, w = лист: 0 — верхний, 1 — нижний), 1 градиенты ddx, 2 ddy, 3 цвет, 4 кубмапа.

struct PC { uint n; uint mode; };          // mode 0 — SampleLevel(0), 1 — SampleGrad
[[vk::push_constant]] ConstantBuffer<PC> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gDir;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gDdx;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gDdy;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gOut;
[[vk::binding(4, 0)]] SamplerCubeArray<float4> gCube;

[shader("compute")] [numthreads(64, 1, 1)]
void cube_probe(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pc.n) return;
    float4 d = gDir[i];
    float4 loc = float4(d.xyz, d.w);
    gOut[i] = pc.mode == 0 ? gCube.SampleLevel(loc, 0) : gCube.SampleGrad(loc, gDdx[i].xyz, gDdy[i].xyz);
}
