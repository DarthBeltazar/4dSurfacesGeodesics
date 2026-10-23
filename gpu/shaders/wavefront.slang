// Wavefront режима A (DD-004 п. 5): init → [args → integrate(K шагов) с компакцией]×итераций → finish.
// Состояние лучей — SoA-буферы; очереди индексов — два буфера (чётность итерации), счётчики на GPU,
// размер диспетчеризации интегрирования — косвенный (vkCmdDispatchIndirect).
// Компиляция: -DSCENE -DEXIT -DNJ=4 -DKERNEL=10 (init) | 11 (args) | 12 (integrate) | 13 (finish) |
//             14 (очередь фолбэка из флагов детектора).
#include "modea.slang"

// Буферы (одинаковая раскладка для всех ядер wavefront)
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> wCam;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> wX;       // x
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> wV;       // v
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> wE;       // 8 float4 на луч
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> wSH;      // s, h, nacc, nrej
[[vk::binding(5, 0)]] RWStructuredBuffer<uint2> wSt;       // проходов, статус
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> wQ0;        // очередь (чётная итерация читает Q0)
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> wQ1;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> wCnt;       // [0], [1] — длины очередей, [2] — итераций с работой
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> wArgs;      // косвенные аргументы (gx, gy, gz)
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> wDir;    // выход: d∞, лист
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> wDd;     // выход: 4 float4 на пиксель
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> wStatus;   // выход: статус
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> wFlag;     // флаги детектора
[[vk::binding(14, 0)]] RWStructuredBuffer<uint4> wFbHdr;   // [0] = (число пикселей фолбэка, spp, seed, 0)
[[vk::binding(15, 0)]] RWStructuredBuffer<uint> wFbList;   // пиксели фолбэка

struct WPC { Params p; uint iter; uint K; uint nspp; uint seed; };
[[vk::push_constant]] ConstantBuffer<WPC> wpc;

RayState load_state(uint i)
{
    RayState st;
    st.x = wX[i]; st.v = wV[i];
    [ForceUnroll] for (int k = 0; k < 4; k++) { st.E.J[k] = wE[8 * i + k]; st.E.Jp[k] = wE[8 * i + 4 + k]; }
    float4 sh = wSH[i];
    st.s = sh.x; st.h = sh.y; st.nacc = uint(sh.z); st.nrej = uint(sh.w);
    uint2 ps = wSt[i];
    st.npass = ps.x; st.status = ps.y;
    return st;
}

void store_state(uint i, RayState st)
{
    wX[i] = st.x; wV[i] = st.v;
    [ForceUnroll] for (int k = 0; k < 4; k++) { wE[8 * i + k] = st.E.J[k]; wE[8 * i + 4 + k] = st.E.Jp[k]; }
    wSH[i] = float4(st.s, st.h, float(st.nacc), float(st.nrej));
    wSt[i] = uint2(st.npass, st.status);
}

void push_queue(uint parity, uint ray)
{
    uint slot;
    InterlockedAdd(wCnt[parity], 1, slot);
    if (parity == 0) wQ0[slot] = ray; else wQ1[slot] = ray;
}

#if KERNEL == 10
// Первичные лучи: M16 + пролёт M25 → состояние; активные — в очередь 0
[shader("compute")] [numthreads(64, 1, 1)]
void wf_init(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(wCam);
    uint W = uint(c.width), H = uint(c.height);
    uint i = id.x;
    if (i >= W * H) return;
    float2 uv = pixel_uv(c, i % W, i / W);
    float4 D = c.f + uv.x * c.r + uv.y * c.u;
    float nD = length(D);
    float4 d0 = D / nD;
    float Pl = c.focus / dot(d0, c.f);
    float dpix = 2 * c.tf / c.width;
    float4 pr = c.r - dot(d0, c.r) * d0, pu = c.u - dot(d0, c.u) * d0;
    RayState st;
    st.x = c.x; st.v = d0;
    st.E.J[0] = c.r; st.E.J[1] = c.u; st.E.J[2] = float4(0); st.E.J[3] = float4(0);
    st.E.Jp[0] = -pr / Pl; st.E.Jp[1] = -pu / Pl; st.E.Jp[2] = dpix * pr / nD; st.E.Jp[3] = dpix * pu / nD;
    st.s = 0; st.h = 0; st.nacc = 0; st.nrej = 0; st.npass = 0; st.status = RUNNING;
    if (c.flight_r > 0)
    {
        float3 q0 = c.x.xyz, dd = d0.xyz;
        float bq = dot(dd, q0), disc = bq * bq - (dot(q0, q0) - c.flight_r * c.flight_r);
        float t = -bq - sqrt(max(disc, 0.0));
        if (disc > 0 && t > 0)
        {
            st.x = c.x + t * d0;
            [ForceUnroll] for (int k = 0; k < 4; k++) st.E.J[k] += t * st.E.Jp[k];
        }
        else
            st.status = DONE_ESCAPE;
    }
    init_step(st, wpc.p);
    store_state(i, st);
    if (st.status == RUNNING) push_queue(0, i);
}
#endif

#if KERNEL == 11
// Косвенные аргументы для итерации iter: из длины входной очереди; выходная очередь обнуляется
[shader("compute")] [numthreads(1, 1, 1)]
void wf_args(uint3 id : SV_DispatchThreadID)
{
    uint pin = wpc.iter & 1, pout = 1 - pin;
    uint n = wCnt[pin];
    wArgs[0] = (n + 63) / 64; wArgs[1] = 1; wArgs[2] = 1;
    wCnt[pout] = 0;
    if (n > 0) wCnt[2] = wpc.iter + 1;
    if (wpc.iter == 0) wFbHdr[0] = uint4(0);
}
#endif

#if KERNEL == 12
// K попыток шага для лучей входной очереди; незавершённые — в выходную очередь (компакция)
[shader("compute")] [numthreads(64, 1, 1)]
void wf_integrate(uint3 id : SV_DispatchThreadID)
{
    uint pin = wpc.iter & 1;
    uint e = id.x;
    if (e >= wCnt[pin]) return;
    uint i = pin == 0 ? wQ0[e] : wQ1[e];
    RayState st = load_state(i);
    trace_k(st, wpc.p, wpc.K);
    store_state(i, st);
    if (st.status == RUNNING) push_queue(1 - pin, i);
}
#endif

#if KERNEL == 13
// Выход: ∂v (M17), d∞ и ∂d∞ дуальными числами (M24) — для всех пикселей
[shader("compute")] [numthreads(64, 1, 1)]
void wf_finish(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(wCam);
    uint i = id.x;
    if (i >= uint(c.width) * uint(c.height)) return;
    RayState st = load_state(i);
    Jet15 s15[4];
    [ForceUnroll] for (int k = 0; k < 4; k++) { float4 e = float4(0); e[k] = 1; s15[k] = Jet15(st.x[k], e, float4x4(0)); }
    Jet15 Jx = sdf(s15[0], s15[1], s15[2], s15[3]);
    float gn = length(Jx.g);
    float4 n = Jx.g / gn;
    float4 Pv = st.v - dot(st.v, n) * n;
    float4 dv[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        float4 PJ = st.E.J[k] - dot(st.E.J[k], n) * n;
        dv[k] = st.E.Jp[k] - (dot(PJ, mul(Jx.H, Pv)) / gn) * n;
    }
    Dual4 X[4], V[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        X[k] = Dual4(st.x[k], float4(st.E.J[0][k], st.E.J[1][k], st.E.J[2][k], st.E.J[3][k]));
        V[k] = Dual4(st.v[k], float4(dv[0][k], dv[1][k], dv[2][k], dv[3][k]));
    }
    Dual4 d[3];
    exit_dir(X, V, d);
    wDir[i] = float4(d[0].a, d[1].a, d[2].a, st.x.w >= 0 ? 1.0 : -1.0);
    [ForceUnroll] for (int k = 0; k < 4; k++) wDd[4 * i + k] = float4(d[0].b[k], d[1].b[k], d[2].b[k], 0);
    wStatus[i] = st.status == RUNNING ? DONE_NMAX : st.status;
}
#endif

#if KERNEL == 14
// Компакция флагов детектора в очередь фолбэка; косвенные аргументы фолбэка — wArgs[3..5]
// (перед ядром wFbHdr[0].x должен быть 0: обнуляет wf_args на итерации 0)
[shader("compute")] [numthreads(64, 1, 1)]
void wf_fbqueue(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(wCam);
    uint i = id.x;
    uint N = uint(c.width) * uint(c.height);
    if (i < N && wFlag[i] != 0)
    {
        uint slot;
        InterlockedAdd(wFbHdr[0].x, 1, slot);
        wFbList[slot] = i;
    }
    if (i == 0) { wFbHdr[0].y = wpc.nspp; wFbHdr[0].z = wpc.seed; }
}

// Аргументы косвенного фолбэка (после wf_fbqueue): [3..5] — порции (FB_CHUNK = 8), [6..8] — редукция
[shader("compute")] [numthreads(1, 1, 1)]
void wf_fbargs(uint3 id : SV_DispatchThreadID)
{
    uint n = wFbHdr[0].x, nchunk = (wpc.nspp + 7) / 8;
    wArgs[3] = (n * nchunk + 63) / 64; wArgs[4] = 1; wArgs[5] = 1;
    wArgs[6] = (n + 63) / 64; wArgs[7] = 1; wArgs[8] = 1;
}
#endif
