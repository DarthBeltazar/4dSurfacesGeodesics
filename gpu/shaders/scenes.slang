// Фигуры и червоточины (DD-002) — порт prototype/geo4d/shapes.py. Сцена выбирается при компиляции:
//   slangc -DSCENE=<id>. Параметры — значения по умолчанию прототипа.
// Каждая сцена задаёт: sdf<T>(x, y, z, w), SCALE (ℓ_s), hcap(x) (DD-001), ALPHA = 0.1/ℓ_s (ADR-0001).
#include "jet.slang"

#define SCENE_HYPERPLANE 0
#define SCENE_S3 1
#define SCENE_S2XR 2
#define SCENE_S1XR2 3
#define SCENE_ELLIPSOID 4
#define SCENE_SPHERITORUS 5
#define SCENE_TORISPHERE 6
#define SCENE_TIGER 7
#define SCENE_ELLIS 8
#define SCENE_COMPACT 9
#define SCENE_TWO_COMPACT 10
#define SCENE_DYNAMIC 11          // окно, режим редактирования (DD-009): вид и параметры многообразия — из буфера камеры

#ifndef SCENE
#error "задайте -DSCENE=<id>"
#endif

// Параметры многообразия (DD-008): дефайны, по умолчанию — значения фигур geo4d.shapes по умолчанию
// (geo4d.gpu_pt.manifold_defines). SCALE = Shape.scale той же фигуры (α, ADR-0001).
#ifndef MF_R
#define MF_R 1.0
#endif
#ifndef MF_RT
#if SCENE == SCENE_TIGER
#define MF_RT 0.3
#else
#define MF_RT 0.4
#endif
#endif
#ifndef MF_R1
#define MF_R1 1.2
#endif
#ifndef MF_R2
#define MF_R2 0.8
#endif
#ifndef MF_B
#define MF_B 1.0
#endif
#ifndef MF_RHO0
#define MF_RHO0 3.0
#endif
#ifndef MF_W
#define MF_W 1.5
#endif
#ifndef MF_C1X
#define MF_C1X -4.0
#define MF_C1Y 0.0
#define MF_C1Z 0.0
#define MF_C2X 4.0
#define MF_C2Y 0.0
#define MF_C2Z 0.0
#endif

// M11: ψ(u) = 1 − exp(1 − 1/(1 − u)) при q = 1 − u > 1e-2, иначе 1
__generic<T : IReal> T compact_h(T rho, float b, float rho0, float W)
{
    T q = 1.0 - (rho - b) * (1.0 / (rho0 - b));
    bool inside = q.value() > 1e-2;
    T qs = jselect(inside, q, T.cst(1.0));
    T psi = jselect(inside, 1.0 - jexp(1.0 - 1.0 / qs), T.cst(1.0));
    return (W * W) * psi;
}

#if SCENE == SCENE_DYNAMIC
// Строки многообразия в буфере камеры после строк объектов (geo4d.gpu_pt.manifold_rows, зеркало — editor.cpp):
//   MF_BASE + 0: (вид = SCENE_*, выход EXIT, SCALE, есть фон)   + 1: параметры (a, b, c) — см. sdf ниже
//   + 2, + 3: центры c₁, c₂ (шары кривизны плоских зон)          + 4: (есть плоские зоны, число шаров, ρ₀, масштаб фона)
//   + 5: (s_max, ℓ_max, 0, невязка f камеры)                     + 6: нормаль M в спроецированной точке камеры (pt_snap)
// Вид одинаков для всех потоков диспетчеризации — ветвление по нему не выбор ветви внутри SDF (M6), а выбор сцены.
#ifndef GCAM_DECLARED
#define GCAM_DECLARED
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
#endif
static const int MF_BASE = 209;           // OBJ_BASE + 9 + 6·PT_MAXOBJ (geo4d.gpu_pt.MF_BASE)
int mf_kind() { return int(gCam[MF_BASE].x); }
int mf_exit() { return int(gCam[MF_BASE].y); }
#define SCALE (gCam[MF_BASE].z)
float4 mf_p() { return gCam[MF_BASE + 1]; }
float3 mf_c1() { return gCam[MF_BASE + 2].xyz; }
float3 mf_c2() { return gCam[MF_BASE + 3].xyz; }
float mf_env_scale() { return gCam[MF_BASE + 4].w; }
__generic<T : IReal> T sdf(T x, T y, T z, T w)
{
    int k = mf_kind();
    float4 p = mf_p();
    if (k == SCENE_S3) return jlength4(x, y, z, w) - p.x;                                         // R
    if (k == SCENE_SPHERITORUS) return jlength2(jlength3(x, y, z) - p.x, w) - p.y;                // R, r
    if (k == SCENE_TORISPHERE) return jlength3(jlength2(x, y) - p.x, z, w) - p.y;                 // R, r
    if (k == SCENE_TIGER) return jlength2(jlength2(x, y) - p.x, jlength2(z, w) - p.y) - p.z;      // R1, R2, r
    if (k == SCENE_ELLIS) return jlength3(x, y, z) - p.x * jcosh(w * (1.0 / p.x));                // b
    if (k == SCENE_COMPACT) return w * w - compact_h(jlength3(x, y, z), p.x, p.y, p.z);           // b, ρ₀, W
    if (k == SCENE_TWO_COMPACT)                                                                   // b, ρ₀, W; c₁, c₂
    {
        float3 c1 = mf_c1(), c2 = mf_c2();
        T r1 = jlength3(x - c1.x, y - c1.y, z - c1.z), r2 = jlength3(x - c2.x, y - c2.y, z - c2.z);
        return w * w - (compact_h(r1, p.x, p.y, p.z) + compact_h(r2, p.x, p.y, p.z) - p.z * p.z);
    }
    return w;                                                                                     // гиперплоскость
}
float hcap(float4 q)
{
    int k = mf_kind();
    float4 p = mf_p();
    if (k == SCENE_ELLIS) return 0.5 * length(q.xyz);
    if (k == SCENE_COMPACT) return max(length(q.xyz) - p.y, 0.25 * p.x);
    if (k == SCENE_TWO_COMPACT) return max(min(length(q.xyz - mf_c1()), length(q.xyz - mf_c2())) - p.y, 0.25 * p.x);
    return SCALE;
}

#elif SCENE == SCENE_HYPERPLANE
static const float SCALE = 1.0;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return w; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_S3
static const float SCALE = MF_R;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength4(x, y, z, w) - MF_R; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_S2XR
static const float SCALE = 1.0;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength3(x, y, z) - 1.0; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_S1XR2
static const float SCALE = 1.0;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength2(x, y) - 1.0; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_ELLIPSOID
static const float SCALE = 1.0;
__generic<T : IReal> T sdf(T x, T y, T z, T w)
{ return jlength4(x * (1.0 / 0.8), y * (1.0 / 1.0), z * (1.0 / 1.3), w * (1.0 / 1.6)) - 1.0; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_SPHERITORUS
static const float SCALE = MF_RT;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength2(jlength3(x, y, z) - MF_R, w) - MF_RT; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_TORISPHERE
static const float SCALE = MF_RT;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength3(jlength2(x, y) - MF_R, z, w) - MF_RT; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_TIGER
static const float SCALE = MF_RT;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength2(jlength2(x, y) - MF_R1, jlength2(z, w) - MF_R2) - MF_RT; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_ELLIS
static const float SCALE = MF_B;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength3(x, y, z) - MF_B * jcosh(w * (1.0 / MF_B)); }
float hcap(float4 p) { return 0.5 * length(p.xyz); }

#elif SCENE == SCENE_COMPACT
static const float SCALE = MF_B;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return w * w - compact_h(jlength3(x, y, z), MF_B, MF_RHO0, MF_W); }
float hcap(float4 p) { return max(length(p.xyz) - MF_RHO0, 0.25 * MF_B); }

#elif SCENE == SCENE_TWO_COMPACT
static const float SCALE = MF_B;
static const float3 MF_C1 = float3(MF_C1X, MF_C1Y, MF_C1Z), MF_C2 = float3(MF_C2X, MF_C2Y, MF_C2Z);
__generic<T : IReal> T sdf(T x, T y, T z, T w)
{
    T r1 = jlength3(x - MF_C1.x, y - MF_C1.y, z - MF_C1.z), r2 = jlength3(x - MF_C2.x, y - MF_C2.y, z - MF_C2.z);
    return w * w - (compact_h(r1, MF_B, MF_RHO0, MF_W) + compact_h(r2, MF_B, MF_RHO0, MF_W) - MF_W * MF_W);
}
float hcap(float4 p) { return max(min(length(p.xyz - MF_C1), length(p.xyz - MF_C2)) - MF_RHO0, 0.25 * MF_B); }
#endif

#if SCENE == SCENE_DYNAMIC
#define ALPHA (0.1 / SCALE)
#else
static const float ALPHA = 0.1 / SCALE;
#endif
