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

#if SCENE == SCENE_HYPERPLANE
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

static const float ALPHA = 0.1 / SCALE;
