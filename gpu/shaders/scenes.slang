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
static const float SCALE = 1.0;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength4(x, y, z, w) - 1.0; }
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
static const float SCALE = 0.4;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength2(jlength3(x, y, z) - 1.0, w) - 0.4; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_TORISPHERE
static const float SCALE = 0.4;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength3(jlength2(x, y) - 1.0, z, w) - 0.4; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_TIGER
static const float SCALE = 0.3;
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength2(jlength2(x, y) - 1.2, jlength2(z, w) - 0.8) - 0.3; }
float hcap(float4 p) { return SCALE; }

#elif SCENE == SCENE_ELLIS
static const float SCALE = 1.0;     // b
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return jlength3(x, y, z) - 1.0 * jcosh(w * (1.0 / 1.0)); }
float hcap(float4 p) { return 0.5 * length(p.xyz); }

#elif SCENE == SCENE_COMPACT
static const float SCALE = 1.0;     // b = 1, ρ₀ = 3, W = 1.5
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return w * w - compact_h(jlength3(x, y, z), 1.0, 3.0, 1.5); }
float hcap(float4 p) { return max(length(p.xyz) - 3.0, 0.25); }

#elif SCENE == SCENE_TWO_COMPACT
static const float SCALE = 1.0;     // центры (±4, 0, 0)
__generic<T : IReal> T sdf(T x, T y, T z, T w)
{
    T r1 = jlength3(x + 4.0, y, z), r2 = jlength3(x - 4.0, y, z);
    return w * w - (compact_h(r1, 1.0, 3.0, 1.5) + compact_h(r2, 1.0, 3.0, 1.5) - 1.5 * 1.5);
}
float hcap(float4 p) { return max(min(length(p.xyz - float3(-4, 0, 0)), length(p.xyz - float3(4, 0, 0))) - 3.0, 0.25); }
#endif

static const float ALPHA = 0.1 / SCALE;
