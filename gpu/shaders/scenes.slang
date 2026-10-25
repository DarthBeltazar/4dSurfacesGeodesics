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
#define SCENE_GRAPH 12            // многообразие из графа сцены (DD-011): SCALE, sdf и hcap — в сгенерированном mf_scene.slang

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

// Интерпретатор байткода (DD-011 §6; программы строит gpu/src/bytecode.cpp, кодировка — geo4d.bytecode): программа
// с строки base — (n, регистров, 0, 0), затем n инструкций (op + 256·dst + 65536·a, b + 256·c, k, 0). Программа одна
// на диспетчеризацию — ветвление по op однородно. Над jet-числами — четыре пути: унарная функция (M4: значение и две
// производные считаются скалярно по op, затем одна операция unary), a ± b, a·b и выбор по значению (M6).
// Область программ (codegen.hpp kProgBase, bytecode.hpp): PROG_BASE — (число шаров кривизны, 0, 0, 0); +1…+16 — их
// центры; +17…+48 — строка начала программы объекта k (вид 3, составная форма); +49 — программа многообразия (MF_PROG).
static const int PROG_BASE = MF_BASE + 7 + 256;   // после строк многообразия и слотов (DD-011 §5)
static const int PROG_DIR = PROG_BASE + 17;
static const int PROG_MF = PROG_BASE + 49;
#ifndef PROG_NREG
#define PROG_NREG 32                       // bytecode.hpp kProgMaxReg
#endif
__generic<T : IReal> T prog_eval(int base, T x, T y, T z, T w)
{
    T r[PROG_NREG];
    r[0] = x; r[1] = y; r[2] = z; r[3] = w;
#ifdef PROG_CONST_ROWS                    // вариант пробы: программа многообразия константным массивом (не из буфера)
    static const float4 kProg[] = { PROG_CONST_ROWS };
    int n = int(kProg[0].x);
#else
    int n = int(gCam[base].x);
#endif
    for (int i = 0; i < n; i++)
    {
#ifdef PROG_CONST_ROWS
        float4 ins = kProg[1 + i];
#else
        float4 ins = gCam[base + 1 + i];
#endif
        uint e0 = uint(ins.x), e1 = uint(ins.y);
        uint op = e0 & 255u, d = (e0 >> 8) & 255u, ia = e0 >> 16, ib = e1 & 255u, ic = e1 >> 8;
        float k = ins.z;
#ifdef PROG_STATIC_REGS                   // вариант пробы: регистры без динамического индекса (выбор развёрнутым циклом)
        T a = r[0], b = r[0], c = r[0];
        [ForceUnroll] for (int j = 1; j < PROG_NREG; j++)
        {
            if (ia == j) a = r[j];
            if (ib == j) b = r[j];
            if (ic == j) c = r[j];
        }
#else
        T a = r[ia], b = r[ib], c = r[ic];
#endif
        float u = a.value();
        float g0 = 0, g1 = 0, g2 = 0;
        int cls = 0;                          // 0 — унарная, 1 — a + g0·b, 2 — a·b, 3 — выбор
        switch (op)
        {
        case 1: g0 = k; break;                                                        // константа
        case 2: cls = 1; g0 = 1; break;
        case 3: cls = 1; g0 = -1; break;
        case 4: cls = 2; break;
        case 6: g0 = -u; g1 = -1; break;
        case 7: g0 = u + k; g1 = 1; break;
        case 8: g0 = u * k; g1 = k; break;
        case 9: g0 = 1.0 / u; g1 = -g0 * g0; g2 = 2.0 * g0 * g0 * g0; break;
        case 10: { float s = sqrt(u); g0 = s; g1 = 0.5 / s; g2 = -0.25 / (u * s); break; }
        case 11: g0 = exp(u); g1 = g0; g2 = g0; break;
        case 12: g0 = log(u); g1 = 1.0 / u; g2 = -g1 * g1; break;
        case 13: g0 = cosh(u); g1 = sinh(u); g2 = g0; break;
        case 14: g0 = sin(u); g1 = cos(u); g2 = -g0; break;
        case 15: g0 = cos(u); g1 = -sin(u); g2 = -g0; break;
        case 16:                                                                      // защищённый корень (M6)
            if (u < LENGTH_EPS2) g0 = sqrt(LENGTH_EPS2);
            else { float s = sqrt(u); g0 = s; g1 = 0.5 / s; g2 = -0.25 / (u * s); }
            break;
        case 17: cls = 3; break;
        case 19: g0 = u * u; g1 = 2.0 * u; g2 = 2.0; break;
        default: g0 = sinh(u); g1 = cosh(u); g2 = g0; break;
        }
        T v;
        if (cls == 0) v = a.unary(g0, g1, g2);
        else if (cls == 1) v = a.add(b.scale(g0));
        else if (cls == 2) v = a.mul(b);
        else v = c.value() > k ? a : b;
#ifdef PROG_STATIC_REGS
        [ForceUnroll] for (int j = 0; j < PROG_NREG; j++)
            if (d == j) r[j] = v;
#else
        r[d] = v;
#endif
    }
    return r[0];
}
// центр шара кривизны i (M29): у многообразия-программы — из области программ (любое число), иначе — c₁, c₂
float3 mf_ball(int i)
{
#ifdef MF_PROG
    return gCam[PROG_BASE + 1 + i].xyz;
#else
    return i == 0 ? mf_c1() : mf_c2();
#endif
}

#ifdef MF_PROG
__generic<T : IReal> T sdf(T x, T y, T z, T w) { return prog_eval(PROG_MF, x, y, z, w); }
#else
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
#endif
float hcap(float4 q)
{
    int k = mf_kind();
    float4 p = mf_p();
    if (k == SCENE_ELLIS) return 0.5 * length(q.xyz);
    if (k == SCENE_COMPACT) return max(length(q.xyz) - p.y, 0.25 * p.x);
    if (k == SCENE_TWO_COMPACT) return max(min(length(q.xyz - mf_c1()), length(q.xyz - mf_c2())) - p.y, 0.25 * p.x);
#ifdef MF_PROG
    int nb = int(gCam[PROG_BASE].x);                    // многообразие графа (листы с горловинами, M35)
    if (k == SCENE_GRAPH && nb > 0)
    {
        float d = 1e30;
        for (int i = 0; i < nb; i++) d = min(d, length(q.xyz - mf_ball(i)));
        return max(d - gCam[MF_BASE + 4].z, 0.25 * SCALE);
    }
#endif
    return SCALE;
}

#elif SCENE == SCENE_GRAPH
#include "mf_scene.slang"

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

#if SCENE == SCENE_DYNAMIC || defined(MF_SLOTS)   // SCALE из буфера: DD-009 или слоты графа (DD-011 §5)
#define ALPHA (0.1 / SCALE)
#else
static const float ALPHA = 0.1 / SCALE;
#endif
