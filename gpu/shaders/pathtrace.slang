// Режим B (DD-005): мегаядро трассировщика путей — порт prototype/geo4d/pathtrace.py во float32.
// Поток на пиксель, один путь за диспетчеризацию (накопление в буферах; номер сэмпла — счётчик пикселя).
// Сцена: -DSCENE=<id> (многообразие, scenes.slang), -DEXIT=1|2 (modea.slang), объекты/материалы/источники/плоские
// зоны — сгенерированный pt_scene.slang (geo4d.gpu_pt.scene_header, каталог через -I).
// Буферы: 0 камера (Cam), 1 Σ цвет и число сэмплов, 2 Σ квадратов, 3 таблица выборки фона (M29), 4 кубмапа-массив,
//         5 статистика пикселя (оборванные пути: потолок или попадание с обратной стороны, проходов Jet7, шагов прямой,
//           сегментов).
#define PT_OBJECTS 1
#if PT_STAGE && SCENE == 11
#define MF_ROW(i) gsObj[OBJ_NROWS + (i)]    // SCENE_DYNAMIC: строки многообразия — в groupshared вслед за объектами
#endif
#include "modea.slang"
#include "pt_scene.slang"

struct PTParams
{
    uint n;             // пикселей
    uint seed;
    float tol_kappa;    // M30: tol = min(tol_max, κ·max(θ_pix, γ))
    float tol_max;
    float c_hit;        // M26: ε_hit = c_hit·w
    float k_cone;       // M31: 0 — фон без мипов
    uint max_depth;
    uint rr_depth;
    uint nee;
    uint mis;           // 0 — степенная эвристика, 1 — «только NEE»
    float s_max;
    uint n_max;
    float ell_max;      // уход для сцен без плоских зон (Эллис)
    uint env_n;         // текселей на грань в таблице фона
    float env_scale;
    float k_lod;        // M31: множитель раствора отскоков для мипа фона (0 — мип 0)
    uint use_list;      // 1 — потоки по списку gPix (n — длина), 2 — длина из gCnt[list_id], список с gPix[list_off]
    float c_mix;        // проход камеры: «смешанный», если g < c_mix·след (след — 3σ полей пикселя и апертуры)
    uint list_id;
    uint list_off;
    uint use_den;       // приложение (DD-007): 1 — вывод денойзера в pt_present
    uint bgra;          // 1 — упаковка BGRA8 (формат swapchain)
    float exposure;
};

#ifndef CAM_STEP
[[vk::push_constant]] ConstantBuffer<PTParams> pp;
#else
static PTParams pp;                 // cam_step не использует параметры трассировщика
#endif
// длина пути, уход (Эллис) и масштаб фона — свойства многообразия: в динамическом варианте (DD-009) — из его строк
#if SCENE == SCENE_DYNAMIC
#define PP_S_MAX (MF_ROW(5).x)
#define PP_ELL_MAX (MF_ROW(5).y)
#define PP_ENV_SCALE (MF_ROW(4).w)
#else
#define PP_S_MAX pp.s_max
#define PP_ELL_MAX pp.ell_max
#define PP_ENV_SCALE pp.env_scale
#endif
#ifndef GCAM_DECLARED
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
#endif
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gAcc;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gAcc2;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gEnv;
[[vk::binding(4, 0)]] SamplerCubeArray<float4> gCube;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> gStats;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> gPix;       // список пикселей (use_list)
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> gAlb;     // AOV: Σ альбедо первого попадания
[[vk::binding(8, 0)]] RWStructuredBuffer<float4> gNrm;     // AOV: Σ нормали в базисе камеры (M32)
// проход камеры (NJ = 4): выход режима A и класс пикселя
[[vk::binding(9, 0)]] RWStructuredBuffer<float4> gDir;
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> gDd;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> gStatus;
[[vk::binding(12, 0)]] RWStructuredBuffer<uint> gClass;    // 0 — фон (режим A), 1 — объект, 2 — смешанный
// кадр одним командным буфером: флаги детектора, счётчики списков, косвенные аргументы
[[vk::binding(13, 0)]] RWStructuredBuffer<uint> gFlag;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> gCnt;      // [0] — пиксели объектов/смешанные, [1] — фолбэк фона
[[vk::binding(15, 0)]] RWStructuredBuffer<uint> gArgs;     // [0..2], [3..5] — группы для списков 0 и 1
// приложение (DD-007): кадр и AOV для денойзера (экспортируемая память), выход денойзера, RGBA8, цвет режима A
[[vk::binding(16, 0)]] RWStructuredBuffer<float4> gFrame;
[[vk::binding(17, 0)]] RWStructuredBuffer<float4> gAlbF;
[[vk::binding(18, 0)]] RWStructuredBuffer<float4> gNrmF;
[[vk::binding(19, 0)]] RWStructuredBuffer<float4> gDen;
[[vk::binding(20, 0)]] RWStructuredBuffer<uint> gPresent;
[[vk::binding(21, 0)]] RWStructuredBuffer<float4> gColor;

// объекты (DD-008): строки float4 в хвосте буфера камеры (geo4d.gpu_pt.object_rows) — правятся без перекомпиляции.
// Два варианта (pt_scene.slang): «запечённый» (PT_DYNAMIC 0) — число объектов, их виды и оси полупространств заданы
// при компиляции (циклы разворачиваются, ветви по виду сворачиваются), из строк читаются только параметры;
// «динамический» (PT_DYNAMIC 1, режим редактирования окна) — число и виды тоже из строк (медленнее, любая правка сразу).
static const int OBJ_BASE = 8;
#if PT_DYNAMIC
static const int OBJ_NROWS = 9 + 6 * PT_MAXOBJ;
#else
static const int OBJ_NROWS = 9 + 6 * PT_NOBJ;
#endif
#if SCENE == SCENE_DYNAMIC
static const int GS_ROWS = OBJ_NROWS + 7;   // + строки многообразия (MF_BASE = OBJ_BASE + OBJ_NROWS при PT_DYNAMIC)
#else
static const int GS_ROWS = OBJ_NROWS;
#endif
#if PT_STAGE
// мегаядро: строки объектов (и многообразия) копируются в groupshared в начале группы (pt_sample)
groupshared float4 gsObj[GS_ROWS];
#define OBJ_ROW(r) gsObj[(r) - OBJ_BASE]
#else
#define OBJ_ROW(r) gCam[r]
#endif
int pt_nobj() { return int(OBJ_ROW(OBJ_BASE).x); }
int pt_nlight() { return int(OBJ_ROW(OBJ_BASE).y); }
int light_obj(int i) { return int(OBJ_ROW(OBJ_BASE + 1 + i / 4)[i % 4]); }
float4 obj_c(int k) { return OBJ_ROW(OBJ_BASE + 9 + 6 * k); }          // центр
float4 obj_s(int k) { return OBJ_ROW(OBJ_BASE + 10 + 6 * k); }         // (вид, a, b, знак)
float4 obj_pl(int k) { return OBJ_ROW(OBJ_BASE + 11 + 6 * k); }        // (1, ось, смещение, знак) у полупространства
float4 mat_a(int k) { return OBJ_ROW(OBJ_BASE + 12 + 6 * k); }         // (альбедо, вид материала)
float4 mat_e(int k) { return OBJ_ROW(OBJ_BASE + 13 + 6 * k); }         // (излучение, шероховатость)
float obj_r(int k) { return OBJ_ROW(OBJ_BASE + 14 + 6 * k).x; }        // радиус ограничивающей 4D-сферы

#if PT_DYNAMIC
#define PT_N pt_nobj()
bool pt_isplane(int k) { return obj_pl(k).x != 0; }
// 4D SDF объекта k по виду из строк (geo4d.objects: ball4, torus_in_sheet, halfspace)
__generic<T : IReal> T obj_g(int k, T x, T y, T z, T w)
{
    float4 c = obj_c(k), s = obj_s(k);
    int kind = int(s.x);
    if (kind == 0) return jlength4(x - c.x, y - c.y, z - c.z, w - c.w) - s.y;
    if (kind == 1) return jlength3(jlength2(x - c.x, y - c.y) - s.y, z - c.z, w - c.w) - s.z;
    int ax = int(s.y);
    T p = w;
    if (ax == 0) p = x;
    else if (ax == 1) p = y;
    else if (ax == 2) p = z;
    return s.w * (p - s.z);
}
#else
#define PT_N PT_NOBJ
bool pt_isplane(int k) { return PT_ISPLANE[k] != 0; }     // obj_g — в pt_scene.slang
#endif

static const float PI = 3.14159265;

// ГСЧ (как в фолбэке modea.slang, KERNEL 4)
uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float urand(inout uint s) { return (pcg(s) >> 8) * (1.0 / 16777216.0); }
float2 gauss2(inout uint s)
{
    float u1 = max(urand(s), 1e-7), u2 = urand(s);
    float r = sqrt(-2 * log(u1));
    return r * float2(cos(6.2831853 * u2), sin(6.2831853 * u2));
}
static const float ENV_UNIFORM = 0.1;

// ---- объекты (M26) ------------------------------------------------------------------------------

static float g_c_hit, g_w0, g_gamma, g_sbase;     // конус текущего сегмента (на поток)

float obj_val(int k, float4 p) { return obj_g(k, Real(p.x), Real(p.y), Real(p.z), Real(p.w)).a; }

float4 obj_grad(int k, float4 p)
{
    Dual4 r = obj_g(k, Dual4(p.x, float4(1, 0, 0, 0)), Dual4(p.y, float4(0, 1, 0, 0)), Dual4(p.z, float4(0, 0, 1, 0)),
                    Dual4(p.w, float4(0, 0, 0, 1)));
    return r.b;
}

float gmin_k(float4 p, out int k)
{
    float g = 1e30;
    k = -1;
    for (int i = 0; i < PT_N; i++) { float v = obj_val(i, p); if (v < g) { g = v; k = i; } }
    return g;
}

float pt_gmin(float4 p) { int k; return gmin_k(p, k); }

// минимум по ограниченным объектам (полупространства на прямой — аналитически)
float gmin_bounded_k(float4 p, out int k)
{
    float g = 1e30;
    k = -1;
    for (int i = 0; i < PT_N; i++) if (!pt_isplane(i)) { float v = obj_val(i, p); if (v < g) { g = v; k = i; } }
    return g;
}

// ближайшее пересечение прямой x + t v (t ≥ 0) с полупространствами (иначе 1e30)
float plane_hit(float4 x, float4 v, out int kp)
{
    float tp = 1e30;
    kp = -1;
    for (int k = 0; k < PT_N; k++)
    {
        float4 pl = obj_pl(k);
        if (pl.x == 0) continue;
        int ax = int(pl.y);
        if (pl.w * v[ax] < 0) { float t = (pl.z - x[ax]) / v[ax]; if (t >= 0 && t < tp) { tp = t; kp = k; } }
    }
    return tp;
}

float ulp_floor(float4 x) { float m = max(max(abs(x.x), abs(x.y)), max(abs(x.z), abs(x.w))); return 32 * 1.2e-7 * max(m, 1.0); }

// ε_hit = max(c_hit·w(s), 32 ulp(|x|)), w(s) = w₀ + γ (s_base + s): допуск перескока шага (h ≤ g + ε_hit).
// Попадание — только смена знака g (или g ниже пола округления): близкий промах (0 < g_min < ε) — не попадание.
float pt_eps_hit(float s, float4 x) { return max(g_c_hit * (g_w0 + g_gamma * (g_sbase + s)), ulp_floor(x)); }
float pt_hit_floor(float4 x) { return ulp_floor(x); }

// ---- плоские зоны (M29) ---------------------------------------------------------------------------

// радиус шара кривизны с допуском float32 — один и тот же во всех проверках (иначе луч, отданный интегратором на
// границе ρ ≈ ρ₀(1 − 1e-6), считался задевающим шар и летел до s_max)
#if PT_FLAT_DYNAMIC
#define RHO_IN (PT_RHO0 * (1 - 1e-5))
#else
static const float RHO_IN = PT_RHO0 * (1 - 1e-5);
#endif

bool in_flat(float4 x)
{
    for (int i = 0; i < PT_NBALL; i++) if (length(x.xyz - pt_ball_c(i)) < RHO_IN) return false;
    return PT_HAS_FLAT != 0;
}

bool ray_clear_balls(float4 x, float4 d)
{
    for (int i = 0; i < PT_NBALL; i++)
    {
        float3 q = x.xyz - pt_ball_c(i);
        float t = max(-dot(q, d.xyz) / dot(d.xyz, d.xyz), 0.0);
        if (length(q + t * d.xyz) < RHO_IN) return false;
    }
    return true;
}

bool seg_clear_balls(float4 x, float4 y)
{
    float3 d = y.xyz - x.xyz;
    for (int i = 0; i < PT_NBALL; i++)
    {
        float3 q = x.xyz - pt_ball_c(i);
        float t = clamp(-dot(q, d) / max(dot(d, d), 1e-30), 0.0, 1.0);
        if (length(q + t * d) < RHO_IN) return false;
    }
    return true;
}

// наименьшее t ≥ 0 входа луча в шар кривизны (иначе 1e30)
float ball_entry(float4 x, float4 d)
{
    float tb = 1e30;
    for (int i = 0; i < PT_NBALL; i++)
    {
        float3 q = x.xyz - pt_ball_c(i);
        float b = dot(q, d.xyz), c = dot(q, q) - RHO_IN * RHO_IN;
        float disc = b * b - c;
        if (disc > 0) { float t = -b - sqrt(disc); if (t >= 0) tb = min(tb, t); }
    }
    return tb;
}

// луч x + t v (прямая) не может задеть ни один объект (мимо ограничивающих сфер, от полупространств);
// planes = false — только ограниченные объекты
bool ahead_clear(float4 x, float4 v, bool planes = true)
{
    for (int k = 0; k < PT_N; k++)
    {
        float4 pl = obj_pl(k);
        if (pl.x > 0) { if (planes && pl.w * v[int(pl.y)] < 0) return false; continue; }
        float4 oc = obj_c(k) - x;
        float tca = dot(oc, v), r = obj_r(k);
        if (dot(oc, oc) < r * r) return false;
        if (tca > 0 && dot(oc, oc) - tca * tca < r * r) return false;
    }
    return true;
}

// интегратор возвращает управление прямому пролёту, когда луч снова в плоской зоне и уходит от ближайшего шара
bool pt_escape(float4 x, float4 v)
{
    if (!in_flat(x)) return false;
    int ib = 0;
    float best = 1e30;
    for (int i = 0; i < PT_NBALL; i++) { float d = length(x.xyz - pt_ball_c(i)); if (d < best) { best = d; ib = i; } }
    return dot(x.xyz - pt_ball_c(ib), v.xyz) > 0;
}

// отрезок x + t d, t ∈ (0, tmax) задевает объект (кроме skip)? Полупространства — аналитически, остальное — sphere tracing
bool line_blocked(float4 x, float4 d, float tmax, int skip, float eps)
{
    float tfar = 0;
    for (int k = 0; k < PT_N; k++)
    {
        if (k == skip) continue;
        float4 pl = obj_pl(k);
        if (pl.x > 0)
        {
            int ax = int(pl.y);
            if (pl.w * d[ax] < 0) { float tp = (pl.z - x[ax]) / d[ax]; if (tp > 0 && tp < tmax) return true; }
        }
        else tfar = max(tfar, dot(obj_c(k) - x, d) + obj_r(k));
    }
    float t = 0, lim = min(tmax, tfar);
    for (int it = 0; it < 512 && t < lim; it++)
    {
        float4 p = x + t * d;
        float g = 1e30;
        for (int k = 0; k < PT_N; k++) if (k != skip && !pt_isplane(k)) g = min(g, obj_val(k, p));
        if (g < ulp_floor(p)) return true;          // пересечение (шаг g + ε/2 перескакивает поверхность)
        t += g + 0.5 * eps;
    }
    return false;
}

// ---- проекция, кадр (M3, M28) -------------------------------------------------------------------

float4 normal_M(float4 x) { Eval e = rhs7(x, float4(0), 0.0); return e.g / length(e.g); }

float4 proj_M(float4 x)
{
    [ForceUnroll] for (int it = 0; it < 2; it++) { Eval e = rhs7(x, float4(0), 0.0); x -= e.f * e.g / dot(e.g, e.g); }
    return x;
}

float det3(float3 a, float3 b, float3 c) { return dot(a, cross(b, c)); }

float4 cross4(float4 a, float4 b, float4 c)
{
    return float4(det3(a.yzw, b.yzw, c.yzw), -det3(a.xzw, b.xzw, c.xzw), det3(a.xyw, b.xyw, c.xyw), -det3(a.xyz, b.xyz, c.xyz));
}

void shading_frame(int k, float4 x, out float4 n, out float4 ng, out float4 t1, out float4 t2)
{
    n = normal_M(x);
    float4 gg = obj_grad(k, x);
    ng = normalize(gg - dot(gg, n) * n);
    float4 q = n * n + ng * ng;
    int ax = 0;
    [ForceUnroll] for (int i = 1; i < 4; i++) if (q[i] < q[ax]) ax = i;
    float4 e = float4(0);
    e[ax] = 1;
    t1 = normalize(e - dot(e, n) * n - dot(e, ng) * ng);
    t2 = normalize(cross4(n, ng, t1));
}

// ---- M27: эрмитово уточнение --------------------------------------------------------------------

float4 hermite(float4 x0, float4 v0, float4 x1, float4 v1, float h, float s, out float4 dx)
{
    float s2 = s * s, s3 = s2 * s;
    dx = ((6 * s2 - 6 * s) * (x0 - x1) + (3 * s2 - 4 * s + 1) * h * v0 + (3 * s2 - 2 * s) * h * v1) / h;
    return (2 * s3 - 3 * s2 + 1) * x0 + (s3 - 2 * s2 + s) * h * v0 + (-2 * s3 + 3 * s2) * x1 + (s3 - s2) * h * v1;
}

void refine_hit(int k, float4 x0, float4 v0, float4 x1, float4 v1, float h, out float4 x, out float4 v, out float sig)
{
    float fa = obj_val(k, x0), fb = obj_val(k, x1);
    sig = 1;
    float4 dx;
    if (fb < 0 && fa > 0)
    {
        float a = 0, b = 1;
        int side = 0;
        for (int it = 0; it < 12; it++)
        {
            float c = (a * fb - b * fa) / (fb - fa);
            float fc = obj_val(k, hermite(x0, v0, x1, v1, h, c, dx));
            if (fc * fb < 0) { a = b; fa = fb; if (side == 1) fa *= 0.5; side = 1; }
            else { if (side == -1) fa *= 0.5; side = -1; }
            b = c; fb = fc;
        }
        sig = abs(fa) < abs(fb) ? a : b;
    }
    x = hermite(x0, v0, x1, v1, h, sig, dx);
    v = h > 1e-3 * SCALE ? dx : lerp(v0, v1, sig);
    [ForceUnroll] for (int it = 0; it < 2; it++)
    {
        x = proj_M(x);
        float dg = dot(obj_grad(k, x), v);
        if (abs(dg) > 1e-6) x -= obj_val(k, x) / dg * v;
    }
    x = proj_M(x);
    float4 n = normal_M(x);
    v = normalize(v - dot(v, n) * n);
}

// ---- сегмент пути ----------------------------------------------------------------------------------

static const uint SEG_ESCAPE = 0, SEG_HIT = 1, SEG_CAP = 2;

// прямой пролёт в плоской зоне: 0 — уход, 1 — попадание, 2 — вход в шар кривизны, 3 — потолок
uint flat_flight(inout float4 x, float4 v, inout float s, out int k, inout uint nsteps)
{
    g_sbase = 0;
    k = -1;
    float step = 0;
    int kp;
    float tp = plane_hit(x, v, kp);                  // полупространства — точно, без sphere tracing
    for (int it = 0; it < 1024; it++)
    {
        nsteps++;
        float eps = pt_eps_hit(s, x);
        float g = gmin_bounded_k(x, k);
        if (g < pt_hit_floor(x))
        {
            // корень на прямой между x − step·v (g_k > 0) и x (g_k < 0): Иллинойс по t ∈ [−step, 0]
            float ta = -step, tb2 = 0, fa = obj_val(k, x - step * v), fb = g;
            if (fa > 0 && fb < 0)
            {
                int side = 0;
                for (int j = 0; j < 16; j++)
                {
                    float tc = (ta * fb - tb2 * fa) / (fb - fa), fc = obj_val(k, x + tc * v);
                    if (fc * fb < 0) { ta = tb2; fa = fb; if (side == 1) fa *= 0.5; side = 1; }
                    else { if (side == -1) fa *= 0.5; side = -1; }
                    tb2 = tc; fb = fc;
                }
                float tr = abs(fa) < abs(fb) ? ta : tb2;
                x += tr * v;
                s += tr;
            }
            return 1;
        }
        if (ray_clear_balls(x, v) && ahead_clear(x, v, false))
        {
            // впереди только полупространства: попадание в ближайшее — точно, иначе уход
            if (tp < 1e30) { x += tp * v; s += tp; k = kp; return 1; }
            return 0;
        }
        float tb = ball_entry(x, v);
        step = g + 0.5 * eps;
        if (tp <= min(step, tb)) { x += tp * v; s += tp; k = kp; return 1; }
        if (tb <= step) { x += tb * v; s += tb; return 2; }
        x += step * v;
        s += step;
        tp -= step;
        if (s > PP_S_MAX) return 3;
    }
    return 3;
}

// Сегмент от (x, v) до попадания/ухода. straight — весь сегмент прошёл по прямой в плоской зоне (M29).
// wr, wu — базис AOV (M32), переносится при transport (в плоской зоне постоянен).
uint trace_segment(inout float4 x, inout float4 v, out float s, out int k, float tol, out bool straight, inout float4 stats,
                   inout float4 wr, inout float4 wu, bool transport)
{
    s = 0;
    k = -1;
    straight = in_flat(x);
    Params p;
    p.n = 1; p.tol = tol; p.h0 = 0.05 * SCALE; p.s_max = PP_S_MAX; p.h_max = 0; p.h_min = 1e-7; p.n_max = pp.n_max;
    p.escape_mode = PT_HAS_FLAT != 0 ? 3 : 1; p.escape_param = PP_ELL_MAX; p.alpha = -1; p.xscale = 0; p.fixed = 0;
    uint nst = 0;
    for (int part = 0; part < 64; part++)
    {
        if (PT_HAS_FLAT != 0 && in_flat(x))
        {
            uint r = flat_flight(x, v, s, k, nst);
            stats.z += nst; nst = 0;
            if (r == 0) return SEG_ESCAPE;
            if (r == 1) return SEG_HIT;
            if (r == 3) return SEG_CAP;
            straight = false;                         // вход в шар кривизны
        }
        g_sbase = s;
        RayState st;
        st.x = x; st.v = v;
        st.s = 0; st.h = 0; st.nacc = 0; st.nrej = 0; st.npass = 0; st.status = RUNNING;
        st.mix = 1e30; st.transport = transport ? 1 : 0; st.wr = wr; st.wu = wu;
        p.s_max = PP_S_MAX - s;
        trace_ray(st, p);
        wr = st.wr; wu = st.wu;
        stats.y += st.npass;
        straight = false;
        if (st.status == DONE_HIT)
        {
            int kk;
            gmin_k(st.x, kk);
            k = kk;
            float h = max(st.s - st.sp, 1e-12), sig;
            refine_hit(k, st.xp, st.vp, st.x, st.v, h, x, v, sig);
            s += st.sp + sig * h;
            return SEG_HIT;
        }
        x = st.x; v = st.v; s += st.s;
        if (st.status != DONE_ESCAPE) return SEG_CAP;
        if (PT_HAS_FLAT == 0) return SEG_ESCAPE;
    }
    return SEG_CAP;
}

// ---- BSDF (локально: z — нормаль n_g) --------------------------------------------------------------

float ggx_D(float3 h, float a) { float c2 = h.z * h.z; float t = c2 * (a * a - 1) + 1; return a * a / (PI * t * t); }
float ggx_lambda(float3 w, float a) { float c2 = max(w.z * w.z, 1e-12); return 0.5 * (-1 + sqrt(1 + a * a * max(1 - c2, 0.0) / c2)); }

float3 bsdf_eval(int kind, float3 alb, float a, float3 wo, float3 wi, out float pdf)
{
    pdf = 0;
    if (wo.z <= 0 || wi.z <= 0) return float3(0);
    if (kind == 0) { pdf = wi.z / PI; return alb / PI * wi.z; }
    float3 h = normalize(wo + wi);
    float D = ggx_D(h, a), Lo = ggx_lambda(wo, a), Li = ggx_lambda(wi, a);
    float3 F = alb + (1 - alb) * pow(1 - saturate(dot(wi, h)), 5.0);
    pdf = D / (1 + Lo) / (4 * wo.z);
    return F * (D / (1 + Lo + Li) / (4 * wo.z));
}

float3 bsdf_sample(int kind, float3 alb, float a, float3 wo, float u1, float u2, out float3 weight, out float pdf)
{
    float r = sqrt(u1), ph = 2 * PI * u2;
    float3 wi;
    if (kind == 0) wi = float3(r * cos(ph), r * sin(ph), sqrt(max(1 - u1, 0.0)));
    else
    {
        float3 Vh = normalize(float3(a * wo.x, a * wo.y, wo.z));
        float lsq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lsq > 0 ? float3(-Vh.y, Vh.x, 0) / sqrt(lsq) : float3(1, 0, 0);
        float3 T2 = cross(Vh, T1);
        float t1 = r * cos(ph), t2 = r * sin(ph), s = 0.5 * (1 + Vh.z);
        t2 = (1 - s) * sqrt(max(1 - t1 * t1, 0.0)) + s * t2;
        float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(1 - t1 * t1 - t2 * t2, 0.0)) * Vh;
        float3 Ne = normalize(float3(a * Nh.x, a * Nh.y, max(Nh.z, 0.0)));
        wi = 2 * dot(wo, Ne) * Ne - wo;
    }
    float3 f = bsdf_eval(kind, alb, a, wo, wi, pdf);
    weight = pdf > 0 ? f / pdf : float3(0);
    return wi;
}

float mis_w(float pa, float pb, bool nee_strategy)
{
    if (pp.mis == 1) return nee_strategy ? 1.0 : 0.0;
    return pa * pa / max(pa * pa + pb * pb, 1e-30);
}

// ---- фон как источник (M29) -------------------------------------------------------------------------

float3 face_dir(uint face, float s, float t)
{
    float u = 2 * s - 1, v = 2 * t - 1;
    float3 d;
    if (face == 0) d = float3(1, -v, -u);
    else if (face == 1) d = float3(-1, -v, u);
    else if (face == 2) d = float3(u, 1, v);
    else if (face == 3) d = float3(u, -1, -v);
    else if (face == 4) d = float3(u, -v, 1);
    else d = float3(-u, -v, -1);
    return normalize(d);
}

uint env_count() { return 6 * pp.env_n * pp.env_n; }

float env_pdf(uint sheet, float3 d)
{
    uint face;
    float2 st = face_st(d, face);
    uint n = pp.env_n;
    uint i = min(uint(st.x * n), n - 1), j = min(uint(st.y * n), n - 1);
    float p = gEnv[2 * env_count() + sheet * env_count() + face * n * n + j * n + i];
    float u = 2 * st.x - 1, v = 2 * st.y - 1;
    return (1 - ENV_UNIFORM) * p * n * n * pow(1 + u * u + v * v, 1.5) / 4 + ENV_UNIFORM / (4 * PI);
}

float3 env_sample(uint sheet, inout uint seed)
{
    if (urand(seed) < ENV_UNIFORM)
    {
        float z = 2 * urand(seed) - 1, ph = 2 * PI * urand(seed), r = sqrt(max(1 - z * z, 0.0));
        return float3(r * cos(ph), r * sin(ph), z);
    }
    float u = urand(seed);
    uint lo = 0, hi = env_count() - 1, base = sheet * env_count();
    while (lo < hi) { uint mid = (lo + hi) / 2; if (gEnv[base + mid] < u) lo = mid + 1; else hi = mid; }
    uint n = pp.env_n, face = lo / (n * n), rem = lo % (n * n);
    return face_dir(face, (rem % n + urand(seed)) / n, (rem / n + urand(seed)) / n);
}

// M31: мип по раствору, набранному на отскоках (gb); пиксельный след первичного луча уже отфильтрован джиттером
float3 env_radiance(float3 d, float4 x, float gb)
{
    float lod = 0;
    gb *= pp.k_lod;
    if (gb > 0)
    {
        Cam c = load_cam(gCam);
        lod = max(log2(gb / (PI / (2 * c.cube_n))), 0.0);
    }
    return gCube.SampleLevel(float4(d, x.w < 0 ? 1 : 0), lod).rgb * PP_ENV_SCALE;
}

// ---- сферические источники ------------------------------------------------------------------------

bool is_light(int k) { for (int i = 0; i < pt_nlight(); i++) if (light_obj(i) == k) return true; return false; }

// 1 − cos θ_max конуса сферы без потери знаков: x/(1 + √(1 − x)), x = r²/d² (у далёких точек 1 − cos → 0 во float32)
float one_minus_cmax(float r, float d2) { float x = min(r * r / d2, 1.0); return x / (1 + sqrt(1 - x)); }

float light_pdf(float4 p, int k)
{
    float3 dc = obj_c(k).xyz - p.xyz;
    return 1.0 / (2 * PI * one_minus_cmax(obj_r(k), dot(dc, dc))) / float(pt_nlight());
}

// ---- ядро -----------------------------------------------------------------------------------------

#if NJ == 0
[shader("compute")] [numthreads(64, 1, 1)]
void pt_sample(uint3 id : SV_DispatchThreadID, uint3 tid : SV_GroupThreadID)
{
#if PT_STAGE
    for (int r = int(tid.x); r < GS_ROWS; r += 64) gsObj[r] = gCam[OBJ_BASE + r];
    GroupMemoryBarrierWithGroupSync();
#endif
    uint n = pp.use_list == 2 ? gCnt[pp.list_id] : pp.n;
    if (id.x >= n) return;
    uint i = pp.use_list == 2 ? gPix[pp.list_off + id.x] : (pp.use_list != 0 ? gPix[id.x] : id.x);
    Cam c = load_cam(gCam);
    float4 acc = gAcc[i];
    uint smp = uint(acc.w);
    uint seed = (i * 9781u + pp.seed * 6271u + smp * 26699u) ^ 0x9e3779b9u;
    pcg(seed);
    uint W = uint(c.width);
    float dpix = 2 * c.tf / c.width;
    float2 uv = pixel_uv(c, i % W, i / W) + gauss2(seed) * c.sigma_pix * dpix;
    float2 ab = c.sigma_ap > 0 ? gauss2(seed) * c.sigma_ap : float2(0);
    float4 D = c.f + uv.x * c.r + uv.y * c.u;
    float4 d0 = D / length(D);
    float4 a = ab.x * c.r + ab.y * c.u;
    float4 target = c.focus / dot(d0, c.f) * d0 - a;
    float4 x = proj_M(c.x + a);
    float4 n0 = normal_M(x);
    float4 v = normalize(target - dot(target, n0) * n0);
    // M32: базис камеры для нормали AOV (касательный, ⟂ v), переносится вдоль первичного сегмента
    float4 wr = normalize(c.r - dot(c.r, n0) * n0 - dot(c.r, v) * v);
    float4 wu = c.u - dot(c.u, n0) * n0 - dot(c.u, v) * v;
    wu = normalize(wu - dot(wu, wr) * wr);
    float3 aov_alb = float3(0), aov_nrm = float3(0);

    float3 T = float3(1), L = float3(0);
    float w = 0, gamma = c.theta_pix, gb = 0, prev_pdf = 0;
    bool prev_flat = false;
    float4 stats = float4(0);
    g_c_hit = pp.c_hit;
    for (uint depth = 0; depth < pp.max_depth; depth++)
    {
        float tol = min(pp.tol_max, pp.tol_kappa * max(c.theta_pix, gamma));       // M30
        g_w0 = w; g_gamma = gamma;
        float4 x0 = x, v0 = v;
        float s;
        int k;
        bool straight;
        uint seg = trace_segment(x, v, s, k, tol, straight, stats, wr, wu, depth == 0);
        stats.w += 1;
        w += gamma * s;                                                             // M31
        if (seg == SEG_CAP) { stats.x += 1; break; }
        if (seg == SEG_ESCAPE)
        {
            float3 d = exit_value(x, v);
            float wgt = 1;
            if (pp.nee != 0 && PT_HAS_FLAT != 0 && prev_flat && straight)
                wgt = mis_w(prev_pdf, env_pdf(x0.w < 0 ? 1 : 0, v0.xyz), false);
            float3 Le = env_radiance(d, x, gb);
            L += T * wgt * Le;
            if (depth == 0) aov_alb = saturate(Le);
            break;
        }
        // попадание в объект k
        float3 em = mat_e(k).rgb;
        if (depth == 0)
        {
            float4 gk = obj_grad(k, x), nM = normal_M(x);
            float4 ngk = normalize(gk - dot(gk, nM) * nM);
            aov_alb = int(mat_a(k).w) == 2 ? saturate(em) : mat_a(k).rgb;
            aov_nrm = float3(dot(ngk, wr), dot(ngk, wu), -dot(ngk, v));
        }
        if (any(em > 0))
        {
            float wgt = 1;
            if (pp.nee != 0 && PT_HAS_FLAT != 0 && prev_flat && straight && is_light(k) && abs(x0.w - x.w) < 1e-4)
                wgt = mis_w(prev_pdf, light_pdf(x0, k), false);
            L += T * wgt * em;
        }
        int kind = int(mat_a(k).w);
        if (kind == 2) break;
        float4 n, ng, t1, t2;
        shading_frame(k, x, n, ng, t1, t2);
        float3 wo = float3(dot(-v, t1), dot(-v, t2), dot(-v, ng));
        if (wo.z <= 0) { stats.x += 1; break; }           // попадание с обратной стороны — обрыв (как потолок шагов)
        float3 alb = mat_a(k).rgb;
        float rough = mat_e(k).w;
        bool flat_here = PT_HAS_FLAT != 0 && in_flat(x);
        // смещение от поверхности: попадание — только по смене знака g (M26), поэтому достаточно g(x₀) выше пола
        // округления; смещение порядка ε_hit (как было сначала) на крупных пикселях уводило точку за соседнюю
        // поверхность в щелях (шар на полу) — пути обрывались попаданием с обратной стороны
        float eh = max(pp.c_hit * w, ulp_floor(x));
        float off = max(1e-4 * SCALE, 4 * ulp_floor(x));
        float4 xo = proj_M(x + off * ng);
        for (int it = 0; it < 3 && obj_val(k, xo) < 2 * ulp_floor(xo); it++) { off *= 2; xo = proj_M(x + off * ng); }
        if (pp.nee != 0 && flat_here)
        {
            // NEE: сферический источник
            if (pt_nlight() > 0)
            {
                int lk = light_obj(min(uint(urand(seed) * pt_nlight()), uint(pt_nlight() - 1)));
                float3 cc = obj_c(lk).xyz, p3 = xo.xyz;
                float r = obj_r(lk);
                float3 ax3 = cc - p3;
                float dist = length(ax3);
                ax3 /= dist;
                float omc = one_minus_cmax(r, dist * dist);
                float ct = 1 - urand(seed) * omc, stt = sqrt(max(1 - ct * ct, 0.0)), ph = 2 * PI * urand(seed);
                float3 aa = abs(ax3.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
                float3 b1 = normalize(cross(ax3, aa)), b2 = cross(ax3, b1);
                float3 d3 = ct * ax3 + stt * cos(ph) * b1 + stt * sin(ph) * b2;
                float3 oc = p3 - cc;
                float bq = dot(oc, d3);
                float tl = -bq - sqrt(max(bq * bq - (dot(oc, oc) - r * r), 0.0));
                float4 wi4 = float4(d3, 0);
                float pl = 1.0 / (2 * PI * omc) / float(pt_nlight()), pb;
                float3 f = bsdf_eval(kind, alb, rough, wo, float3(dot(wi4, t1), dot(wi4, t2), dot(wi4, ng)), pb);
                bool vis = abs(obj_c(lk).w - x.w) < 1e-4 && any(f > 0) && seg_clear_balls(xo, xo + tl * wi4) &&
                           !line_blocked(xo, wi4, tl * (1 - 1e-4), lk, eh);
                if (vis) L += T * f * mat_e(lk).rgb / pl * mis_w(pl, pb, true);
            }
            // NEE: фон
            {
                uint sh = x.w < 0 ? 1 : 0;
                float3 d3 = env_sample(sh, seed);
                float pe = env_pdf(sh, d3), pb;
                float4 wi4 = float4(d3, 0);
                float3 f = bsdf_eval(kind, alb, rough, wo, float3(dot(wi4, t1), dot(wi4, t2), dot(wi4, ng)), pb);
                if (any(f > 0) && ray_clear_balls(xo, wi4) && !line_blocked(xo, wi4, 1e30, -1, eh))
                    L += T * f * env_radiance(d3, x, gb) / pe * mis_w(pe, pb, true);
            }
        }
        float3 wgt;
        float pb;
        float3 wil = bsdf_sample(kind, alb, rough, wo, urand(seed), urand(seed), wgt, pb);
        if (pb <= 0) break;
        T *= wgt;
        float4 wi4 = wil.x * t1 + wil.y * t2 + wil.z * ng;
        float4 no = normal_M(xo);
        x = xo;
        v = normalize(wi4 - dot(wi4, no) * no);
        prev_flat = flat_here;
        prev_pdf = pb;
        float dg = pp.k_cone * (kind == 0 ? 1.0 : 2 * rough);                         // M31
        gamma += dg;
        gb += dg;
        if (depth + 1 >= pp.rr_depth)
        {
            float q = clamp(max(T.x, max(T.y, T.z)), 0.05, 0.95);
            if (urand(seed) >= q) break;
            T /= q;
        }
    }
    gAcc[i] = acc + float4(L, 1);
    gAcc2[i] = gAcc2[i] + float4(L * L, 0);
    gStats[i] = gStats[i] + stats;
    gAlb[i] = gAlb[i] + float4(aov_alb, 0);
    gNrm[i] = gNrm[i] + float4(aov_nrm, 0);
}

// ---- списки пикселей на GPU (кадр графом) ----------------------------------------------------------
[shader("compute")] [numthreads(1, 1, 1)]
void pt_clear(uint3 id : SV_DispatchThreadID) { gCnt[0] = 0; gCnt[1] = 0; }

// список 0 — объект/смешанный (с gPix[0]), список 1 — фон, помеченный детектором (с gPix[list_off])
[shader("compute")] [numthreads(64, 1, 1)]
void pt_lists(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pp.n) return;
    uint slot;
    if (gClass[i] != 0) { InterlockedAdd(gCnt[0], 1, slot); gPix[slot] = i; }
    else if (gFlag[i] != 0) { InterlockedAdd(gCnt[1], 1, slot); gPix[pp.list_off + slot] = i; }
}

// ---- приложение (DD-007) --------------------------------------------------------------------------
[shader("compute")] [numthreads(64, 1, 1)]
void pt_reset(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pp.n) return;
    gAcc[i] = float4(0); gAcc2[i] = float4(0); gAlb[i] = float4(0); gNrm[i] = float4(0); gStats[i] = float4(0);
}

bool modea_pixel(uint i) { return gClass[i] == 0 && gFlag[i] == 0; }

// сцены без фона (компактные пространства): режим A не нужен — все пиксели в трассировщик, без прохода камеры
[shader("compute")] [numthreads(64, 1, 1)]
void pt_classall(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pp.n) return;
    gClass[i] = 1;
    gFlag[i] = 0;
}

// кадр для денойзера: фон — режим A, остальное — среднее трассировщика; AOV — средние (у фона — цвет фона, нормаль 0)
[shader("compute")] [numthreads(64, 1, 1)]
void pt_compose(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pp.n) return;
    float4 a = gAcc[i];
    float c = max(a.w, 1.0);
    bool mA = modea_pixel(i);
    float3 col = mA ? gColor[i].rgb : a.rgb / c;
    gFrame[i] = float4(col, 1);
    gAlbF[i] = float4(mA ? saturate(col) : gAlb[i].rgb / c, 1);
    gNrmF[i] = float4(mA ? float3(0) : gNrm[i].rgb / c, 1);
}

// вывод: фон режима A — точный, остальное — денойзер или шумный кадр; тонмаппинг x/(1+x), γ 2.2; RGBA8/BGRA8
[shader("compute")] [numthreads(64, 1, 1)]
void pt_present(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= pp.n) return;
    float3 col = modea_pixel(i) ? gColor[i].rgb : (pp.use_den != 0 ? gDen[i].rgb : gFrame[i].rgb);
    col = max(col * pp.exposure, 0.0);
    col = pow(col / (1 + col), 1.0 / 2.2);
    uint3 q = uint3(saturate(col) * 255 + 0.5);
    gPresent[i] = pp.bgra != 0 ? (q.b | (q.g << 8) | (q.r << 16) | (255u << 24)) : (q.r | (q.g << 8) | (q.b << 16) | (255u << 24));
}

[shader("compute")] [numthreads(1, 1, 1)]
void pt_args(uint3 id : SV_DispatchThreadID)
{
    gArgs[0] = (gCnt[0] + 63) / 64; gArgs[1] = 1; gArgs[2] = 1;
    gArgs[3] = (gCnt[1] + 63) / 64; gArgs[4] = 1; gArgs[5] = 1;
}
#endif

// ---- проход камеры (DD-005 п. 2): центральные лучи с полями Якоби, объекты, класс пикселя -----------------
#if NJ > 0
static float g_fp_pix, g_fp_ap;     // 3σ пикселя (в пикселях) и 3σ апертуры

// c_mix · радиус следа: поля пикселя (J₂, J₃ — на пиксель) и апертуры (J₀, J₁ — на единицу сдвига линзы)
float pt_footprint(Fields E)
{
    float rp = g_fp_pix * max(length(E.J[2]), length(E.J[3]));
    float ra = g_fp_ap * max(length(E.J[0]), length(E.J[1]));
    return pp.c_mix * max(rp, ra);
}

// прямой пролёт с переносом полей (M25) и учётом следа: 0 — уход, 1 — попадание, 2 — вход в шар, 3 — потолок
uint flat_flight_E(inout float4 x, float4 v, inout float s, inout Fields E, inout float mix)
{
    g_sbase = 0;
    for (int it = 0; it < 1024; it++)
    {
        float eps = pt_eps_hit(s, x);
        int k;
        float g = gmin_k(x, k);
        mix = min(mix, g - pt_footprint(E));
        if (g < pt_hit_floor(x)) return 1;
        if (ray_clear_balls(x, v) && ahead_clear(x, v)) return 0;
        float tb = ball_entry(x, v), step = g + 0.5 * eps;
        float t = min(tb, step);
        x += t * v;
        s += t;
        [ForceUnroll] for (int j = 0; j < 4; j++) E.J[j] += t * E.Jp[j];
        if (tb <= step) return 2;
        if (s > PP_S_MAX) return 3;
    }
    return 3;
}

[shader("compute")] [numthreads(64, 1, 1)]
void pt_camera(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(gCam);
    uint W = uint(c.width), H = uint(c.height);
    uint i = id.x;
    if (i >= W * H) return;
    float2 uv = pixel_uv(c, i % W, i / W);
    // M16: направление пикселя и начальные данные полей (как modea_central)
    float4 D = c.f + uv.x * c.r + uv.y * c.u;
    float nD = length(D);
    float4 d0 = D / nD;
    float Pl = c.focus / dot(d0, c.f);
    float dpix = 2 * c.tf / c.width;
    float4 pr = c.r - dot(d0, c.r) * d0, pu = c.u - dot(d0, c.u) * d0;
    Fields E;
    E.J[0] = c.r; E.J[1] = c.u; E.J[2] = float4(0); E.J[3] = float4(0);
    E.Jp[0] = -pr / Pl; E.Jp[1] = -pu / Pl; E.Jp[2] = dpix * pr / nD; E.Jp[3] = dpix * pu / nD;
    g_fp_pix = 3 * c.sigma_pix; g_fp_ap = 3 * c.sigma_ap;
    g_c_hit = pp.c_hit; g_w0 = 0; g_gamma = c.theta_pix;
    Params p;
    p.n = 1; p.tol = min(pp.tol_max, pp.tol_kappa * c.theta_pix); p.h0 = 0.05 * SCALE; p.s_max = PP_S_MAX; p.h_max = 0;
    p.h_min = 1e-7; p.n_max = pp.n_max; p.escape_mode = PT_HAS_FLAT != 0 ? 3 : 1; p.escape_param = PP_ELL_MAX;
    p.alpha = -1; p.xscale = 0; p.fixed = 0;
    float4 x = c.x, v = d0;
    float s = 0, mix = 1e30;
    uint cls = 0;
    float4 stats = float4(0);
    for (int part = 0; part < 64; part++)
    {
        if (PT_HAS_FLAT != 0 && in_flat(x))
        {
            uint r = flat_flight_E(x, v, s, E, mix);
            if (r == 0) break;
            if (r == 1) { cls = 1; break; }
            if (r == 3) { cls = 2; break; }
        }
        g_sbase = s;
        RayState st;
        st.x = x; st.v = v; st.E = E;
        st.s = 0; st.h = 0; st.nacc = 0; st.nrej = 0; st.npass = 0; st.status = RUNNING;
        st.mix = 1e30; st.transport = 0; st.wr = float4(0); st.wu = float4(0);
        p.s_max = PP_S_MAX - s;
        trace_ray(st, p);
        stats.y += st.npass;
        mix = min(mix, st.mix);
        x = st.x; v = st.v; E = st.E; s += st.s;
        if (st.status == DONE_HIT) { cls = 1; break; }
        if (st.status != DONE_ESCAPE) { cls = 2; break; }
        if (PT_HAS_FLAT == 0) break;
    }
    if (cls == 0 && mix < 0) cls = 2;
    gClass[i] = cls;
    gStatus[i] = cls == 0 ? DONE_ESCAPE : DONE_HIT;
    if (cls != 0) { gDir[i] = float4(0, 0, 1, x.w >= 0 ? 1.0 : -1.0); [ForceUnroll] for (int k = 0; k < 4; k++) gDd[4 * i + k] = float4(0); return; }
    // ∂v = J′ − ⟨SJ, v⟩n (M17); d∞ и ∂d∞ дуальными числами (M24) — как modea_central
    Jet15 s15[4];
    [ForceUnroll] for (int k = 0; k < 4; k++) { float4 e = float4(0); e[k] = 1; s15[k] = Jet15(x[k], e, float4x4(0)); }
    Jet15 Jx = sdf(s15[0], s15[1], s15[2], s15[3]);
    float gn = length(Jx.g);
    float4 n = Jx.g / gn;
    float4 Pv = v - dot(v, n) * n;
    float4 dv[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        float4 PJ = E.J[k] - dot(E.J[k], n) * n;
        dv[k] = E.Jp[k] - dot(PJ, mul(Jx.H, Pv)) / gn * n;
    }
    Dual4 X[4], V[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        X[k] = Dual4(x[k], float4(E.J[0][k], E.J[1][k], E.J[2][k], E.J[3][k]));
        V[k] = Dual4(v[k], float4(dv[0][k], dv[1][k], dv[2][k], dv[3][k]));
    }
    Dual4 d[3];
    exit_dir(X, V, d);
    gDir[i] = float4(d[0].a, d[1].a, d[2].a, x.w >= 0 ? 1.0 : -1.0);
    [ForceUnroll] for (int k = 0; k < 4; k++) gDd[4 * i + k] = float4(d[0].b[k], d[1].b[k], d[2].b[k], 0);
}
#endif

// ---- M33: шаг камеры по M (этап 5, DD-007); -DCAM_STEP --------------------------------------------------
#ifdef CAM_STEP
struct CamMove
{
    float4 move;        // (a, b, c, 0): перемещение a·f + b·r + c·u; pt_snap: w ≠ 0 — спроецировать и камеру
    float4 rot;         // (рыскание, тангаж, крен, margin столкновений)
    uint substeps;
};
[[vk::push_constant]] ConstantBuffer<CamMove> cm;

float4 gs_one(float4 w, float4 n, float4 a, float4 b) { w -= dot(w, n) * n + dot(w, a) * a + dot(w, b) * b; return normalize(w); }

[shader("compute")] [numthreads(1, 1, 1)]
void cam_step(uint3 id : SV_DispatchThreadID)
{
    float4 x = gCam[0], f = gCam[1], r = gCam[2], u = gCam[3];
    float ps = cm.rot.x, th = cm.rot.y, ph = cm.rot.z;
    float4 f1 = cos(ps) * f + sin(ps) * r, r1 = -sin(ps) * f + cos(ps) * r;
    f = cos(th) * f1 + sin(th) * u; u = -sin(th) * f1 + cos(th) * u; r = r1;
    float4 r2 = cos(ph) * r + sin(ph) * u; u = -sin(ph) * r + cos(ph) * u; r = r2;
    float4 m = cm.move.x * f + cm.move.y * r + cm.move.z * u;
    float s = length(m);
    float4 X = x, n;
    if (s > 0)
    {
        float4 V = m / s;
        float h = s / float(cm.substeps);
        for (uint k = 0; k < cm.substeps; k++)
        {
            float4 k1v = rhs7(X, V, ALPHA).acc;
            float4 k2v = rhs7(X + 0.5 * h * V, V + 0.5 * h * k1v, ALPHA).acc;
            float4 k3v = rhs7(X + 0.5 * h * (V + 0.5 * h * k1v), V + 0.5 * h * k2v, ALPHA).acc;
            float4 k4v = rhs7(X + h * (V + 0.5 * h * k2v), V + h * k3v, ALPHA).acc;
            X = X + h / 6 * (V + 2 * (V + 0.5 * h * k1v) + 2 * (V + 0.5 * h * k2v) + (V + h * k3v));
            V = V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v);
            X = proj_M(X);
            n = normal_M(X);
            V = normalize(V - dot(V, n) * n);
            f -= dot(f, n) * n; r -= dot(r, n) * n; u -= dot(u, n) * n;      // перенос проекцией (M32/M33)
        }
    }
    n = normal_M(X);
    f = gs_one(f, n, float4(0), float4(0));
    r = gs_one(r, n, f, float4(0));
    u = gs_one(u, n, f, r);
    if (s > 0 && pt_gmin(X) >= cm.rot.w) x = X;                          // столкновение — без перемещения
    gCam[0] = x; gCam[1] = f; gCam[2] = r; gCam[3] = u;
}

// DD-008: центры объектов с флагом on_M — на M (M3, Ньютон по ∇f); в плоской зоне ∇f ∥ e_w — меняется только w
[shader("compute")] [numthreads(1, 1, 1)]
void pt_snap(uint3 id : SV_DispatchThreadID)
{
    for (int k = 0; k < PT_N; k++)
    {
        int row = OBJ_BASE + 9 + 6 * k;
        if (gCam[row + 5].y == 0 || int(gCam[row + 1].x) == 2) continue;
        float4 x = gCam[row];
        for (int it = 0; it < 30; it++) { Eval e = rhs7(x, float4(0), 0.0); x -= e.f * e.g / dot(e.g, e.g); }
        if (all(isfinite(x))) gCam[row] = x;
    }
#if SCENE == SCENE_DYNAMIC
    // смена многообразия в окне (DD-009): точка камеры — на M (M3, Ньютон по ∇f); нормаль — в строку MF_BASE + 6,
    // невязка f — в MF_BASE + 5.w; касательный базис по подсказкам f, u, r строит хост (как geo4d.camera.make_camera)
    if (cm.move.w != 0)
    {
        float4 x = gCam[0];
        for (int it = 0; it < 30; it++) { Eval e = rhs7(x, float4(0), 0.0); x -= e.f * e.g / dot(e.g, e.g); }
        Eval e = rhs7(x, float4(0), 0.0);
        gCam[0] = x;
        gCam[MF_BASE + 6] = e.g / length(e.g);
        gCam[MF_BASE + 5].w = e.f;
    }
#endif
}
#endif
