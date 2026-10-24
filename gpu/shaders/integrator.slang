// Интегратор луча (DD-001) и канал Якоби (DD-003) — порт prototype/geo4d/{rhs,jacobi,integrators}.py во float32.
// Компиляция: -DSCENE=<id>, -DNJ=0|4 (число полей Якоби), -DJACOBI_JET=0 (Jet15) | 1 (HVP, M23).
#include "scenes.slang"

#ifndef NJ
#define NJ 0
#endif
#ifndef JACOBI_JET
#define JACOBI_JET 0
#endif

static const uint RUNNING = 0, DONE_SMAX = 1, DONE_ESCAPE = 2, DONE_NMAX = 3, DONE_HIT = 4;

#ifdef PT_OBJECTS
// Режим B (DD-005): предел шага по объектам M26 и остановка при попадании; функции pt_gmin, pt_eps_hit,
// pt_escape — в pathtrace.slang. Предыдущее принятое состояние — для эрмитова уточнения M27.
#endif

struct Params
{
    uint n;             // число лучей
    float tol;          // M8
    float h0;           // начальный шаг
    float s_max;
    float h_max;        // ≤ 0 — hcap сцены
    float h_min;
    uint n_max;         // потолок попыток шага
    uint escape_mode;   // 0 — нет, 1 — Эллис (|ℓ| ≥ param), 2 — компактная (ρ ≥ param и удаляется)
    float escape_param;
    float alpha;        // < 0 — ALPHA сцены
    float xscale;       // ≤ 0 — SCALE сцены
    uint fixed;         // 1 — фиксированный шаг h0
};

struct Fields { float4 J[4]; float4 Jp[4]; };       // E = (J, J′)

struct Eval { float4 acc; float f; float4 g; Fields dE; };

// ---- правая часть ------------------------------------------------------------------------------

// M2 на Jet7 (один проход)
Eval rhs7(float4 x, float4 v, float alpha)
{
    Jet7 s[4];
    [ForceUnroll] for (int k = 0; k < 4; k++) { float4 e = float4(0); e[k] = 1; s[k] = Jet7(x[k], v[k], 0, e); }
    Jet7 J = sdf(s[0], s[1], s[2], s[3]);
    Eval r;
    r.f = J.a; r.g = J.d;
    float lam = -(2 * J.c + 2 * alpha * J.b + alpha * alpha * J.a) / dot(J.d, J.d);
    r.acc = lam * J.d;
    return r;
}

#if NJ > 0
float4 projT(float4 u, float4 n) { return u - dot(u, n) * n; }

// M15: правая часть полей по Sv, SJ_k (S — M14), общая для Jet15 и HVP
void jacobi_tail(inout Eval r, float4 v, Fields E, float4 n, float4 Sv, float4 SJ[4])
{
    float vSv = dot(v, Sv);
    [ForceUnroll] for (int k = 0; k < NJ; k++)
    {
        r.dE.J[k] = E.Jp[k] - dot(E.J[k], Sv) * n;
        r.dE.Jp[k] = -vSv * SJ[k] + dot(SJ[k], v) * Sv - dot(E.Jp[k], Sv) * n;
    }
}

#if JACOBI_JET == 0
// Один проход Jet15: f, ∇f, H
Eval rhsJ(float4 x, float4 v, Fields E, float alpha)
{
    Jet15 s[4];
    [ForceUnroll] for (int k = 0; k < 4; k++) { float4 e = float4(0); e[k] = 1; s[k] = Jet15(x[k], e, float4x4(0)); }
    Jet15 J = sdf(s[0], s[1], s[2], s[3]);
    Eval r;
    r.f = J.a; r.g = J.g;
    float g2 = dot(J.g, J.g), gn = sqrt(g2);
    float4 Hv = mul(J.H, v);
    r.acc = -(dot(v, Hv) + 2 * alpha * dot(J.g, v) + alpha * alpha * J.a) / g2 * J.g;
    float4 n = J.g / gn;
    float4 Sv = projT(mul(J.H, projT(v, n)), n) / gn;               // M14: P H P v / |∇f|
    float4 SJ[4];
    [ForceUnroll] for (int k = 0; k < NJ; k++) SJ[k] = projT(mul(J.H, projT(E.J[k], n)), n) / gn;
    jacobi_tail(r, v, E, n, Sv, SJ);
    return r;
}
#else
JetHVP hvp(float4 x, float4 w)
{
    JetHVP s[4];
    [ForceUnroll] for (int k = 0; k < 4; k++) { float4 e = float4(0); e[k] = 1; s[k] = JetHVP(x[k], w[k], e, float4(0)); }
    return sdf(s[0], s[1], s[2], s[3]);
}

// 1 + NJ проходов HVP (M23)
Eval rhsJ(float4 x, float4 v, Fields E, float alpha)
{
    JetHVP J0 = hvp(x, v);
    Eval r;
    r.f = J0.a; r.g = J0.d;
    float g2 = dot(J0.d, J0.d), gn = sqrt(g2);
    r.acc = -(dot(v, J0.m) + 2 * alpha * J0.b + alpha * alpha * J0.a) / g2 * J0.d;
    float4 n = J0.d / gn;
    float4 Sv = projT(J0.m, n) / gn;
    float4 SJ[4];
    for (int k = 0; k < NJ; k++) SJ[k] = projT(hvp(x, projT(E.J[k], n)).m, n) / gn;
    jacobi_tail(r, v, E, n, Sv, SJ);
    return r;
}
#endif
#endif

Eval rhs(float4 x, float4 v, Fields E, float alpha)
{
#if NJ > 0
    return rhsJ(x, v, E, alpha);
#else
    return rhs7(x, v, alpha);
#endif
}

#if NJ > 0
#if JACOBI_JET == 0
static const uint PASSES_PER_EVAL = 1;
#else
static const uint PASSES_PER_EVAL = 1 + NJ;
#endif
#else
static const uint PASSES_PER_EVAL = 1;
#endif

// E + h·Σ cᵢ dEᵢ
Fields fma(Fields E, float h, Fields d)
{
    Fields r;
    [ForceUnroll] for (int k = 0; k < NJ; k++) { r.J[k] = E.J[k] + h * d.J[k]; r.Jp[k] = E.Jp[k] + h * d.Jp[k]; }
    return r;
}

// ---- выход -------------------------------------------------------------------------------------

bool escaped(float4 x, float4 v, Params p)
{
    if (p.escape_mode == 1) { float r2 = p.escape_param * p.escape_param + SCALE * SCALE; return dot(x.xyz, x.xyz) >= r2; }
    if (p.escape_mode == 2) return dot(x.xyz, x.xyz) >= p.escape_param * p.escape_param && dot(x.xyz, v.xyz) > 0;
#ifdef PT_OBJECTS
    if (p.escape_mode == 3) return pt_escape(x, v);
#endif
    return false;
}

// ---- BS32 (M7) с FSAL «stale», проекцией M3 и нормой M8 ----------------------------------------

struct RayState
{
    float4 x, v;
    Fields E;
    float s, h;
    uint nacc, nrej, npass, status;
#ifdef PT_OBJECTS
    float4 xp, vp;      // предыдущее принятое состояние (M27)
    float sp;
#endif
};

// Начальный шаг луча
void init_step(inout RayState st, Params p)
{
    float hc = p.h_max > 0 ? p.h_max : hcap(st.x);
    st.h = min(min(p.h0, hc), p.s_max - st.s);
#ifdef PT_OBJECTS
    st.h = min(st.h, max(pt_gmin(st.x), 0.0) + pt_eps_hit(st.s, st.x));      // M26
    st.xp = st.x; st.vp = st.v; st.sp = st.s;
#endif
}

void trace_ray(inout RayState st, Params p)
{
    init_step(st, p);
    trace_k(st, p, 0xffffffffu);
}

// До K попыток шага; k1 вычисляется заново в начале (FSAL между диспетчеризациями не хранится, DD-004 п. 5)
void trace_k(inout RayState st, Params p, uint K)
{
    float alpha = p.alpha < 0 ? ALPHA : p.alpha;
    float xs = p.xscale <= 0 ? SCALE : p.xscale;
    Eval k1 = rhs(st.x, st.v, st.E, alpha);
    st.npass += PASSES_PER_EVAL;
    uint budget = K;
    while (st.status == RUNNING && budget-- > 0)
    {
        float h = st.h;
        float4 x = st.x, v = st.v;
        // стадия 2: c = 1/2
        float4 x2 = x + 0.5 * h * v, v2 = v + 0.5 * h * k1.acc;
        Eval k2 = rhs(x2, v2, fma(st.E, 0.5 * h, k1.dE), alpha);
        // стадия 3: c = 3/4
        float4 x3 = x + 0.75 * h * v2, v3 = v + 0.75 * h * k2.acc;
        Eval k3 = rhs(x3, v3, fma(st.E, 0.75 * h, k2.dE), alpha);
        // y⁺ = y + h(2/9 k1 + 1/3 k2 + 4/9 k3)
        float4 xn = x + h * ((2.0 / 9) * v + (1.0 / 3) * v2 + (4.0 / 9) * v3);
        float4 vn = v + h * ((2.0 / 9) * k1.acc + (1.0 / 3) * k2.acc + (4.0 / 9) * k3.acc);
        Fields En = fma(fma(fma(st.E, h * (2.0 / 9), k1.dE), h * (1.0 / 3), k2.dE), h * (4.0 / 9), k3.dE);
        Eval k4 = rhs(xn, vn, En, alpha);
        st.npass += 3 * PASSES_PER_EVAL;
        float4 n = k4.g / length(k4.g);
        float err = 0;
        if (p.fixed == 0)
        {
            // e = h Σ (b − b̂) k, b − b̂ = (−5/72, 1/12, 1/9, −1/8)
            float4 ex = h * ((-5.0 / 72) * v + (1.0 / 12) * v2 + (1.0 / 9) * v3 + (-1.0 / 8) * vn);
            float4 ev = h * ((-5.0 / 72) * k1.acc + (1.0 / 12) * k2.acc + (1.0 / 9) * k3.acc + (-1.0 / 8) * k4.acc);
            ex -= dot(ex, n) * n;
            ev -= dot(ev, n) * n;
            err = max(length(ex) / xs, length(ev));               // M8
        }
        bool accept = p.fixed != 0 || err <= p.tol || h <= p.h_min;
        if (accept)
        {
            // M3: шаг Ньютона, проекция v и полей
            float g2 = dot(k4.g, k4.g);
            st.x = xn - k4.f * k4.g / g2;
            float4 vp = vn - dot(vn, n) * n;
            st.v = vp / length(vp);
#if NJ > 0
            [ForceUnroll] for (int k = 0; k < NJ; k++) { st.E.J[k] = En.J[k] - dot(En.J[k], n) * n; st.E.Jp[k] = En.Jp[k] - dot(En.Jp[k], n) * n; }
#endif
#ifdef PT_OBJECTS
            st.xp = x; st.vp = v; st.sp = st.s;
#endif
            k1 = k4;                                                // FSAL «stale» (DD-001)
            st.s += h;
            st.nacc++;
        }
        else
            st.nrej++;
        float hn = p.fixed != 0 ? p.h0 : h * clamp(0.9 * pow(p.tol / max(err, 1e-30), 1.0 / 3.0), 0.2, 5.0);
        hn = min(hn, p.h_max > 0 ? p.h_max : hcap(st.x));
#ifdef PT_OBJECTS
        float gx = pt_gmin(st.x), eh = pt_eps_hit(st.s, st.x);
        hn = min(hn, max(gx, 0.0) + eh);                            // M26: h ≤ g + ε_hit
#endif
        float remaining = p.s_max - st.s;
        st.h = max(min(hn, remaining), 0.0);
        if (remaining <= 1e-6 * p.s_max) st.status = DONE_SMAX;
        if (accept && escaped(st.x, st.v, p)) st.status = DONE_ESCAPE;
#ifdef PT_OBJECTS
        if (accept && gx < pt_hit_floor(st.x)) st.status = DONE_HIT;    // смена знака g (или |g| у пола округления)
#endif
        if (st.status == RUNNING && st.nacc + st.nrej >= p.n_max) st.status = DONE_NMAX;
    }
}
