// Режим A на GPU (DD-003, DD-004): центральный луч с полями Якоби → выход и ∂d∞ (дуальные числа, M24) →
// след (M18) → SampleGrad; детектор (M20); стохастический фолбэк. Порт prototype/geo4d/modea.py.
// Компиляция: -DSCENE=<id> -DEXIT=1 (Эллис) | 2 (компактная) -DKERNEL=1 (центральные, -DNJ=4) | 2 (оттенение) |
//             3 (детектор) | 4 (фолбэк, -DNJ=0).
#include "integrator.slang"

#ifndef EXIT
#define EXIT 1
#endif
#ifndef KERNEL
#define KERNEL 1
#endif

// Камера: 7 float4 (см. prototype/geo4d/gpu.py: camera_buffer)
struct Cam
{
    float4 x, f, r, u;
    float tf, width, height, focus;
    float sigma_ap, flight_r, k_grad, sigma_pix;
    float theta_pix, det_lo, det_hi, tau;
    float filter_mode, cube_n, pad0, pad1;    // 0 — аппаратный SampleGrad, 1 — программный фильтр M19
};

Cam load_cam(RWStructuredBuffer<float4> b)
{
    Cam c;
    c.x = b[0]; c.f = b[1]; c.r = b[2]; c.u = b[3];
    float4 a = b[4], q = b[5], w = b[6], m = b[7];
    c.tf = a.x; c.width = a.y; c.height = a.z; c.focus = a.w;
    c.sigma_ap = q.x; c.flight_r = q.y; c.k_grad = q.z; c.sigma_pix = q.w;
    c.theta_pix = w.x; c.det_lo = w.y; c.det_hi = w.z; c.tau = w.w;
    c.filter_mode = m.x; c.cube_n = m.y; c.pad0 = m.z; c.pad1 = m.w;
    return c;
}

// ---- M19: координаты грани и программный анизотропный гауссов фильтр -----------------------------

float2 face_st(float3 d, out uint face)
{
    float3 a = abs(d);
    float sc, tc, ma;
    if (a.x >= a.y && a.x >= a.z) { face = d.x > 0 ? 0 : 1; ma = a.x; sc = d.x > 0 ? -d.z : d.z; tc = -d.y; }
    else if (a.y >= a.z) { face = d.y > 0 ? 2 : 3; ma = a.y; sc = d.x; tc = d.y > 0 ? d.z : -d.z; }
    else { face = d.z > 0 ? 4 : 5; ma = a.z; sc = d.z > 0 ? d.x : -d.x; tc = -d.y; }
    return 0.5 * (float2(sc, tc) / ma + 1);
}

// масштаб «направление → тексели уровня 0» вдоль касательной оси (численно, на той же грани)
float texels_per_radian(float3 d, float3 axis, float N)
{
    const float delta = 1e-3;
    uint f0, f1;
    float2 st0 = face_st(d, f0);
    float2 st1 = face_st(normalize(d + delta * axis), f1);
    if (f1 != f0) st1 = 2 * st0 - face_st(normalize(d - delta * axis), f1);
    return length(st1 - st0) * N / delta;
}

float4 sample_ellipse_sw(SamplerCubeArray<float4> cube, float3 d, float layer, float3 a1, float s1, float3 a2, float s2, float N)
{
    float tpr = texels_per_radian(d, a1, N);
    float smin = s2 * texels_per_radian(d, a2, N), smaj = s1 * tpr;
    float L = log2(max(max(2 * smin, 4 * smaj / 15), 1.0));
    float blur = 0.5 * exp2(L);
    float seff = sqrt(max(smaj * smaj - blur * blur, 0.0));
    int n = int(clamp(ceil(2 * seff / blur) + 1, 1.0, 16.0));
    float4 acc = float4(0);
    float wsum = 0;
    for (int k = 0; k < n; k++)
    {
        float u = n > 1 ? (float(k) - 0.5 * (n - 1)) / float(n - 1) * 4.0 : 0.0;
        float w = exp(-0.5 * u * u);
        acc += w * cube.SampleLevel(float4(normalize(d + u * (seff / tpr) * a1), layer), L);
        wsum += w;
    }
    return acc / wsum;
}

float2 pixel_uv(Cam c, uint px, uint py)
{
    return float2(((px + 0.5) / c.width * 2 - 1) * c.tf, (1 - (py + 0.5) / c.height * 2) * c.tf * c.height / c.width);
}

// ---- функции выхода на IReal (M24) ------------------------------------------------------------

static const float GL_X[8] = { 0.019855071751231856, 0.10166676129318664, 0.2372337950418355, 0.4082826787521751,
                               0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681 };
static const float GL_W[8] = { 0.05061426814518813, 0.11119051722668724, 0.15685332293894363, 0.18134189168918100,
                               0.18134189168918100, 0.15685332293894363, 0.11119051722668724, 0.05061426814518813 };

// Эллис (M10/M17/M24): d∞ = cos√y·r̂ + |q|·G·sinc√y·e⊥, y = L²G²
__generic<T : IReal> void ellis_dir(T x[4], T v[4], out T d[3])
{
    const float b = SCALE;
    T rq = jlength3(x[0], x[1], x[2]);
    T inv = 1.0 / rq;
    T rh[3] = { x[0] * inv, x[1] * inv, x[2] * inv };
    T dt = v[0] * rh[0] + v[1] * rh[1] + v[2] * rh[2];
    T ep[3] = { v[0] - dt * rh[0], v[1] - dt * rh[1], v[2] - dt * rh[2] };
    T L2 = rq * rq * (ep[0] * ep[0] + ep[1] * ep[1] + ep[2] * ep[2]);
    T U = 1.0 / jsqrt(rq * rq - b * b);
    T G = T.cst(0);
    [ForceUnroll] for (int k = 0; k < 8; k++)
    {
        T u = U * GL_X[k];
        G = G + GL_W[k] / (jsqrt(1.0 + (b * b) * u * u) * jsqrt(1.0 + ((b * b) - L2) * u * u));
    }
    G = G * U;
    T y = L2 * G * G;
    T cy = jcos_sqrt(y), sy = jsinc_sqrt(y);
    [ForceUnroll] for (int i = 0; i < 3; i++) d[i] = cy * rh[i] + rq * G * sy * ep[i];
}

// Компактная червоточина: d∞ = v_q/|v_q|
__generic<T : IReal> void compact_dir(T x[4], T v[4], out T d[3])
{
    T n = jlength3(v[0], v[1], v[2]);
    [ForceUnroll] for (int i = 0; i < 3; i++) d[i] = v[i] / n;
}

__generic<T : IReal> void exit_dir(T x[4], T v[4], out T d[3])
{
#if EXIT == 1
    ellis_dir(x, v, d);
#else
    compact_dir(x, v, d);
#endif
}

float3 exit_value(float4 x, float4 v)
{
    Real X[4] = { Real(x.x), Real(x.y), Real(x.z), Real(x.w) };
    Real V[4] = { Real(v.x), Real(v.y), Real(v.z), Real(v.w) };
    Real d[3];
    exit_dir(X, V, d);
    return float3(d[0].a, d[1].a, d[2].a);
}

// ---- центральные лучи ---------------------------------------------------------------------------
#if KERNEL == 1 && NJ > 0

[[vk::push_constant]] ConstantBuffer<Params> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gDir;     // d∞.xyz, лист (+1/−1)
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gDd;      // 4 float4 на пиксель: ∂d∞ по полям (ap_r, ap_u, pix_r, pix_u)
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gStats;   // s, nacc, nrej, проходов
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> gStatus;

[shader("compute")] [numthreads(64, 1, 1)]
void modea_central(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(gCam);
    uint W = uint(c.width), H = uint(c.height);
    uint i = id.x;
    if (i >= W * H) return;
    float2 uv = pixel_uv(c, i % W, i / W);
    // M16: направление пикселя и начальные данные полей
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
    bool active = true;
    if (c.flight_r > 0)
    {
        // M25: аналитический пролёт до сферы |q| = flight_r (центр 0) с переносом полей
        float3 q0 = c.x.xyz, dd = d0.xyz;
        float bq = dot(dd, q0), disc = bq * bq - (dot(q0, q0) - c.flight_r * c.flight_r);
        float t = -bq - sqrt(max(disc, 0.0));
        active = disc > 0 && t > 0;
        if (active)
        {
            st.x = c.x + t * d0;
            [ForceUnroll] for (int k = 0; k < 4; k++) st.E.J[k] += t * st.E.Jp[k];
        }
        else
            st.status = DONE_ESCAPE;
    }
    if (active) trace_ray(st, pc);
    // ∂v = J′ − ⟨SJ, v⟩n (M17); S из Jet15 в точке выхода (в плоской зоне S = 0)
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
        float sjv = dot(PJ, mul(Jx.H, Pv)) / gn;
        dv[k] = st.E.Jp[k] - sjv * n;
    }
    // M24: d∞ и ∂d∞ одним проходом дуальных чисел по 4 полям
    Dual4 X[4], V[4];
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        X[k] = Dual4(st.x[k], float4(st.E.J[0][k], st.E.J[1][k], st.E.J[2][k], st.E.J[3][k]));
        V[k] = Dual4(st.v[k], float4(dv[0][k], dv[1][k], dv[2][k], dv[3][k]));
    }
    Dual4 d[3];
    exit_dir(X, V, d);
    gDir[i] = float4(d[0].a, d[1].a, d[2].a, st.x.w >= 0 ? 1.0 : -1.0);
    [ForceUnroll] for (int k = 0; k < 4; k++) gDd[4 * i + k] = float4(d[0].b[k], d[1].b[k], d[2].b[k], 0);
    gStats[i] = float4(st.s, float(st.nacc), float(st.nrej), float(st.npass));
    gStatus[i] = st.status;
}
#endif

// ---- оттенение: след M18 → SampleGrad ----------------------------------------------------------
#if KERNEL == 2
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gDir;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gDd;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gStatus;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> gColor;
[[vk::binding(5, 0)]] SamplerCubeArray<float4> gCube;

[shader("compute")] [numthreads(64, 1, 1)]
void modea_shade(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(gCam);
    uint i = id.x;
    if (i >= uint(c.width) * uint(c.height)) return;
    if (gStatus[i] != DONE_ESCAPE) { gColor[i] = float4(1, 0, 1, 1); return; }
    float4 dd = gDir[i];
    float3 d = dd.xyz;
    // касательный базис сферы направлений
    float3 a = abs(d.x) < 0.9 ? float3(1, 0, 0) : float3(0, 1, 0);
    float3 t1 = normalize(a - dot(a, d) * d), t2 = cross(d, t1);
    float sxx = 0, sxy = 0, syy = 0;
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        float3 g = gDd[4 * i + k].xyz;
        float w = k < 2 ? c.sigma_ap * c.sigma_ap : c.sigma_pix * c.sigma_pix;
        float gx = dot(g, t1), gy = dot(g, t2);
        sxx += w * gx * gx; sxy += w * gx * gy; syy += w * gy * gy;
    }
    // собственные значения и векторы 2×2
    float tr = 0.5 * (sxx + syy), df = sqrt(max(0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy, 0.0));
    float l1 = tr + df, l2 = max(tr - df, 0.0);
    float2 e1 = abs(sxy) > 1e-30 ? normalize(float2(l1 - syy, sxy)) : (sxx >= syy ? float2(1, 0) : float2(0, 1));
    float3 a1 = e1.x * t1 + e1.y * t2, a2 = -e1.y * t1 + e1.x * t2;
    float layer = dd.w > 0 ? 0 : 1;
    if (c.filter_mode > 0.5)
        gColor[i] = sample_ellipse_sw(gCube, d, layer, a1, sqrt(l1), a2, sqrt(l2), c.cube_n);
    else
    {
        float3 gx3 = c.k_grad * sqrt(l1) * a1, gy3 = c.k_grad * sqrt(l2) * a2;
        gColor[i] = gCube.SampleGrad(float4(d, layer), gx3, gy3);
    }
}
#endif

// ---- детектор нелинейности M20 -------------------------------------------------------------------
#if KERNEL == 3
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> gDir;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gDd;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> gStatus;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> gFlag;

[shader("compute")] [numthreads(64, 1, 1)]
void modea_detect(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(gCam);
    uint W = uint(c.width), H = uint(c.height);
    uint i = id.x;
    if (i >= W * H) return;
    uint px = i % W, py = i / W;
    bool flag = gStatus[i] != DONE_ESCAPE;
    float3 d = gDir[i].xyz;
    float sh = gDir[i].w;
    float3 A0 = gDd[4 * i + 2].xyz, A1 = gDd[4 * i + 3].xyz;      // пиксель вправо, вверх
    // |det| Грама против плоского pinhole того же пикселя
    float2 uv = pixel_uv(c, px, py);
    float4 D = c.f + uv.x * c.r + uv.y * c.u;
    float nD = length(D);
    float4 d0 = D / nD;
    float dpix = 2 * c.tf / c.width;
    float4 q0 = dpix * (c.r - dot(d0, c.r) * d0) / nD, q1 = dpix * (c.u - dot(d0, c.u) * d0) / nD;
    float g = sqrt(abs(dot(A0, A0) * dot(A1, A1) - dot(A0, A1) * dot(A0, A1)));
    float g0 = sqrt(abs(dot(q0, q0) * dot(q1, q1) - dot(q0, q1) * dot(q0, q1)));
    float rho = g / g0;
    flag = flag || rho < c.det_lo || rho > c.det_hi;
    // невязка второго порядка и смена листа по 4 соседям
    int2 off[4] = { int2(1, 0), int2(-1, 0), int2(0, -1), int2(0, 1) };
    float3 pred[4] = { d + A0, d - A0, d + A1, d - A1 };
    [ForceUnroll] for (int k = 0; k < 4; k++)
    {
        int nx = int(px) + off[k].x, ny = int(py) + off[k].y;
        if (nx < 0 || ny < 0 || nx >= int(W) || ny >= int(H)) continue;
        uint j = uint(ny) * W + uint(nx);
        float4 dn = gDir[j];
        flag = flag || gStatus[j] != DONE_ESCAPE || dn.w != sh;
        flag = flag || length(normalize(pred[k]) - dn.xyz) / c.theta_pix > c.tau;
    }
    gFlag[i] = flag ? 1 : 0;
}
#endif

// ---- фолбэк: стохастические лучи пикселя (гаусс пикселя и апертуры), мип 0 -------------------------
#if KERNEL == 4
[[vk::push_constant]] ConstantBuffer<Params> pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> gCam;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint4> gFb;       // [0] = (число пикселей, spp, seed, 0)
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> gPix;       // индексы пикселей
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> gColor;   // по одному на элемент списка
[[vk::binding(4, 0)]] SamplerCubeArray<float4> gCube;

uint pcg(inout uint s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
float urand(inout uint s) { return (pcg(s) >> 8) * (1.0 / 16777216.0); }
float2 gauss2(inout uint s)
{
    float u1 = max(urand(s), 1e-7), u2 = urand(s);
    float r = sqrt(-2 * log(u1));
    return r * float2(cos(6.2831853 * u2), sin(6.2831853 * u2));
}

#ifdef FB_CHUNK
// Порции по FB_CHUNK сэмплов: поток на (пиксель, порция), частичные суммы — в gPartial, затем modea_fbreduce
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> gPartial;
#endif

[shader("compute")] [numthreads(64, 1, 1)]
void modea_fallback(uint3 id : SV_DispatchThreadID)
{
    Cam c = load_cam(gCam);
    uint4 fb = gFb[0];
#ifdef FB_CHUNK
    uint nchunk = (fb.y + FB_CHUNK - 1) / FB_CHUNK;
    uint e = id.x / nchunk, chunk = id.x % nchunk;
    if (e >= fb.x) return;
    uint s0 = chunk * FB_CHUNK, s1 = min(s0 + FB_CHUNK, fb.y);
#else
    uint e = id.x;
    if (e >= fb.x) return;
    uint s0 = 0, s1 = fb.y;
#endif
    uint i = gPix[e], W = uint(c.width);
    float2 uv0 = pixel_uv(c, i % W, i / W);
    float dpix = 2 * c.tf / c.width;
    float alpha = pc.alpha < 0 ? ALPHA : pc.alpha;
    uint seed = (i * 9781u + fb.z * 6271u + s0 * 7919u) ^ 0x9e3779b9u;
    float4 acc = float4(0);
    for (uint smp = s0; smp < s1; smp++)
    {
        float2 jit = gauss2(seed) * c.sigma_pix * dpix;
        float2 ab = c.sigma_ap > 0 ? gauss2(seed) * c.sigma_ap : float2(0);
        float2 uv = uv0 + jit;
        float4 D = c.f + uv.x * c.r + uv.y * c.u;
        float4 d0 = D / length(D);
        float4 a = ab.x * c.r + ab.y * c.u;
        float4 target = c.focus / dot(d0, c.f) * d0 - a;
        // M16: точка линзы → M (Ньютон), направление → касательная плоскость
        float4 x = c.x + a;
        [ForceUnroll] for (int it = 0; it < 3; it++) { Eval e3 = rhs7(x, target, alpha); x -= e3.f * e3.g / dot(e3.g, e3.g); }
        Eval ev = rhs7(x, target, alpha);
        float4 n = ev.g / length(ev.g);
        float4 vt = target - dot(target, n) * n;
        RayState st;
        st.x = x; st.v = vt / length(vt);
        st.s = 0; st.h = 0; st.nacc = 0; st.nrej = 0; st.npass = 0; st.status = RUNNING;
        trace_ray(st, pc);
        if (st.status != DONE_ESCAPE) { acc += float4(1, 0, 1, 1); continue; }
        float3 d = exit_value(st.x, st.v);
        acc += gCube.SampleLevel(float4(d, st.x.w >= 0 ? 0 : 1), 0);
    }
#ifdef FB_CHUNK
    gPartial[id.x] = acc;
#elif defined(FB_SCATTER)
    gColor[i] = acc / float(fb.y);           // сразу в пиксель кадра (wavefront)
#else
    gColor[e] = acc / float(fb.y);
#endif
}

#ifdef FB_CHUNK
// Сумма порций пикселя → цвет кадра
[shader("compute")] [numthreads(64, 1, 1)]
void modea_fbreduce(uint3 id : SV_DispatchThreadID)
{
    uint4 fb = gFb[0];
    uint e = id.x;
    if (e >= fb.x) return;
    uint nchunk = (fb.y + FB_CHUNK - 1) / FB_CHUNK;
    float4 acc = float4(0);
    for (uint k = 0; k < nchunk; k++) acc += gPartial[e * nchunk + k];
    gColor[gPix[e]] = acc / float(fb.y);
}
#endif
#endif
