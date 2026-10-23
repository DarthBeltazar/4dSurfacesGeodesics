// Jet-числа (M4, M5, M23, M24) и операции M6 — порт prototype/geo4d/jet.py во float32.
// SDF пишутся один раз как generic-функции над IReal и работают на Real, Jet7, JetHVP, Jet15, Dual4.

interface IReal
{
    static This cst(float c);
    This add(This o);
    This mul(This o);
    This neg();
    This scale(float k);
    // унарная функция по значению, первой и второй производной в точке value() (M4)
    This unary(float g0, float g1, float g2);
    float value();
}

__generic<T : IReal> T operator+(T a, T b) { return a.add(b); }
__generic<T : IReal> T operator-(T a, T b) { return a.add(b.neg()); }
__generic<T : IReal> T operator*(T a, T b) { return a.mul(b); }
__generic<T : IReal> T operator-(T a) { return a.neg(); }
__generic<T : IReal> T operator*(float k, T b) { return b.scale(k); }
__generic<T : IReal> T operator*(T a, float k) { return a.scale(k); }
__generic<T : IReal> T operator+(T a, float k) { return a.add(T.cst(k)); }
__generic<T : IReal> T operator+(float k, T a) { return a.add(T.cst(k)); }
__generic<T : IReal> T operator-(T a, float k) { return a.add(T.cst(-k)); }
__generic<T : IReal> T operator-(float k, T a) { return a.neg().add(T.cst(k)); }

__generic<T : IReal> T jrecip(T u) { float a = u.value(); return u.unary(1.0 / a, -1.0 / (a * a), 2.0 / (a * a * a)); }
__generic<T : IReal> T operator/(T a, T b) { return a.mul(jrecip(b)); }
__generic<T : IReal> T operator/(T a, float k) { return a.scale(1.0 / k); }
__generic<T : IReal> T operator/(float k, T b) { return jrecip(b).scale(k); }

__generic<T : IReal> T jsqrt(T u) { float a = u.value(); float s = sqrt(a); return u.unary(s, 0.5 / s, -0.25 / (a * s)); }
__generic<T : IReal> T jexp(T u) { float e = exp(u.value()); return u.unary(e, e, e); }
__generic<T : IReal> T jlog(T u) { float a = u.value(); return u.unary(log(a), 1.0 / a, -1.0 / (a * a)); }
__generic<T : IReal> T jcosh(T u) { float a = u.value(); return u.unary(cosh(a), sinh(a), cosh(a)); }
__generic<T : IReal> T jsinh(T u) { float a = u.value(); return u.unary(sinh(a), cosh(a), sinh(a)); }
__generic<T : IReal> T jcos(T u) { float a = u.value(); return u.unary(cos(a), -sin(a), -cos(a)); }
__generic<T : IReal> T jsin(T u) { float a = u.value(); return u.unary(sin(a), cos(a), -sin(a)); }

// Выбор ветви по значению; обе ветви — одна гладкая функция (M6)
__generic<T : IReal> T jselect(bool c, T a, T b) { return c ? a : b; }

// M6: защищённая длина √(max(|u|², ε²)), ε = 1e-18 во float32
static const float LENGTH_EPS2 = 1e-36;
__generic<T : IReal> T jlength2(T a, T b) { T q = a * a + b * b; return jsqrt(jselect(q.value() < LENGTH_EPS2, T.cst(LENGTH_EPS2), q)); }
__generic<T : IReal> T jlength3(T a, T b, T c) { T q = a * a + b * b + c * c; return jsqrt(jselect(q.value() < LENGTH_EPS2, T.cst(LENGTH_EPS2), q)); }
__generic<T : IReal> T jlength4(T a, T b, T c, T d) { T q = a * a + b * b + c * c + d * d; return jsqrt(jselect(q.value() < LENGTH_EPS2, T.cst(LENGTH_EPS2), q)); }

// M6: C∞ гладкий минимум в устойчивой форме
__generic<T : IReal> T jsmin(T a, T b, float k)
{
    bool lt = a.value() < b.value();
    T lo = jselect(lt, a, b), hi = jselect(lt, b, a);
    return lo - k * jlog(1.0 + jexp((lo - hi) / k));
}

// M24: (cos√y, s = sin√y/√y, s′, s″); ряды при y < 1e-2 (во float32 порог выше, чем в прототипе)
float4 sqrt_family(float y)
{
    if (y < 1e-2)
        return float4(1 - y / 2 + y * y / 24 - y * y * y / 720, 1 - y / 6 + y * y / 120 - y * y * y / 5040,
                      -1.0 / 6 + y / 60 - y * y / 1680, 1.0 / 60 - y / 840 + y * y / 30240);
    float r = sqrt(y), c = cos(r), s = sin(r) / r;
    float s1 = (c - s) / (2 * y);
    float s2 = (-0.5 * s - s1) / (2 * y) - s1 / y;
    return float4(c, s, s1, s2);
}
__generic<T : IReal> T jcos_sqrt(T u) { float4 f = sqrt_family(u.value()); return u.unary(f.x, -0.5 * f.y, -0.5 * f.z); }
__generic<T : IReal> T jsinc_sqrt(T u) { float4 f = sqrt_family(u.value()); return u.unary(f.y, f.z, f.w); }

// ---- типы -------------------------------------------------------------------------------------

struct Real : IReal
{
    float a;
    __init(float x) { a = x; }
    static Real cst(float c) { return Real(c); }
    Real add(Real o) { return Real(a + o.a); }
    Real mul(Real o) { return Real(a * o.a); }
    Real neg() { return Real(-a); }
    Real scale(float k) { return Real(a * k); }
    Real unary(float g0, float g1, float g2) { return Real(g0); }
    float value() { return a; }
}

// M4: a + b t + c t² + Σ dᵢsᵢ
struct Jet7 : IReal
{
    float a, b, c;
    float4 d;
    __init(float a_, float b_, float c_, float4 d_) { a = a_; b = b_; c = c_; d = d_; }
    static Jet7 cst(float k) { return Jet7(k, 0, 0, float4(0)); }
    Jet7 add(Jet7 o) { return Jet7(a + o.a, b + o.b, c + o.c, d + o.d); }
    Jet7 mul(Jet7 o) { return Jet7(a * o.a, a * o.b + b * o.a, a * o.c + b * o.b + c * o.a, a * o.d + d * o.a); }
    Jet7 neg() { return Jet7(-a, -b, -c, -d); }
    Jet7 scale(float k) { return Jet7(a * k, b * k, c * k, d * k); }
    Jet7 unary(float g0, float g1, float g2) { return Jet7(g0, g1 * b, g1 * c + 0.5 * g2 * b * b, g1 * d); }
    float value() { return a; }
}

// M23: a + b t + Σ dᵢsᵢ + Σ mᵢ t sᵢ  → f, ∇f·w, ∇f, H·w
struct JetHVP : IReal
{
    float a, b;
    float4 d, m;
    __init(float a_, float b_, float4 d_, float4 m_) { a = a_; b = b_; d = d_; m = m_; }
    static JetHVP cst(float k) { return JetHVP(k, 0, float4(0), float4(0)); }
    JetHVP add(JetHVP o) { return JetHVP(a + o.a, b + o.b, d + o.d, m + o.m); }
    JetHVP mul(JetHVP o) { return JetHVP(a * o.a, a * o.b + b * o.a, a * o.d + d * o.a, a * o.m + m * o.a + b * o.d + d * o.b); }
    JetHVP neg() { return JetHVP(-a, -b, -d, -m); }
    JetHVP scale(float k) { return JetHVP(a * k, b * k, d * k, m * k); }
    JetHVP unary(float g0, float g1, float g2) { return JetHVP(g0, g1 * b, g1 * d, g1 * m + g2 * b * d); }
    float value() { return a; }
}

// M5: a + gᵀε + ½εᵀHε, H хранится полностью (симметрия не используется — вариант для замера регистров)
struct Jet15 : IReal
{
    float a;
    float4 g;
    float4x4 H;
    __init(float a_, float4 g_, float4x4 H_) { a = a_; g = g_; H = H_; }
    static Jet15 cst(float k) { return Jet15(k, float4(0), float4x4(0)); }
    Jet15 add(Jet15 o) { return Jet15(a + o.a, g + o.g, H + o.H); }
    Jet15 mul(Jet15 o)
    {
        float4x4 gg = float4x4(g.x * o.g, g.y * o.g, g.z * o.g, g.w * o.g);
        return Jet15(a * o.a, a * o.g + o.a * g, a * o.H + o.a * H + gg + transpose(gg));
    }
    Jet15 neg() { return Jet15(-a, -g, -H); }
    Jet15 scale(float k) { return Jet15(a * k, g * k, H * k); }
    Jet15 unary(float f0, float f1, float f2) { return Jet15(f0, f1 * g, f1 * H + f2 * float4x4(g.x * g, g.y * g, g.z * g, g.w * g)); }
    float value() { return a; }
}

// M24: дуальные числа по 4 направлениям (4 поля Якоби)
struct Dual4 : IReal
{
    float a;
    float4 b;
    __init(float a_, float4 b_) { a = a_; b = b_; }
    static Dual4 cst(float k) { return Dual4(k, float4(0)); }
    Dual4 add(Dual4 o) { return Dual4(a + o.a, b + o.b); }
    Dual4 mul(Dual4 o) { return Dual4(a * o.a, a * o.b + b * o.a); }
    Dual4 neg() { return Dual4(-a, -b); }
    Dual4 scale(float k) { return Dual4(a * k, b * k); }
    Dual4 unary(float g0, float g1, float g2) { return Dual4(g0, g1 * b); }
    float value() { return a; }
}
