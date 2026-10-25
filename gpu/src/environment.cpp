// Фон окна (DD-010) — см. environment.hpp.
#include "environment.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>

namespace geo4d {

namespace {

constexpr double kPi = 3.14159265358979323846;

// M19: для грани k — (главная ось, знак, (ось s, знак), (ось t, знак)) — geo4d.cubemap._FACES
struct Face { int ai, sg, si, ss, ti, ts; };
const Face kFaces[6] = {{0, +1, 2, -1, 1, -1}, {0, -1, 2, +1, 1, -1}, {1, +1, 0, +1, 2, +1},
                        {1, -1, 0, +1, 2, -1}, {2, +1, 0, +1, 1, -1}, {2, -1, 0, -1, 1, -1}};

// geo4d.render: цвета листов (нижний, верхний край градиента) и полюсов
const double kSheetBase[2][2][3] = {{{0.05, 0.08, 0.18}, {0.25, 0.35, 0.6}},      // лист +1: холодный
                                    {{0.2, 0.08, 0.03}, {0.65, 0.4, 0.2}}};       // лист −1: тёплый
const double kPoles[6][2][3] = {{{1, 0, 0}, {1.0, 0.15, 0.1}}, {{-1, 0, 0}, {0.1, 0.9, 0.9}},
                                {{0, 1, 0}, {0.15, 1.0, 0.15}}, {{0, -1, 0}, {1.0, 0.2, 1.0}},
                                {{0, 0, 1}, {1.0, 0.95, 0.2}}, {{0, 0, -1}, {0.2, 0.3, 1.0}}};

double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }

// jnp.remainder: остаток со знаком делителя
double remainder(double x, double y) {
    double r = std::fmod(x, y);
    if (r != 0 && ((r < 0) != (y < 0))) r += y;
    return r;
}

// geo4d.render.background(d, sheet): градиент по широте + сетка 15° + цветные полюса
std::array<double, 3> background(std::array<double, 3> d, int sheet_idx) {
    const double grid = 15.0, line_deg = 0.5, pole = 8.0;
    const double deg = 180 / kPi, rad = kPi / 180;
    double nd = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (auto& a : d) a = a / nd;
    double lat = std::asin(clip(d[2], -1, 1)) * deg;
    double lon = std::atan2(d[1], d[0]) * deg;
    double t = 0.5 + 0.5 * d[2];
    std::array<double, 3> out;
    for (int c = 0; c < 3; ++c) out[c] = (1 - t) * kSheetBase[sheet_idx][0][c] + t * kSheetBase[sheet_idx][1][c];
    double dl = std::abs(remainder(lat + grid / 2, grid) - grid / 2);
    double dn = std::abs(remainder(lon + grid / 2, grid) - grid / 2) * std::cos(lat * rad);
    double line = clip(1.0 - std::min(dl, dn) / line_deg, 0.0, 1.0);
    for (int c = 0; c < 3; ++c) out[c] = out[c] * (1 - line) + line * 0.9;
    for (auto& pc : kPoles) {
        double dp = d[0] * pc[0][0] + d[1] * pc[0][1] + d[2] * pc[0][2];
        double ang = std::acos(clip(dp, -1, 1)) * deg;
        double w = clip((pole - ang) / 1.0, 0.0, 1.0);
        for (int c = 0; c < 3; ++c) out[c] = out[c] * (1 - w) + w * 2.0 * pc[1][c];
    }
    return out;
}

// geo4d.cubemap.face_to_dir в центре текселя (i, j) уровня n
std::array<double, 3> face_dir(int face, int i, int j, int n) {
    const Face& F = kFaces[face];
    double s = (i + 0.5) / n, t = (j + 0.5) / n;
    std::array<double, 3> d{0, 0, 0};
    d[F.ai] = F.sg;
    d[F.si] = F.ss * (2 * s - 1);
    d[F.ti] = F.ts * (2 * t - 1);
    double l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    for (auto& a : d) a /= l;
    return d;
}

// попарная сумма numpy (pairwise_sum, PW_BLOCKSIZE = 128)
double pairwise_sum(const double* a, size_t n) {
    if (n < 8) {
        double r = 0;
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int k = 0; k < 8; ++k) r[k] = a[k];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int k = 0; k < 8; ++k) r[k] += a[i + k];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// geo4d.pathtrace.env_table: уровень max(0, L − 1 − 5) цепочки мипов (бокс 2×2 во float32), тексель ∝ яркость ·
// телесный угол, pmf и cdf по листу; упаковка geo4d.gpu_pt.env_buffer
void make_table(EnvMap& e) {
    int levels = 1;
    for (int m = e.n; m > 1; m /= 2) ++levels;
    int target = std::max(0, levels - 1 - 5);
    int n = e.n;
    std::vector<float> lev = e.cube;
    for (int k = 0; k < target; ++k) {
        int h = n / 2;
        std::vector<float> nx(size_t(2) * 6 * h * h * 3);
        for (int sf = 0; sf < 12; ++sf)
            for (int j = 0; j < h; ++j)
                for (int i = 0; i < h; ++i)
                    for (int c = 0; c < 3; ++c) {
                        auto at = [&](int jj, int ii) { return lev[((size_t(sf) * n + jj) * n + ii) * 3 + c]; };
                        float s = at(2 * j, 2 * i) + at(2 * j + 1, 2 * i);
                        s = s + at(2 * j, 2 * i + 1);
                        s = s + at(2 * j + 1, 2 * i + 1);
                        nx[((size_t(sf) * h + j) * h + i) * 3 + c] = 0.25f * s;
                    }
        lev.swap(nx);
        n = h;
    }
    const size_t per = size_t(6) * n * n;
    std::vector<double> pmf(2 * per);
    for (int sh = 0; sh < 2; ++sh)
        for (int f = 0; f < 6; ++f)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const float* px = &lev[(((size_t(sh) * 6 + f) * n + j) * n + i) * 3];
                    double lum = double(px[0]) * 0.2126 + double(px[1]) * 0.7152 + double(px[2]) * 0.0722;
                    double u = 2 * ((i + 0.5) / n) - 1, v = 2 * ((j + 0.5) / n) - 1;
                    double omega = 4.0 / double(n * n) / std::pow(1 + u * u + v * v, 1.5);
                    pmf[sh * per + (size_t(f) * n + j) * n + i] = std::max(lum, 0.0) * omega + 1e-30;
                }
    e.table_n = n;
    e.table.assign(4 * per, 0.0f);
    for (int sh = 0; sh < 2; ++sh) {
        double* p = &pmf[sh * per];
        double sum = 0.0 + pairwise_sum(p, per);
        double acc = 0;
        for (size_t k = 0; k < per; ++k) {
            p[k] = p[k] / sum;
            acc += p[k];
            e.table[sh * per + k] = float(acc);
            e.table[2 * per + sh * per + k] = float(p[k]);
        }
    }
}

}  // namespace

EnvMap procedural_env(int n) {
    EnvMap e;
    e.n = n;
    e.cube.assign(size_t(2) * 6 * n * n * 3, 0.0f);
    std::vector<std::thread> th;
    for (int sf = 0; sf < 12; ++sf)
        th.emplace_back([&e, n, sf] {
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    auto c = background(face_dir(sf % 6, i, j, n), sf / 6);
                    for (int k = 0; k < 3; ++k) e.cube[((size_t(sf) * n + j) * n + i) * 3 + k] = float(c[k]);
                }
        });
    for (auto& t : th) t.join();
    make_table(e);
    return e;
}

EnvMap constant_env(float value, int n) {
    EnvMap e;
    e.n = n;
    e.cube.assign(size_t(2) * 6 * n * n * 3, value);
    make_table(e);
    return e;
}

}  // namespace geo4d
