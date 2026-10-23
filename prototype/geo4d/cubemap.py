"""Кубмапы фона (M19): раскладка граней, запекание, мипы, билинейная/трилинейная выборка, анизотропный
гауссов фильтр по эллипсу следа (режим A), чтение/запись Radiance .hdr."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

MAX_TAPS = 16

# M19: для грани k — (индекс главной оси, знак, (ось sc, знак), (ось tc, знак))
_FACES = [(0, +1, (2, -1), (1, -1)), (0, -1, (2, +1), (1, -1)),
          (1, +1, (0, +1), (2, +1)), (1, -1, (0, +1), (2, -1)),
          (2, +1, (0, +1), (1, -1)), (2, -1, (0, -1), (1, -1))]


def dir_to_face(d):
    """d (..., 3) -> (face (...), s (...), t (...)), s, t ∈ [0, 1]."""
    a = jnp.abs(d)
    ax = jnp.argmax(a, axis=-1)
    comp = jnp.take_along_axis(d, ax[..., None], -1)[..., 0]
    face = 2 * ax + (comp < 0)
    ma = jnp.abs(comp)
    sc = jnp.zeros_like(ma)
    tc = jnp.zeros_like(ma)
    for k, (_, _, (si, ss), (ti, ts)) in enumerate(_FACES):
        sel = face == k
        sc = jnp.where(sel, ss * d[..., si], sc)
        tc = jnp.where(sel, ts * d[..., ti], tc)
    return face, 0.5 * (sc / ma + 1), 0.5 * (tc / ma + 1)


def face_to_dir(face, s, t):
    """Обратное отображение (numpy): face (int), s, t массивы -> d (..., 3) нормированные."""
    ai, sg, (si, ss), (ti, ts) = _FACES[face]
    d = np.zeros(np.shape(s) + (3,))
    d[..., ai] = sg
    d[..., si] = ss * (2 * np.asarray(s) - 1)
    d[..., ti] = ts * (2 * np.asarray(t) - 1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def _centers(N, border=1):
    """Координаты центров текселей с рамкой: i = −1..N (рамка — продолжение плоскости грани)."""
    c = (np.arange(-border, N + border) + 0.5) / N
    return np.meshgrid(c, c)                     # [j, i]: t по строкам, s по столбцам


def bake(fn, N, sheets=(+1, -1)):
    """Кубмапа (len(sheets), 6, N+2, N+2, 3) float32 с рамкой в 1 тексель: fn(d (..., 3), sheet (...)) -> rgb
    в центрах текселей. Рамка запекается точно (направления продолжения плоскости грани) — бесшовная выборка."""
    S, T = _centers(N)
    out = np.zeros((len(sheets), 6, N + 2, N + 2, 3), np.float32)
    for si, sh in enumerate(sheets):
        for f in range(6):
            d = face_to_dir(f, S, T)
            out[si, f] = np.asarray(fn(jnp.asarray(d), jnp.full(d.shape[:-1], sh)))
    return out


def _fill_border(c):
    """Рамка уровня мипа — билинейная выборка внутренних текселей соседних граней по направлению."""
    N = c.shape[2] - 2
    S, T = _centers(N)
    lev = jnp.asarray(c)
    ring = np.ones((N + 2, N + 2), bool)
    ring[1:-1, 1:-1] = False
    for f in range(6):
        d = face_to_dir(f, S[ring], T[ring])
        face, s, t = dir_to_face(jnp.asarray(d))
        for si in range(c.shape[0]):
            c[si, f][ring] = np.asarray(_bilinear(lev, jnp.full(face.shape, si), face, s, t, clamp_inner=True))
    return c


def build_mips(cube):
    """Цепочка мипов: бокс 2×2 внутри грани, рамка — с соседних граней; до 1×1 (+рамка)."""
    levels = [jnp.asarray(cube)]
    c = np.asarray(cube)
    while c.shape[2] - 2 > 1:
        inner = c[:, :, 1:-1, 1:-1]
        inner = 0.25 * (inner[:, :, 0::2, 0::2] + inner[:, :, 1::2, 0::2] + inner[:, :, 0::2, 1::2]
                        + inner[:, :, 1::2, 1::2])
        n = inner.shape[2]
        c = np.zeros(inner.shape[:2] + (n + 2, n + 2, 3), np.float32)
        c[:, :, 1:-1, 1:-1] = inner
        c = _fill_border(c)
        levels.append(jnp.asarray(c))
    return levels


def _bilinear(level, sheet_idx, face, s, t, clamp_inner=False):
    """Билинейная выборка уровня с рамкой (N+2). clamp_inner — только внутренние тексели (для заполнения рамки)."""
    N = level.shape[2] - 2
    lo, hi = (1.0, N + 0.0) if clamp_inner else (0.0, N + 1.0)
    x = jnp.clip(s * N + 0.5, lo, hi)            # +1 рамка −0.5 центр
    y = jnp.clip(t * N + 0.5, lo, hi)
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.minimum(x0 + 1, int(hi))
    y1 = jnp.minimum(y0 + 1, int(hi))
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    g = lambda yy, xx: level[sheet_idx, face, yy, xx]
    return ((1 - fy) * ((1 - fx) * g(y0, x0) + fx * g(y0, x1)) + fy * ((1 - fx) * g(y1, x0) + fx * g(y1, x1)))


def sample_level(mips, sheet_idx, d, level):
    """Трилинейная выборка: level — дробный уровень мипа (...)."""
    face, s, t = dir_to_face(d)
    L = jnp.clip(level, 0.0, len(mips) - 1.0)
    l0 = jnp.floor(L)
    fr = (L - l0)[..., None]
    out = 0.0
    for k, lev in enumerate(mips):
        w = jnp.where((l0 == k)[..., None], 1 - fr, 0.0) + jnp.where((l0 + 1 == k)[..., None], fr, 0.0)
        out = out + w * _bilinear(lev, sheet_idx, face, s, t)
    return out


def sample_bilinear0(mips, sheet_idx, d):
    face, s, t = dir_to_face(d)
    return _bilinear(mips[0], sheet_idx, face, s, t)


def texels_per_radian(d, axis, N, delta=1e-5):
    """Масштаб отображения направление → тексели уровня 0 вдоль касательной оси (численно, с той же грани)."""
    f0, s0, t0 = dir_to_face(d)
    out = []
    for sgn in (+1.0, -1.0):
        d1 = d + sgn * delta * axis
        d1 = d1 / jnp.linalg.norm(d1, axis=-1, keepdims=True)
        f1, s1, t1 = dir_to_face(d1)
        dist = jnp.sqrt((s1 - s0) ** 2 + (t1 - t0) ** 2) * N / delta
        out.append(jnp.where(f1 == f0, dist, jnp.nan))
    return jnp.where(jnp.isnan(out[0]), out[1], out[0])


def sample_ellipse(mips, sheet_idx, d, a_major, sig_major, a_minor, sig_minor):
    """M19: анизотропный гауссов фильтр по эллипсу следа. Сигмы — в радианах, оси — единичные ⊥ d.

    Уровень L выбирается так, чтобы размытие уровня (≈ 2^L/2 текселя) покрывало малую ось и чтобы
    ≤ MAX_TAPS отсчётов с шагом ≤ 2·размытия покрыли ±2σ_eff вдоль большой оси (шаг выбран измерением,
    см. docs/reports/stage-2.md). При анизотропии ≳ 8 и очень больших следах малая ось перемывается —
    как при аппаратном зажиме анизотропии."""
    N = mips[0].shape[2] - 2
    tpr = texels_per_radian(d, a_major, N)
    s_min = sig_minor * texels_per_radian(d, a_minor, N)          # σ в текселях уровня 0
    s_maj = sig_major * tpr
    L = jnp.log2(jnp.maximum(jnp.maximum(2 * s_min, 4 * s_maj / (MAX_TAPS - 1)), 1.0))
    blur = 0.5 * 2.0 ** L
    s_eff = jnp.sqrt(jnp.maximum(s_maj ** 2 - blur ** 2, 0.0))     # недостающая часть большой оси
    n = jnp.clip(jnp.ceil(2 * s_eff / blur) + 1, 1, MAX_TAPS)
    k = jnp.arange(MAX_TAPS)
    nn = n[..., None]
    u = jnp.where(nn > 1, (k - (nn - 1) / 2) / jnp.maximum(nn - 1, 1) * 4.0, 0.0)   # ∈ [−2, 2] в σ_eff
    w = jnp.where(k < nn, jnp.exp(-0.5 * u ** 2), 0.0)
    w = w / jnp.sum(w, -1, keepdims=True)
    tk = u * (s_eff / tpr)[..., None]                                # радианы вдоль большой оси
    dirs = d[..., None, :] + tk[..., None] * a_major[..., None, :]
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
    cols = sample_level(mips, sheet_idx[..., None], dirs, jnp.broadcast_to(L[..., None], tk.shape))
    return jnp.sum(w[..., None] * cols, -2), L, n


# ---- Radiance .hdr (RGBE) ----------------------------------------------------------------------

def write_hdr(path, img):
    """Несжатый RGBE (для тестов и экспорта). img: (H, W, 3) float."""
    img = np.asarray(img, np.float64)
    H, W, _ = img.shape
    m = img.max(-1)
    e = np.where(m > 1e-32, np.floor(np.log2(np.maximum(m, 1e-32))) + 1, 0)
    sc = np.where(m > 1e-32, 256.0 / 2.0 ** e, 0)
    rgbe = np.zeros((H, W, 4), np.uint8)
    rgbe[..., :3] = np.clip(img * sc[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {H} +X {W}\n".encode())
        fh.write(rgbe.tobytes())


def read_hdr(path):
    """Чтение Radiance .hdr (несжатый и новый RLE). Возвращает (H, W, 3) float32."""
    data = open(path, "rb").read()
    hdr_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", hdr_end)
    parts = data[hdr_end:line_end].split()
    H, W = int(parts[1]), int(parts[3])
    p = line_end + 1
    out = np.zeros((H, W, 4), np.uint8)
    for y in range(H):
        if W >= 8 and W < 32768 and data[p] == 2 and data[p + 1] == 2 and (data[p + 2] << 8 | data[p + 3]) == W:
            p += 4
            for c in range(4):
                x = 0
                while x < W:
                    cnt = data[p]
                    p += 1
                    if cnt > 128:
                        cnt -= 128
                        out[y, x:x + cnt, c] = data[p]
                        p += 1
                    else:
                        out[y, x:x + cnt, c] = np.frombuffer(data[p:p + cnt], np.uint8)
                        p += cnt
                    x += cnt
        else:
            out[y] = np.frombuffer(data[p:p + 4 * W], np.uint8).reshape(W, 4)
            p += 4 * W
    e = out[..., 3].astype(np.int32)
    sc = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (out[..., :3] * sc[..., None]).astype(np.float32)


def equirect_to_fn(img):
    """Функция фона по равнопромежуточной панораме (ближайший тексель), для запекания в кубмапу."""
    H, W, _ = img.shape
    im = jnp.asarray(img)

    def fn(d, sheet):
        lon = jnp.arctan2(d[..., 1], d[..., 0])
        lat = jnp.arcsin(jnp.clip(d[..., 2], -1, 1))
        x = jnp.clip(((lon / (2 * np.pi) + 0.5) * W).astype(jnp.int32), 0, W - 1)
        y = jnp.clip(((0.5 - lat / np.pi) * H).astype(jnp.int32), 0, H - 1)
        return im[y, x]
    return fn


def sample_gauss_exact(mips, sheet_idx, d, a1, s1, a2, s2, spacing=0.7, nmax=128, chunk=2048):
    """Эталонный гауссов фильтр по эллипсу: плотная сетка по ±3σ с шагом ≤ spacing текселя уровня 0
    (билинейно), гауссовы веса. Для разложения ошибки режима A на «след Якоби» и «фильтр мипами».
    Квадратура Гаусса–Эрмита 8×8 для этого непригодна: на следах ≳ 5 текселей пропускает тонкие детали."""
    N = mips[0].shape[2] - 2
    d, a1, s1, a2, s2 = (np.asarray(t) for t in (d, a1, s1, a2, s2))
    t1 = s1 * np.asarray(texels_per_radian(jnp.asarray(d), jnp.asarray(a1), N))
    t2 = s2 * np.asarray(texels_per_radian(jnp.asarray(d), jnp.asarray(a2), N))
    out = np.zeros(d.shape)

    @jax.jit
    def run(d, a1, s1, a2, s2, sheet, u1, u2):
        w = jnp.exp(-0.5 * (u1[:, None] ** 2 + u2[None, :] ** 2))
        w = w / jnp.sum(w)
        dd = (d[:, None, None, :] + (u1[None, :, None, None] * s1[:, None, None, None]) * a1[:, None, None, :]
              + (u2[None, None, :, None] * s2[:, None, None, None]) * a2[:, None, None, :])
        dd = dd / jnp.linalg.norm(dd, axis=-1, keepdims=True)
        c = sample_bilinear0(mips, jnp.broadcast_to(sheet[:, None, None], dd.shape[:-1]), dd)
        return jnp.einsum("ij,nijc->nc", w, c)

    for i in range(0, len(d), chunk):
        sl = slice(i, i + chunk)
        n1 = int(np.clip(np.ceil(6 * np.max(t1[sl]) / spacing), 1, nmax)) | 1
        n2 = int(np.clip(np.ceil(6 * np.max(t2[sl]) / spacing), 1, nmax)) | 1
        u1 = np.linspace(-3, 3, n1) if n1 > 1 else np.zeros(1)
        u2 = np.linspace(-3, 3, n2) if n2 > 1 else np.zeros(1)
        out[sl] = np.asarray(run(*(jnp.asarray(t[sl]) for t in (d, a1, s1, a2, s2, np.asarray(sheet_idx))),
                                 jnp.asarray(u1), jnp.asarray(u2)))
    return out
