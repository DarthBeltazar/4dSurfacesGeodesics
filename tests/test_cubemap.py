"""M19: кубмапа — раскладка, выборка, фильтр по эллипсу против прямого гауссова интеграла, .hdr."""
import numpy as np
import jax.numpy as jnp
import pytest

from geo4d.cubemap import (dir_to_face, face_to_dir, bake, build_mips, sample_bilinear0, sample_ellipse,
                           write_hdr, read_hdr)

N = 128


def smooth_fn(d, sheet):
    v = jnp.sin(6 * d[..., 0]) * jnp.cos(4 * d[..., 1]) + 0.5 * jnp.sin(9 * d[..., 2] + 1) + 1.5
    return jnp.stack([v, v * 0.5 + sheet * 0.1, jnp.cos(3 * d[..., 0] * d[..., 1]) + 1.2], -1)


@pytest.fixture(scope="module")
def mips():
    return build_mips(bake(smooth_fn, N))


def rand_dirs(n, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_face_roundtrip():
    d = rand_dirs(5000)
    f, s, t = (np.asarray(a) for a in dir_to_face(jnp.asarray(d)))
    back = np.stack([face_to_dir(int(fi), si, ti) for fi, si, ti in zip(f, s, t)])
    assert np.max(np.abs(back - d)) < 1e-12
    assert np.all((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1))
    assert set(f.tolist()) == set(range(6))


def test_bilinear_reproduces_function(mips):
    d = rand_dirs(5000, 1)
    got = np.asarray(sample_bilinear0(mips, jnp.zeros(len(d), int), jnp.asarray(d)))
    ref = np.asarray(smooth_fn(jnp.asarray(d), jnp.ones(len(d))))
    assert np.max(np.abs(got - ref)) < 5e-3        # билинейная интерполяция, h² ~ (2/N)²·|f″|


def tangent_axes(d, rng):
    a = np.random.default_rng(rng.integers(1 << 30) + 7777).normal(size=d.shape)   # не коррелировать с rand_dirs
    a -= np.sum(a * d, 1, keepdims=True) * d
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(d, a)
    return a, b


def gauss_hermite_ref(mips, d, a1, s1, a2, s2, n=24):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    acc = 0.0
    for xi, wi in zip(x, w):
        for yj, wj in zip(x, w):
            dd = d + (xi * s1)[:, None] * a1 + (yj * s2)[:, None] * a2
            dd /= np.linalg.norm(dd, axis=1, keepdims=True)
            acc = acc + wi * wj * np.asarray(sample_bilinear0(mips, jnp.zeros(len(d), int), jnp.asarray(dd)))
    return acc


@pytest.mark.parametrize("s_major_tex,aniso,p95,emax", [(0.3, 1, 0.03, 0.08), (2.0, 1, 0.03, 0.08),
                                                         (6.0, 4, 0.03, 0.08), (10.0, 16, 0.03, 0.08),
                                                         (20.0, 8, 0.03, 0.08),
                                                         (40.0, 16, 0.07, 0.3)])   # зажим анизотропии
def test_ellipse_filter_vs_gaussian_integral(mips, s_major_tex, aniso, p95, emax):
    rng = np.random.default_rng(int(s_major_tex * 10) + aniso)
    d = rand_dirs(400, 2)
    a1, a2 = tangent_axes(d, rng)
    texel = 2.0 / N                                   # угловой размер текселя в центре грани
    s1 = np.full(len(d), s_major_tex * texel)
    s2 = s1 / aniso
    got, L, n = sample_ellipse(mips, jnp.zeros(len(d), int), jnp.asarray(d), jnp.asarray(a1), jnp.asarray(s1),
                               jnp.asarray(a2), jnp.asarray(s2))
    ref = gauss_hermite_ref(mips, d, a1, s1, a2, s2)
    err = np.abs(np.asarray(got) - ref)
    # амплитуда сигнала ~1; допуски — по измерению (docs/reports/stage-2.md)
    assert np.percentile(err, 95) < p95 and err.max() < emax, (np.percentile(err, 95), err.max())


def test_constant_cubemap_filter():
    m = build_mips(bake(lambda d, sh: jnp.ones(d.shape) * 0.7, 32))
    d = rand_dirs(200, 3)
    rng = np.random.default_rng(3)
    a1, a2 = tangent_axes(d, rng)
    got, _, _ = sample_ellipse(m, jnp.zeros(len(d), int), jnp.asarray(d), jnp.asarray(a1), jnp.full(len(d), 0.3),
                               jnp.asarray(a2), jnp.full(len(d), 0.02))
    assert np.max(np.abs(np.asarray(got) - 0.7)) < 1e-6


def test_hdr_roundtrip(tmp_path):
    img = np.random.default_rng(0).uniform(0, 4, (16, 24, 3)).astype(np.float32)
    write_hdr(tmp_path / "x.hdr", img)
    back = read_hdr(tmp_path / "x.hdr")
    assert back.shape == img.shape
    assert np.max(np.abs(back - img) / np.maximum(img.max(-1, keepdims=True), 1e-6)) < 1 / 128
