"""U1 на GPU (DD-004 п. 10): Jet7, HVP-jet и Jet15 во float32 против jax.grad/jax.hessian во float64."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d import gpu
from geo4d.shapes import SHAPES, sample, two_compact_wormholes

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")]

ALL = dict(SHAPES, two_compact_wh=two_compact_wormholes())
N = 1000


@pytest.mark.parametrize("name", sorted(ALL))
def test_gpu_jets(name):
    sh = ALL[name]
    P, _ = sample(sh, N, seed=1)
    rng = np.random.default_rng(101)
    P = P + 0.05 * sh.scale * rng.normal(size=P.shape)
    W = rng.normal(size=P.shape)
    P32, W32 = gpu.f32(P), gpu.f32(W)
    spv = gpu.compile_shader("jets_u1.slang", "jets_u1", {"SCENE": gpu.SCENE_IDS[name]})
    out, rep = gpu.run(spv, "jets_u1", [("in", P32), ("in", W32), ("out", ("f4", (N * 8, 4)))],
                       push=[("u32", N)], groups=gpu.groups_for(N))
    o = out[2].reshape(N, 8, 4).astype(np.float64)
    # эталон — в тех же (округлённых до float32) точках, во float64
    Pd, Wd = jnp.asarray(P32, jnp.float64), jnp.asarray(W32, jnp.float64)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    Hw = np.einsum("nij,nj->ni", H, np.asarray(Wd))
    wHw = np.einsum("ni,ni->n", np.asarray(Wd), Hw)
    # масштабы: относительная ошибка float32 ~ 1e-6·(обусловленность выражения)
    rel = lambda got, ref, sc: np.max(np.abs(got - ref) / np.maximum(sc, np.abs(ref)))
    sc = np.max(np.abs(H), axis=(1, 2))[:, None] + 1.0
    assert rel(o[:, 0, 0], f, 1.0) < 2e-5
    assert rel(o[:, 1], G, 1.0) < 2e-5
    assert rel(o[:, 0, 1], np.sum(G * np.asarray(Wd), 1), 1.0) < 2e-5
    assert rel(o[:, 0, 2], wHw, np.linalg.norm(np.asarray(Wd), axis=1) ** 2 * sc[:, 0]) < 2e-5
    assert rel(o[:, 2], Hw, sc * np.linalg.norm(np.asarray(Wd), axis=1)[:, None]) < 2e-5
    assert rel(o[:, 4:].reshape(N, 4, 4), H, sc[:, :, None]) < 2e-5
