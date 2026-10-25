"""Проба интерпретатора байткода (concept-standalone-nodes.md, п. 3): программа geo4d.bytecode для двух горловин
(M12) на CPU совпадает с формулой фигуры, а на GPU (scenes.slang SCENE_DYNAMIC + MF_PROG) даёт Jet7/HVP/Jet15 с той же
точностью, что зашитая сцена (U1, test_gpu_u1)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from geo4d import bytecode as B, gpu, gpu_pt as G
from geo4d.shapes import sample, two_compact_wormholes

N = 1000


def _points():
    sh = two_compact_wormholes()
    P, _ = sample(sh, N, seed=1)
    rng = np.random.default_rng(101)
    P = P + 0.05 * sh.scale * rng.normal(size=P.shape)
    return sh, gpu.f32(P), gpu.f32(rng.normal(size=P.shape))


def test_program_matches_shape_cpu():
    sh, P, _ = _points()
    R = B.two_compact_program()
    fp = lambda p: B.evaluate(R, p[0], p[1], p[2], p[3])
    Pd = jnp.asarray(P, jnp.float64)
    for d in (lambda f: f, jax.grad, jax.hessian):
        np.testing.assert_array_equal(np.asarray(jax.vmap(d(fp))(Pd)), np.asarray(jax.vmap(d(sh.f_vec))(Pd)))


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
def test_program_gpu_jets():
    sh, P32, W32 = _points()
    prog = B.two_compact_program()
    cam = np.zeros((G.MF_BASE + G.MF_ROWS + len(prog), 4), np.float32)
    cam[G.MF_BASE, 0] = gpu.SCENE_IDS["two_compact_wh"]
    cam[G.MF_BASE + G.MF_ROWS:] = prog
    spv = gpu.compile_shader("prog_u1.slang", "prog_u1", {"SCENE": gpu.SCENE_IDS["dynamic"], "MF_PROG": 1})
    out, _ = gpu.run(spv, "prog_u1", [("in", cam), ("in", P32), ("in", W32), ("out", ("f4", (N * 8, 4)))],
                     push=[("u32", N)], groups=gpu.groups_for(N))
    o = out[3].reshape(N, 8, 4).astype(np.float64)
    spv0 = gpu.compile_shader("jets_u1.slang", "jets_u1", {"SCENE": gpu.SCENE_IDS["two_compact_wh"]})
    out0, _ = gpu.run(spv0, "jets_u1", [("in", P32), ("in", W32), ("out", ("f4", (N * 8, 4)))],
                      push=[("u32", N)], groups=gpu.groups_for(N))
    o0 = out0[2].reshape(N, 8, 4).astype(np.float64)
    # эталон float64 в тех же точках — как в test_gpu_u1
    Pd, Wd = jnp.asarray(P32, jnp.float64), np.asarray(W32, np.float64)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    Gr = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    Hw = np.einsum("nij,nj->ni", H, Wd)
    sc = np.max(np.abs(H), axis=(1, 2))[:, None] + 1.0
    wn = np.linalg.norm(Wd, axis=1)
    rel = lambda got, ref, s: np.max(np.abs(got - ref) / np.maximum(s, np.abs(ref)))
    errs = {}
    for name, x in [("prog", o), ("native", o0)]:
        errs[name] = [rel(x[:, 0, 0], f, 1.0), rel(x[:, 1], Gr, 1.0), rel(x[:, 0, 1], np.sum(Gr * Wd, 1), 1.0),
                      rel(x[:, 0, 2], np.einsum("ni,ni->n", Wd, Hw), wn ** 2 * sc[:, 0]),
                      rel(x[:, 2], Hw, sc * wn[:, None]), rel(x[:, 4:].reshape(N, 4, 4), H, sc[:, :, None])]
    print(errs)
    assert max(errs["prog"]) < 2e-5
