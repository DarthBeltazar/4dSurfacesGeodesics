"""DD-014: CSG многообразий в графе — ноды mf.union / mf.intersect / mf.subtract, mf.sphere (M38).

Эталон — независимые формы прототипа (shapes.plane_sphere_union, shapes.slab_minus_ball), не текст из C++:
- CPU (`geo4d_gpu eval-graph`) = эталон (значение, градиент, гессиан, float64);
- метаданные: листы, шар носителя R + 1.25k, ℓ_s = 0.5/κ_max (κ_max — по плотной выборке прототипа), ошибки загрузки
  (сфера не соединена, две основы, сферы без основы); одна сфера — замкнутое пространство;
- GPU: jet-производные и геодесические (в том числе сквозь шейку) против DOP853 и прототипа (≤ 4×)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import gpu
from geo4d.integrators import Config, make_integrator
from geo4d.reference import dop853
from geo4d.shapes import plane_sphere_union, sample, slab_minus_ball

from test_node_manifolds import _doc, _eval_graph, _header

pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")

R, D, K = 3.0, 0.5, 2.0
SLAB = dict(W=1.5, B=2.0, k=1.0)


def bubble_doc(R=R, d=D, k=K, flip=(0, 0), node="mf.union", objects=False):
    nodes = [{"id": "plane", "type": "mf.hyperplane"},
             {"id": "s", "type": "mf.sphere", "params": {"c": [0.0, 0.0, 0.0, R + d], "R": R}},
             {"id": "u", "type": node, "params": {"k": k, "flip_a": flip[0], "flip_b": flip[1]}}]
    links = [{"from": "plane.out", "to": "u.a"}, {"from": "s.out", "to": "u.b"}]
    return _doc(nodes, links, "u", objects=objects)


def slab_doc(W=SLAB["W"], B=SLAB["B"], k=SLAB["k"]):
    nodes = [{"id": "base", "type": "mf.two_sheets", "params": {"W": W}},
             {"id": "s", "type": "mf.sphere", "params": {"c": [0.0, 0.0, 0.0, 0.0], "R": B}},
             {"id": "u", "type": "mf.subtract", "params": {"k": k}}]
    links = [{"from": "base.out", "to": "u.a"}, {"from": "s.out", "to": "u.b"}]
    return _doc(nodes, links, "u", objects=False)


CASES = {"bubble": (lambda: plane_sphere_union(R, D, K), bubble_doc, R + 1.25 * K, [0.0]),
         "slab": (lambda: slab_minus_ball(**SLAB), slab_doc, SLAB["B"] + 1.25 * SLAB["k"], [SLAB["W"], -SLAB["W"]])}


def _kappa_max(sh, n=6000):
    """κ_max = max |собств. значение P H P/|∇f|| по выборке на M (как prototype/scripts/manifold_csg.py)."""
    P, _ = sample(sh, n, seed=0)
    Pd = jnp.asarray(P)
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    g = np.linalg.norm(G, axis=1)
    nn = G / g[:, None]
    Pr = np.eye(4)[None] - nn[:, :, None] * nn[:, None, :]
    return np.abs(np.linalg.eigvalsh(Pr @ H @ Pr / g[:, None, None])).max()


@pytest.mark.parametrize("case", sorted(CASES))
def test_cpu_equals_reference(tmp_path, case):
    make, doc, r_c, sheets = CASES[case]
    sh = make()
    P, _ = sample(sh, 300, seed=2)
    P = P + 0.05 * np.random.default_rng(3).normal(size=P.shape)
    r, out = _eval_graph(tmp_path, doc(), P)
    assert r.returncode == 0, r.stderr
    Pd = jnp.asarray(P)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    assert np.max(np.abs(np.array(out["f"]) - f) / (1 + np.abs(f))) < 1e-12
    assert np.max(np.abs(np.array(out["g"]) - G) / (1 + np.abs(G))) < 1e-11
    assert np.max(np.abs(np.array(out["H"]) - H) / (1 + np.abs(H))) < 1e-10
    m = out["meta"]
    assert m["flat"] and m["sheets"] == sheets and m["exit"] == 2 and not m["compact"]
    assert m["centers"] == [[0.0, 0.0, 0.0]] and abs(m["rho0"] - r_c) < 1e-12
    # ℓ_s = 0.5/κ_max: выборка C++ (1500 затравок) находит максимум не хуже 10 % от плотной выборки прототипа
    kap = _kappa_max(sh)
    assert 0.5 / kap * 0.999 <= m["scale"] <= 0.5 / kap / 0.9, (m["scale"], 0.5 / kap)


def test_single_sphere_is_compact(tmp_path):
    doc = _doc([{"id": "s", "type": "mf.sphere", "params": {"c": [1.0, 0.0, 0.0, 0.0], "R": 2.0}}], [], "s",
               objects=False)
    r, out = _eval_graph(tmp_path, doc, [[3.0, 0.0, 0.0, 0.0]])
    assert r.returncode == 0, r.stderr
    m = out["meta"]
    assert m["compact"] and not m["flat"] and m["scale"] == 2.0 and abs(m["s_max"] - 16 * np.pi) < 1e-12
    assert abs(out["f"][0]) < 1e-15


def test_flip_and_intersect(tmp_path):
    """Пересечение с flip_b = вычитание; пересечение с flip_a, flip_b = −объединение (M38)."""
    pts = np.random.default_rng(4).normal(size=(50, 4)) * 2 + np.array([0, 0, 0, 1.5])
    r1, a = _eval_graph(tmp_path, bubble_doc(d=-1.0, node="mf.intersect", flip=(0, 1)), pts)
    r2, b = _eval_graph(tmp_path, bubble_doc(d=-1.0, node="mf.subtract"), pts)
    r3, c = _eval_graph(tmp_path, bubble_doc(d=-1.0, node="mf.intersect", flip=(1, 1)), pts)
    for r in (r1, r2, r3):
        assert r.returncode == 0, r.stderr
    assert np.array_equal(a["f"], b["f"])
    r4, u = _eval_graph(tmp_path, bubble_doc(d=-1.0), pts)
    assert np.allclose(np.array(c["f"]), -np.array(u["f"]), rtol=0, atol=1e-14)


BAD = {
    "not_connected": (lambda: bubble_doc(k=0.9), "не соединена"),
    "two_bases": (lambda: _doc([{"id": "a", "type": "mf.hyperplane"}, {"id": "b", "type": "mf.two_sheets"},
                                {"id": "u", "type": "mf.union"}],
                               [{"from": "a.out", "to": "u.a"}, {"from": "b.out", "to": "u.b"}], "u", objects=False),
                  "две основы"),
    "no_base": (lambda: _doc([{"id": "a", "type": "mf.sphere"},
                              {"id": "b", "type": "mf.sphere", "params": {"c": [0.0, 0.0, 0.0, 5.0], "R": 1.0}},
                              {"id": "u", "type": "mf.union"}],
                             [{"from": "a.out", "to": "u.a"}, {"from": "b.out", "to": "u.b"}], "u", objects=False),
                "нужна основа"),
    "ellis_piece": (lambda: _doc([{"id": "a", "type": "mf.hyperplane"}, {"id": "b", "type": "mf.ellis"},
                                  {"id": "u", "type": "mf.union"}],
                                 [{"from": "a.out", "to": "u.a"}, {"from": "b.out", "to": "u.b"}], "u", objects=False),
                    "не входит в объединение"),
    "pinched": (lambda: bubble_doc(k=1.002), "почти пережата"),       # min |∇f| = 0.0125 (прототип)
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_errors(tmp_path, case):
    make, word = BAD[case]
    r, _ = _eval_graph(tmp_path, make(), [[0, 0, 0, 0]])
    assert r.returncode != 0 and word in r.stderr, r.stderr


def _derived(sh, meta):
    """Прототип с метаданными окна: ℓ_s и hcap (шар носителя), как derive в manifold_csg.py."""
    ell, r_c = meta["scale"], meta["rho0"]
    sh.scale = ell
    sh.hcap = lambda x: jnp.maximum(jnp.linalg.norm(x[:3]) - r_c, 0.25 * ell)
    return sh


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_gpu_jets(tmp_path, case):
    make, doc, _, _ = CASES[case]
    sh = make()
    N = 1000
    P, _ = sample(sh, N, seed=5)
    P = P + 0.02 * np.random.default_rng(6).normal(size=P.shape)
    Wd = np.random.default_rng(7).normal(size=P.shape)
    P32, W32 = gpu.f32(P), gpu.f32(Wd)
    spv = gpu.compile_shader("jets_u1.slang", "jets_u1", {"SCENE": gpu.SCENE_IDS["graph"]}, include=_header(tmp_path, doc()))
    out, _ = gpu.run(spv, "jets_u1", [("in", P32), ("in", W32), ("out", ("f4", (N * 8, 4)))], push=[("u32", N)],
                     groups=gpu.groups_for(N))
    o = out[2].reshape(N, 8, 4).astype(np.float64)
    Pd = jnp.asarray(P32, jnp.float64)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    rel = lambda got, ref, s: np.max(np.abs(got - ref) / np.maximum(s, np.abs(ref)))
    sc = np.max(np.abs(H), axis=(1, 2))[:, None] + 1.0
    assert rel(o[:, 0, 0], f, 1.0) < 5e-5
    assert rel(o[:, 1], G, 1.0) < 5e-5
    assert rel(o[:, 4:].reshape(N, 4, 4), H, sc[:, :, None]) < 5e-5


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(CASES))
def test_gpu_geodesics_vs_dop853(tmp_path, case):
    make, doc, r_c, sheets = CASES[case]
    r, out = _eval_graph(tmp_path, doc(), [])
    assert r.returncode == 0, r.stderr
    sh = _derived(make(), out["meta"])
    TOL, L = 1e-5, 12.0
    X, V = sample(sh, 24, seed=4)
    n = 12                                        # и лучи с листа прямо в шейку
    X = np.r_[X, np.c_[np.full(n, -(r_c + 1.0)), np.linspace(0.0, 2.0, n), np.zeros(n), np.full(n, sheets[0])]]
    V = np.r_[V, np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))]
    X, V = X.astype(np.float32).astype(np.float64), V.astype(np.float32).astype(np.float64)
    xr, _ = dop853(sh, X, V, L, alpha=0.1 / R)
    g, _ = gpu.trace("graph", X, V, tol=TOL, s_max=L, h0=0.1 * sh.scale, include=out["include"])
    p = make_integrator(sh, Config(tol=TOL, s_max=L))(X, V)
    e_g = np.max(np.linalg.norm(g["x"] - xr, axis=1))
    e_p = np.max(np.linalg.norm(np.asarray(p["x"]) - xr, axis=1))
    assert np.all(g["status"] == 1)
    assert e_g < 4 * e_p + 1e-5 * R, (e_g, e_p)


# ---- свой радиус шара носителя (DD-014, шаг B) ----------------------------------------------------------------------

WM = 1.4                                          # W у ламп _doc (лампа (0, −3, 1, W), r = 0.3)


def mixed_doc(k=2.0, objects=True):
    """Листы ±W с горловиной ρ₀ = 2 рядом с лампой (|q − c| = 2.5) ∪ сфера R = 3 над верхним листом далеко справа."""
    nodes = [{"id": "base", "type": "mf.two_sheets", "params": {"W": WM}},
             {"id": "t", "type": "mf.throat", "params": {"c": [-2.5, -3.0, 1.0], "b": 0.8, "rho0": 2.0}},
             {"id": "s", "type": "mf.sphere", "params": {"c": [8.0, 0.0, 0.0, WM + 3.5], "R": 3.0}},
             {"id": "u", "type": "mf.union", "params": {"k": k}}]
    links = [{"from": "base.out", "to": "t.base"}, {"from": "t.out", "to": "u.a"}, {"from": "s.out", "to": "u.b"}]
    return _doc(nodes, links, "u", objects=objects)


def test_own_radii_meta_and_nee(tmp_path):
    r, out = _eval_graph(tmp_path, mixed_doc(), [[0.0, 0.0, 0.0, WM]])
    assert r.returncode == 0, r.stderr
    m = out["meta"]
    assert m["centers"] == [[-2.5, -3.0, 1.0], [8.0, 0.0, 0.0]] and m["radii"] == [2.0, 3.0 + 1.25 * 2.0]
    assert m["rho0"] == 5.5 and m["sheets"] == [WM, -WM]
    # лампа в 2.5 от горловины ρ₀ = 2 (r = 0.3) — источник NEE; с общим радиусом 5.5 её бы не было
    assert out["nee"] == [0]


def test_throats_with_different_rho0(tmp_path):
    from test_node_manifolds import sheets_doc
    doc = sheets_doc()
    doc["graphs"]["main"]["nodes"][2]["params"]["rho0"] = 2.5
    r, out = _eval_graph(tmp_path, doc, [[0.0, 0.0, 0.0, 0.0]])
    assert r.returncode == 0, r.stderr
    assert out["meta"]["radii"] == [2.8, 2.5, 2.8] and out["meta"]["rho0"] == 2.8
