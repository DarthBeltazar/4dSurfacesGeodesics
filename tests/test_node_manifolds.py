"""DD-011, шаг 3а: новые многообразия графа — «Два листа» + «Горловина» (M35) и «Многообразие-формула» (M37).

Эталон M35 — независимая формула на Python (shapes.compact_h), не текст из C++:
- CPU (`geo4d_gpu eval-graph`) = эталон (значение, градиент, гессиан, float64); M35 с двумя одинаковыми
  горловинами = M12;
- ошибки графа: пересечение шаров носителей, разный ρ₀, b ≥ ρ₀, горловина не на листах, формула с ошибкой;
- GPU (mf_scene.slang): jet-производные (как test_gpu_u1) и геодезические против DOP853 и прототипа (≤ 4×);
- mf.formula, равная S³ сцены S3.json, даёт побитово тот же кадр; сцена с тремя горловинами рисуется в окне."""
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import gpu
from geo4d.integrators import Config, make_integrator
from geo4d.jet import ops
from geo4d.reference import dop853
from geo4d.shapes import Shape, compact_h, compact_wormhole, sample, two_compact_wormholes

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")

W = 1.4
THROATS = [((-5.0, 0.5, 0.0), 1.0), ((4.0, -1.0, 0.5), 0.8), ((0.0, 6.0, -1.0), 1.2)]   # (c, b)
RHO0 = 2.8


def m35(W, throats, rho0):
    """M35 на Python: f = w² − W² + Σ (W² − hᵢ(|q − cᵢ|)); Shape для прототипа и DOP853."""
    def f(p):
        r = p[3] * p[3] - W * W
        for c, b in throats:
            r = r + (W * W - compact_h(ops.length(p[0] - c[0], p[1] - c[1], p[2] - c[2]), b, rho0, W))
        return r

    def raw(rng, n):
        k = rng.integers(0, len(throats), n)
        out = np.zeros((n, 4))
        for i, (c, b) in enumerate(throats):
            P = compact_wormhole(b, rho0, W).sample_raw(rng, n)
            P[:, :3] += np.asarray(c)
            out[k == i] = P[k == i]
        return out

    def hcap(x):
        d = jnp.min(jnp.stack([jnp.linalg.norm(x[:3] - jnp.asarray(c)) for c, _ in throats]))
        return jnp.maximum(d - rho0, 0.25 * min(b for _, b in throats))

    return Shape("sheets", f, min(b for _, b in throats), raw, hcap=hcap)


def _doc(manifold_nodes, manifold_links, out_id, objects=True):
    nodes = manifold_nodes + [{"id": "scene", "type": "out.scene"}]
    links = manifold_links + [{"from": f"{out_id}.out", "to": "scene.manifold"}]
    if objects:
        nodes += [{"id": "lamp", "type": "obj.ball4", "params": {"c": [0.0, -3.0, 1.0, W], "r": 0.3}},
                  {"id": "e", "type": "mat.emit", "params": {"emit": [20.0, 18.0, 14.0]}},
                  {"id": "floor", "type": "obj.halfspace", "params": {"axis": 2, "offset": -2.5, "sign": 1.0}},
                  {"id": "grey", "type": "mat.diffuse", "params": {"albedo": [0.6, 0.6, 0.6]}},
                  {"id": "o1", "type": "out.object", "params": {"name": "лампа"}},
                  {"id": "o2", "type": "out.object", "params": {"name": "пол"}}]
        links += [{"from": "lamp.out", "to": "o1.shape"}, {"from": "e.out", "to": "o1.material"},
                  {"from": "floor.out", "to": "o2.shape"}, {"from": "grey.out", "to": "o2.material"},
                  {"from": "o1.out", "to": "scene.objects"}, {"from": "o2.out", "to": "scene.objects"}]
    return {"format": "geo4d", "version": 2, "graphs": {"main": {"nodes": nodes, "links": links}},
            "camera": {"x": [0.0, -12.0, 0.5, W], "f": [0.0, 1.0, -0.05, 0.0], "up": [0.0, 0.0, 1.0, 0.0], "fov": 75.0},
            "settings": {"speed": 2.0}}


def sheets_doc(W=W, throats=THROATS, rho0=RHO0, **kw):
    nodes = [{"id": "base", "type": "mf.two_sheets", "params": {"W": W}}]
    links, prev = [], "base"
    for i, (c, b) in enumerate(throats):
        nodes.append({"id": f"t{i}", "type": "mf.throat", "params": {"c": list(c), "b": b, "rho0": rho0}})
        links.append({"from": f"{prev}.out", "to": f"t{i}.base"})
        prev = f"t{i}"
    return _doc(nodes, links, prev, **kw)


def _eval_graph(tmp_path, doc, pts):
    (tmp_path / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps({"points": np.asarray(pts).tolist()}), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "eval-graph", str(tmp_path / "s.g4d"), str(tmp_path / "in.json"),
                        str(tmp_path / "out.json")], capture_output=True, text=True, encoding="utf-8")
    return r, (json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) if r.returncode == 0 else None)


def _points(sh, n=200, seed=2):
    P, _ = sample(sh, n, seed=seed)
    return P + 0.05 * np.random.default_rng(seed + 1).normal(size=P.shape)


def test_m35_cpu_equals_reference(tmp_path):
    sh = m35(W, THROATS, RHO0)
    P = _points(sh)
    r, out = _eval_graph(tmp_path, sheets_doc(), P)
    assert r.returncode == 0, r.stderr
    Pd = jnp.asarray(P)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    assert np.max(np.abs(np.array(out["f"]) - f) / (1 + np.abs(f))) < 1e-12
    assert np.max(np.abs(np.array(out["g"]) - G) / (1 + np.abs(G))) < 1e-11
    assert np.max(np.abs(np.array(out["H"]) - H) / (1 + np.abs(H))) < 1e-10
    m = out["meta"]
    assert m["flat"] and m["sheets"] == [W, -W] and m["rho0"] == RHO0 and not m["legacy"]
    assert m["scale"] == min(b for _, b in THROATS) and len(m["centers"]) == 3 and m["exit"] == 2
    # проекция на M (C++) — на M
    proj = np.array(out["project"])
    assert np.max(np.abs(np.asarray(jax.vmap(sh.f_vec)(jnp.asarray(proj))))) < 1e-10


def test_m35_equals_m12(tmp_path):
    ref = two_compact_wormholes((-4.0, 0.0, 0.0), (4.0, 0.0, 0.0), 1.0, 3.0, 1.5)
    P = _points(ref)
    r, out = _eval_graph(tmp_path, sheets_doc(W=1.5, throats=[((-4.0, 0.0, 0.0), 1.0), ((4.0, 0.0, 0.0), 1.0)],
                                              rho0=3.0), P)
    assert r.returncode == 0, r.stderr
    Pd = jnp.asarray(P)
    assert np.max(np.abs(np.array(out["f"]) - np.asarray(jax.vmap(ref.f_vec)(Pd)))) < 1e-13
    G = np.asarray(jax.vmap(jax.grad(ref.f_vec))(Pd))
    assert np.max(np.abs(np.array(out["g"]) - G)) < 1e-12


BAD = {
    "overlap": (sheets_doc(throats=[((0.0, 0.0, 0.0), 1.0), ((5.0, 0.0, 0.0), 1.0)]), "пересекается"),
    "rho0_differs": (None, "один ρ₀"),
    "b_ge_rho0": (sheets_doc(throats=[((0.0, 0.0, 0.0), 3.0)]), "0 < b < ρ₀"),
    "throat_on_S3": (_doc([{"id": "s", "type": "mf.S3"}, {"id": "t", "type": "mf.throat"}],
                          [{"from": "s.out", "to": "t.base"}], "t", objects=False), "только в «Два листа»"),
    "bad_formula": (_doc([{"id": "f", "type": "mf.formula", "params": {"formula": "abs(x) - 1"}}], [], "f",
                         objects=False), "негладкая"),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_errors(tmp_path, case):
    doc, word = BAD[case]
    if case == "rho0_differs":
        doc = sheets_doc()
        doc["graphs"]["main"]["nodes"][2]["params"]["rho0"] = 2.5
    r, _ = _eval_graph(tmp_path, doc, [[0, 0, 0, 0]])
    assert r.returncode != 0 and word in r.stderr, r.stderr


def _header(tmp_path, doc):
    r, out = _eval_graph(tmp_path, doc, [])
    assert r.returncode == 0, r.stderr
    return out["include"]


@pytest.mark.gpu
def test_m35_gpu_jets(tmp_path):
    sh = m35(W, THROATS, RHO0)
    N = 1000
    P = _points(sh, N, seed=5)
    rng = np.random.default_rng(7)
    Wd = rng.normal(size=P.shape)
    P32, W32 = gpu.f32(P), gpu.f32(Wd)
    spv = gpu.compile_shader("jets_u1.slang", "jets_u1", {"SCENE": gpu.SCENE_IDS["graph"]}, include=_header(tmp_path, sheets_doc()))
    out, _ = gpu.run(spv, "jets_u1", [("in", P32), ("in", W32), ("out", ("f4", (N * 8, 4)))], push=[("u32", N)],
                     groups=gpu.groups_for(N))
    o = out[2].reshape(N, 8, 4).astype(np.float64)
    Pd = jnp.asarray(P32, jnp.float64)
    f = np.asarray(jax.vmap(sh.f_vec)(Pd))
    G = np.asarray(jax.vmap(jax.grad(sh.f_vec))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(sh.f_vec))(Pd))
    rel = lambda got, ref, s: np.max(np.abs(got - ref) / np.maximum(s, np.abs(ref)))
    sc = np.max(np.abs(H), axis=(1, 2))[:, None] + 1.0
    assert rel(o[:, 0, 0], f, 1.0) < 2e-5
    assert rel(o[:, 1], G, 1.0) < 2e-5
    assert rel(o[:, 4:].reshape(N, 4, 4), H, sc[:, :, None]) < 2e-5


@pytest.mark.gpu
def test_m35_gpu_geodesics_vs_dop853(tmp_path):
    sh = m35(W, THROATS, RHO0)
    TOL, L = 1e-5, 10.0
    X, V = sample(sh, 32, seed=4)
    X, V = X.astype(np.float32).astype(np.float64), V.astype(np.float32).astype(np.float64)
    xr, _ = dop853(sh, X, V, L)
    g, _ = gpu.trace("graph", X, V, tol=TOL, s_max=L, h0=0.1 * sh.scale, include=_header(tmp_path, sheets_doc()))
    p = make_integrator(sh, Config(tol=TOL, s_max=L))(X, V)
    e_g = np.max(np.linalg.norm(g["x"] - xr, axis=1))
    e_p = np.max(np.linalg.norm(np.asarray(p["x"]) - xr, axis=1))
    assert np.all(g["status"] == 1)
    assert e_g < 4 * e_p + 1e-5 * sh.scale, (e_g, e_p)


def _render(tmp_path, scene, name, legacy=False, bake=False):
    d = tmp_path / name
    r = subprocess.run([str(gpu.EXE), "export-graph", str(scene), str(d), "--res", "96x54", "--compile"]
                       + (["--legacy-codegen"] if legacy else []) + (["--bake-params"] if bake else []),
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    np.savetxt(tmp_path / "s.txt", np.zeros((6, 6)), fmt="%.6f")
    g = d / "app_graph.txt"
    g.write_text(g.read_text(encoding="utf-8") + f"hidden\nbench {gpu.qpath(tmp_path / 's.txt')} "
                 f"{gpu.qpath(d / 'f.json')}\ndump 5 {gpu.qpath(d)}\n", encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "app", str(g)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return {k: np.load(d / f"{k}_5.npy") for k in ("Frame", "Present")}, r.stdout


@pytest.mark.gpu
def test_formula_s3_equals_legacy_s3(tmp_path):
    """mf.formula «length(x, y, z, w) - a» с a = R, ℓ_s = R, s_max = 8πR — та же сцена, что S3.json: кадр побитово."""
    r = subprocess.run([str(gpu.EXE), "migrate", str(ROOT / "scenes" / "S3.json"), str(tmp_path / "s3.g4d")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / "s3.g4d").read_text(encoding="utf-8"))
    for n in doc["graphs"]["main"]["nodes"]:
        if n["id"] == "manifold":
            R = n["params"].get("R", 1.0)
            n.update(type="mf.formula", params={"formula": "length(x, y, z, w) - a", "a": R, "scale": R,
                                                "s_max": 8 * np.pi * R})
    (tmp_path / "f.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    a, _ = _render(tmp_path, tmp_path / "f.g4d", "formula", bake=True)      # побитово — с запечёнными числами
    b, _ = _render(tmp_path, ROOT / "scenes" / "S3.json", "legacy", legacy=True)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


@pytest.mark.gpu
def test_three_throats_render(tmp_path):
    (tmp_path / "t.g4d").write_text(json.dumps(sheets_doc(), ensure_ascii=False), encoding="utf-8")
    fr, out = _render(tmp_path, tmp_path / "t.g4d", "three")
    assert "editgraph" not in (tmp_path / "three" / "app_graph.txt").read_text(encoding="utf-8")
    rgb = (fr["Present"] & 0xFFFFFF)
    assert rgb.std() > 0 and np.isfinite(fr["Frame"]).all()
    assert "rows_check 0" in out
    # лампа на листе w = W вне шаров носителей — NEE-источник (M34 по плоским зонам M35)
    assert "flat 2 " in (tmp_path / "three" / "app_graph.txt").read_text(encoding="utf-8")


def test_window_accepts_g4d(tmp_path):
    """`geo4d_gpu сцена.g4d` — окно без Python (раньше принимался только .json) и пример scenes/three_wormholes.g4d."""
    r = subprocess.run([str(gpu.EXE), str(ROOT / "scenes" / "three_wormholes.g4d"), "--res", "96x54", "--graph-only",
                        str(tmp_path / "g")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    assert "flat 2 " in (tmp_path / "g" / "app_graph.txt").read_text(encoding="utf-8")
