"""DD-015: ноды полей (fld.ball, fld.halfspace, fld.formula) в двух ролях, правило типов портов, xf над многообразием (M40).

- сцены, где прежние ноды заменены полями, дают тот же граф окна и те же шейдеры (export-graph — текст побитово);
- правило типов: «только объект» к многообразию и «только многообразие» к объекту (в том числе через цепочку xf)
  отвергаются с именем ноды, закрепившей тип; одно поле в двух ролях сразу — допустимо;
- xf над многообразием: f = s·f₀(T⁻¹p) на CPU (C++ = прототип), метаданные M40 (листы, центры, радиусы, ℓ_s ×s),
  ошибки (поворот с осью w у листов, вид каталога под xf), формула под xf; GPU-геодезические против DOP853;
- объект-формула: g/L, шар формулой = прежнему шару."""
import copy
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import gpu
from geo4d.integrators import Config, make_integrator
from geo4d.reference import dop853
from geo4d.shapes import Shape, plane_sphere_union, sample

from test_node_csg import BANDS, K, R, D, bubble_doc
from test_node_manifolds import _doc, _eval_graph

pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")
ROOT = Path(__file__).resolve().parents[1]
LIB = {n["type"]: n for f in (ROOT / "nodes").glob("*.json") for n in json.loads(f.read_text(encoding="utf-8"))}


def _defaults(t):
    return {p["name"]: p["default"] for p in LIB[t].get("params", []) if "default" in p}


# прежняя нода → (поле, переименование параметров, явные значения поля)
OLD2NEW = {"mf.sphere": ("fld.ball", {}, {}), "obj.ball4": ("fld.ball", {"r": "R"}, {}),
           "obj.halfspace": ("fld.halfspace", {}, {}), "mf.formula": ("fld.formula", {}, {}),
           "mf.hyperplane": ("fld.halfspace", {}, {"axis": 3, "offset": 0.0, "sign": 1.0})}


def to_fields(doc):
    d = copy.deepcopy(doc)
    n_conv = 0
    for n in d["graphs"]["main"]["nodes"]:
        if n["type"] not in OLD2NEW:
            continue
        new, ren, extra = OLD2NEW[n["type"]]
        p = {**_defaults(n["type"]), **n.get("params", {})}
        n["params"] = {**{ren.get(k, k): v for k, v in p.items()}, **extra}
        n["type"] = new
        n_conv += 1
    return d, n_conv


def _exe(*args):
    return subprocess.run([str(gpu.EXE), *map(str, args)], capture_output=True, text=True, encoding="utf-8")


def _export(d, doc):
    d.mkdir(parents=True)
    (d / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    r = _exe("export-graph", d / "s.g4d", d / "g", "--res", "96x54")
    assert r.returncode == 0, r.stderr
    t = (d / "g" / "app_graph.txt").read_text(encoding="utf-8").replace(str(d), "<D>").replace(d.as_posix(), "<D>")
    r, out = _eval_graph(d, doc, [])                 # сгенерированные заголовки (mf_scene, pt_scene, …) — текстом
    assert r.returncode == 0, r.stderr
    hdr = sorted((q.name, q.read_text(encoding="utf-8")) for q in Path(out["include"]).glob("*.slang"))
    assert hdr
    return [l for l in t.splitlines() if not l.startswith("editor ")], hdr


def _scenes():
    out = [p for p in sorted((ROOT / "scenes").glob("*.g4d"))]
    return out + [ROOT / "scenes" / f"{s}.json" for s in ("two_wormholes", "S3", "hyperplane")
                  if (ROOT / "scenes" / f"{s}.json").exists()]


@pytest.mark.parametrize("scene", _scenes(), ids=lambda p: p.name)
def test_fields_give_same_graph(tmp_path, scene):
    if scene.suffix == ".json":
        r = _exe("migrate", scene, tmp_path / "m.g4d")
        assert r.returncode == 0, r.stderr
        doc = json.loads((tmp_path / "m.g4d").read_text(encoding="utf-8"))
    else:
        doc = json.loads(scene.read_text(encoding="utf-8"))
    new, n = to_fields(doc)
    if n == 0:
        pytest.skip("нет прежних нод полей")
    a, b = _export(tmp_path / "old", doc), _export(tmp_path / "new", new)
    assert a[0] == b[0]
    assert a[1] == b[1]


# ---- правило типов ----------------------------------------------------------------------------------------------

def _mf(nodes, links, out):
    return _doc(nodes, links, out, objects=False)


def _with_object(doc, nodes, links, shape):
    d = copy.deepcopy(doc)
    g = d["graphs"]["main"]
    g["nodes"] += nodes + [{"id": "mx", "type": "mat.diffuse"}, {"id": "ox", "type": "out.object"}]
    g["links"] += links + [{"from": f"{shape}.out", "to": "ox.shape"}, {"from": "mx.out", "to": "ox.material"},
                           {"from": "ox.out", "to": "scene.objects"}]
    return d


BAD_TYPES = {
    "objunion_as_manifold": (lambda: _mf([{"id": "a", "type": "fld.ball"}, {"id": "b", "type": "fld.ball"},
                                          {"id": "u", "type": "obj.union"}],
                                         [{"from": "a.out", "to": "u.a"}, {"from": "b.out", "to": "u.b"}], "u"),
                             "только объект"),
    "ellis_via_xf_as_object": (lambda: _with_object(_mf([{"id": "p", "type": "fld.halfspace", "params": {"axis": 3}}], [], "p"),
                                                    [{"id": "e", "type": "mf.ellis"}, {"id": "t", "type": "xf.translate"}],
                                                    [{"from": "e.out", "to": "t.shape"}], "t"),
                               "только многообразие"),
    "throat_as_object": (lambda: _with_object(_mf([{"id": "p", "type": "fld.halfspace", "params": {"axis": 3}}], [], "p"),
                                              [{"id": "s2", "type": "mf.two_sheets"}], [], "s2"),
                         "только многообразие"),
}


@pytest.mark.parametrize("case", sorted(BAD_TYPES))
def test_type_rule_rejects(tmp_path, case):
    make, word = BAD_TYPES[case]
    r, _ = _eval_graph(tmp_path, make(), [])
    assert r.returncode != 0 and word in r.stderr, r.stderr


def test_one_field_in_both_roles(tmp_path):
    """Одна fld.ball — сфера пузыря (многообразие) и объект-шар: каждая роль читает её своим кодом."""
    doc = bubble_doc()
    g = doc["graphs"]["main"]
    for n in g["nodes"]:
        if n["id"] == "s":
            n["type"] = "fld.ball"
    doc = _with_object(doc, [], [], "s")
    r, out = _eval_graph(tmp_path, doc, [[0.0, 0.0, 0.0, 0.0]])
    assert r.returncode == 0, r.stderr
    assert out["meta"]["radii"] == [R + 1.25 * K]
    assert [o["kind"] for o in out["objects"]] == ["ball4"] and out["objects"][0]["center"] == [0.0, 0.0, 0.0, R + D]


# ---- xf над многообразием (M40) ------------------------------------------------------------------------------------

def _xf_bubble(xf):
    """bubble (плоскость ∪ сфера) под цепочкой преобразований xf = [(тип, параметры), …] (первое — ближе к куску)."""
    doc = bubble_doc()
    g = doc["graphs"]["main"]
    g["links"] = [l for l in g["links"] if l["to"] != "scene.manifold"]
    prev = "u"
    for i, (t, p) in enumerate(xf):
        g["nodes"].append({"id": f"x{i}", "type": t, "params": p})
        g["links"].append({"from": f"{prev}.out", "to": f"x{i}.shape"})
        prev = f"x{i}"
    g["links"].append({"from": f"{prev}.out", "to": "scene.manifold"})
    return doc


XF = {"translate": ([("xf.translate", {"t": [1.0, -2.0, 0.5, 0.7]})], 1.0),
      "rotate_xy": ([("xf.rotate", {"i": 0, "j": 1, "angle": 30.0, "center": [0.3, -0.2, 0.1, 0.4]})], 1.0),
      "rot_scale": ([("xf.rotate", {"i": 0, "j": 2, "angle": 25.0, "center": [0.3, -0.2, 0.1, 0.4]}),
                     ("xf.scale", {"s": 1.5, "center": [0.3, -0.2, 0.1, 0.4]}),
                     ("xf.translate", {"t": [0.5, 1.0, 0.0, -0.4]})], 1.5)}


def _fwd_inv(xf):
    """T и T⁻¹ по цепочке (как в scene_graph.cpp: каждое преобразование применяется к результату предыдущего)."""
    steps = []
    for t, p in xf:
        if t == "xf.translate":
            tv = np.array(p["t"])
            steps.append((lambda q, tv=tv: q + tv, lambda q, tv=tv: q - tv, 1.0))
        elif t == "xf.rotate":
            i, j, a, c = p["i"], p["j"], np.deg2rad(p["angle"]), np.array(p["center"])
            M = np.eye(4)
            M[i, i] = M[j, j] = np.cos(a)
            M[j, i], M[i, j] = np.sin(a), -np.sin(a)
            steps.append((lambda q, M=M, c=c: c + M @ (q - c), lambda q, M=M, c=c: c + M.T @ (q - c), 1.0))
        else:
            s, c = p["s"], np.array(p["center"])
            steps.append((lambda q, s=s, c=c: c + s * (q - c), lambda q, s=s, c=c: c + (q - c) / s, s))
    fwd = lambda q: _chain([f for f, _, _ in steps], q)
    inv = lambda q: _chain([g for _, g, _ in steps][::-1], q)
    return fwd, inv, float(np.prod([s for _, _, s in steps]))


def _chain(fs, q):
    for f in fs:
        q = f(q)
    return q


@pytest.mark.parametrize("case", sorted(XF))
def test_xf_manifold_cpu_and_meta(tmp_path, case):
    xf, s = XF[case]
    fwd, inv, s2 = _fwd_inv(xf)
    assert s2 == s
    sh = plane_sphere_union(R, D, K)
    (tmp_path / "a").mkdir()
    r0, out0 = _eval_graph(tmp_path / "a", bubble_doc(), [])
    assert r0.returncode == 0, r0.stderr
    P0, _ = sample(sh, 100, seed=2)
    P = np.array([fwd(p) for p in P0]) + 0.03 * np.random.default_rng(3).normal(size=P0.shape)
    (tmp_path / "b").mkdir()
    r, out = _eval_graph(tmp_path / "b", _xf_bubble(xf), P)
    assert r.returncode == 0, r.stderr
    f_ref = np.array([s * float(sh.f_vec(jnp.asarray(inv(p)))) for p in P])
    G_ref = np.array(jax.vmap(jax.grad(lambda q: s * sh.f_vec(jnp.asarray(inv_j(xf, q)))))(jnp.asarray(P)))
    assert np.max(np.abs(np.array(out["f"]) - f_ref)) < 1e-10
    assert np.max(np.abs(np.array(out["g"]) - G_ref)) < 1e-9
    m0, m = out0["meta"], out["meta"]
    c0 = np.array([0.0, 0.0, 0.0, R + D])
    assert np.allclose(m["centers"], [fwd(c0)[:3]]) and np.allclose(m["radii"], [s * r for r in m0["radii"]])
    assert np.allclose(m["sheets"], [fwd(np.zeros(4))[3]])
    assert abs(m["scale"] / (s * m0["scale"]) - 1) < 0.02          # κ_max по той же (преобразованной) выборке


def inv_j(xf, q):
    """T⁻¹ на jnp (для градиента)."""
    for t, p in reversed(xf):
        if t == "xf.translate":
            q = q - jnp.asarray(p["t"])
        elif t == "xf.rotate":
            i, j, a, c = p["i"], p["j"], np.deg2rad(p["angle"]), jnp.asarray(p["center"])
            M = np.eye(4)
            M[i, i] = M[j, j] = np.cos(a)
            M[j, i], M[i, j] = np.sin(a), -np.sin(a)
            q = c + jnp.asarray(M.T) @ (q - c)
        else:
            c = jnp.asarray(p["center"])
            q = c + (q - c) / p["s"]
    return q


@pytest.mark.gpu
def test_xf_manifold_gpu_geodesics(tmp_path):
    """Пузырь под поворотом+масштабом+сдвигом: GPU против DOP853 ≤ 4× прототипа (тот же hcap M39/M40)."""
    xf, s = XF["rot_scale"]
    fwd, inv, _ = _fwd_inv(xf)
    r, out = _eval_graph(tmp_path, _xf_bubble(xf), [])
    assert r.returncode == 0, r.stderr
    m = out["meta"]
    base = plane_sphere_union(R, D, K)
    t0, lip, ell_p = BANDS["bubble"]
    rc, cq = m["radii"][0], jnp.asarray(m["centers"][0])
    # T⁻¹(p) = A p + b — аффинное; на jet-числах — линейной комбинацией координат
    b = inv(np.zeros(4))
    A = np.array([inv(e) - b for e in np.eye(4)]).T
    aff = lambda p: tuple(sum(float(A[k, j]) * p[j] for j in range(4)) + float(b[k]) for k in range(4))
    f2 = lambda p: s * base.f(aff(p))
    sh = Shape("bubble/xf", f2, m["scale"], lambda rng, n: np.array([fwd(p) for p in base.sample_raw(rng, n)]),
               # M39 после T: |t′| − k′ = s·(|t| − k) в прообразе, масштаб кусков ×s
               hcap=lambda x: jnp.maximum(jnp.linalg.norm(x[:3] - cq) - rc,
                                          jnp.minimum(0.25 * s * ell_p, jnp.maximum(s * t0(inv_j(xf, x)) / lip, 0.25 * m["scale"]))))
    X, V = sample(sh, 24, seed=4)
    X, V = X.astype(np.float32).astype(np.float64), V.astype(np.float32).astype(np.float64)
    TOL, L = 1e-5, 12.0 * s
    xr, _ = dop853(sh, X, V, L, alpha=0.1 / (s * R), max_step=0.05 * s)
    g, _ = gpu.trace("graph", X, V, tol=TOL, s_max=L, h0=0.1 * sh.scale, include=out["include"])
    p = make_integrator(sh, Config(tol=TOL, s_max=L))(X, V)
    e_g = np.max(np.linalg.norm(g["x"] - xr, axis=1))
    e_p = np.max(np.linalg.norm(np.asarray(p["x"]) - xr, axis=1))
    assert np.all(g["status"] == 1)
    assert e_g < 4 * e_p + 1e-5 * s * R, (e_g, e_p)


XF_BAD = {
    "rotate_w_with_sheets": (lambda: _xf_bubble([("xf.rotate", {"i": 0, "j": 3, "angle": 10.0})]), "осью w"),
    "catalog_under_xf": (lambda: _mf([{"id": "m", "type": "mf.S3"}, {"id": "t", "type": "xf.translate"}],
                                     [{"from": "m.out", "to": "t.shape"}], "t"), "вид каталога"),
    "halfspace_axis_z": (lambda: _mf([{"id": "p", "type": "fld.halfspace", "params": {"axis": 2}}], [], "p"), "оси w"),
}


@pytest.mark.parametrize("case", sorted(XF_BAD))
def test_xf_manifold_errors(tmp_path, case):
    make, word = XF_BAD[case]
    r, _ = _eval_graph(tmp_path, make(), [])
    assert r.returncode != 0 and word in r.stderr, r.stderr


def test_formula_manifold_under_xf(tmp_path):
    """Формула S³ под сдвигом и масштабом = формула сдвинутой S³: f, scale и s_max ×s."""
    form = lambda p: {"id": "f", "type": "fld.formula", "params": {"formula": "length(x, y, z, w) - a", "a": 2.0,
                                                                    "scale": 2.0, "s_max": 40.0, **p}}
    doc = _mf([form({}), {"id": "t", "type": "xf.scale", "params": {"s": 1.5, "center": [1.0, 0.0, 0.0, 0.0]}},
               {"id": "u", "type": "xf.translate", "params": {"t": [0.0, 1.0, 0.0, 0.0]}}],
              [{"from": "f.out", "to": "t.shape"}, {"from": "t.out", "to": "u.shape"}], "u")
    P = np.random.default_rng(5).normal(size=(50, 4)) * 2
    r, out = _eval_graph(tmp_path, doc, P)
    assert r.returncode == 0, r.stderr
    c = np.array([1.0, 0.0, 0.0, 0.0]) * (1 - 1.5) + np.array([0.0, 1.0, 0.0, 0.0])    # центр S³ после T
    ref = 1.5 * (np.linalg.norm((P - c) / 1.5, axis=1) - 2.0)
    assert np.max(np.abs(np.array(out["f"]) - ref)) < 1e-12
    assert out["meta"]["scale"] == pytest.approx(3.0) and out["meta"]["s_max"] == pytest.approx(60.0)


def fields_doc():
    """Сцена на полях: пузырь (fld.halfspace ∪ fld.ball) под поворотом+масштабом+сдвигом, объекты — лампа fld.ball и
    тор формулой fld.formula (L = 1), тигр каталога как объект — для байткода, интерпретатора и предпросмотра."""
    doc = _xf_bubble(XF["rot_scale"][0])
    for n in doc["graphs"]["main"]["nodes"]:
        if n["id"] == "plane":
            n.update(type="fld.halfspace", params={"axis": 3, "offset": 0.0, "sign": 1.0})
        if n["id"] == "s":
            n["type"] = "fld.ball"
    doc = _with_object(doc, [{"id": "lb", "type": "fld.ball", "params": {"c": [0.0, -4.0, 1.0, -0.4], "R": 0.4}}], [], "lb")
    g = doc["graphs"]["main"]
    for n in g["nodes"]:
        if n["id"] == "mx":
            n.update(type="mat.emit", params={"emit": [20.0, 18.0, 14.0]})
    tor = {"id": "tor", "type": "fld.formula", "params": {"formula": "length(length(x - a, y) - b, z - c, w + 0.4) - d",
                                                           "a": 2.0, "b": 0.8, "c": 0.5, "d": 0.2, "L": 1.0}}
    tig = {"id": "tig", "type": "mf.tiger", "params": {"R1": 0.6, "R2": 0.5, "r": 0.15}}
    tgt = {"id": "tgt", "type": "xf.translate", "params": {"t": [-1.5, 1.0, 0.8, -0.4]}}
    g["nodes"] += [tor, {"id": "m2", "type": "mat.diffuse"}, {"id": "o2", "type": "out.object"},
                   tig, tgt, {"id": "m3", "type": "mat.metal"}, {"id": "o3", "type": "out.object"}]
    g["links"] += [{"from": "tor.out", "to": "o2.shape"}, {"from": "m2.out", "to": "o2.material"},
                   {"from": "o2.out", "to": "scene.objects"},
                   {"from": "tig.out", "to": "tgt.shape"}, {"from": "tgt.out", "to": "o3.shape"},
                   {"from": "m3.out", "to": "o3.material"}, {"from": "o3.out", "to": "scene.objects"}]
    return doc


def test_fields_doc_builds(tmp_path):
    r, out = _eval_graph(tmp_path, fields_doc(), [])
    assert r.returncode == 0, r.stderr
    assert [o["kind"] for o in out["objects"]] == ["ball4", "graph", "graph"]


# ---- объект-формула ----------------------------------------------------------------------------------------------

def test_formula_object_equals_ball(tmp_path):
    base = _mf([{"id": "p", "type": "fld.halfspace", "params": {"axis": 3}}], [], "p")
    ball = _with_object(base, [{"id": "b", "type": "fld.ball", "params": {"c": [0.5, -1.0, 0.2, 0.0], "R": 0.7}}], [], "b")
    form = _with_object(base, [{"id": "b", "type": "fld.formula",
                                "params": {"formula": "length(x - a, y - b, z - c, w) - d", "a": 0.5, "b": -1.0,
                                           "c": 0.2, "d": 0.7, "L": 2.0}}], [], "b")
    P = np.random.default_rng(6).normal(size=(40, 4))
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    r1, o1 = _eval_graph(tmp_path / "a", ball, P)
    r2, o2 = _eval_graph(tmp_path / "b", form, P)
    assert r1.returncode == 0 and r2.returncode == 0, r1.stderr + r2.stderr
    a, b = o1["objects"][0], o2["objects"][0]
    assert a["kind"] == "ball4" and b["kind"] == "graph" and b["radius"] == -1.0
    assert np.allclose(np.array(b["f"]) * 2.0, a["f"], atol=1e-12)      # g/L, L = 2


# ---- многообразия каталога в роли объекта (S³, сферитор, торисфера, тигр) ---------------------------------------------

def _dist(t, P, p):
    """Расстояние до поверхности (эталон, numpy): S³ — сфера, сферитор — 2-сфера в w = 0, торисфера — окружность в xy,
    тигр — тор S¹(R1) × S¹(R2); минус r — тело трубки."""
    q3, xy, zw = np.linalg.norm(P[:, :3], axis=1), np.linalg.norm(P[:, :2], axis=1), np.linalg.norm(P[:, 2:], axis=1)
    if t == "mf.S3":
        return np.linalg.norm(P, axis=1) - p["R"]
    if t == "mf.spheritorus":
        return np.hypot(q3 - p["R"], P[:, 3]) - p["r"]
    if t == "mf.torisphere":
        return np.sqrt((xy - p["R"]) ** 2 + P[:, 2] ** 2 + P[:, 3] ** 2) - p["r"]
    return np.hypot(xy - p["R1"], zw - p["R2"]) - p["r"]


CATALOG = {"mf.S3": ({"R": 1.2}, 1.2), "mf.spheritorus": ({"R": 1.5, "r": 0.4}, 1.9),
           "mf.torisphere": ({"R": 1.5, "r": 0.4}, 1.9),
           "mf.tiger": ({"R1": 1.0, "R2": 1.5, "r": 0.3}, float(np.hypot(1.3, 1.8)))}


@pytest.mark.parametrize("t", sorted(CATALOG))
def test_catalog_manifold_as_object(tmp_path, t):
    """g = точное расстояние (|∇g| = 1 вне осей), ограничивающая сфера около 0 радиуса r₀ (+ сдвиг xf.translate),
    в роли многообразия у тех же нод — прежний путь (test_fields_give_same_graph, S3.json)."""
    params, r0 = CATALOG[t]
    shift = [0.5, -2.0, 0.3, 0.0]
    base = _mf([{"id": "p", "type": "fld.halfspace", "params": {"axis": 3}}], [], "p")
    doc = _with_object(base, [{"id": "m", "type": t, "params": params}, {"id": "tr", "type": "xf.translate", "params": {"t": shift}}],
                       [{"from": "m.out", "to": "tr.shape"}], "tr")
    P = np.random.default_rng(8).normal(size=(300, 4)) * 1.5 + np.array(shift)
    r, out = _eval_graph(tmp_path, doc, P)
    assert r.returncode == 0, r.stderr
    o = out["objects"][0]
    assert o["kind"] == "graph"
    assert np.allclose(o["center"], shift) and o["radius"] == pytest.approx(r0)
    assert np.max(np.abs(np.array(o["f"]) - _dist(t, P - np.array(shift), params))) < 1e-9
    gn = np.linalg.norm(np.array(o["g"]), axis=1)
    assert np.max(gn) < 1 + 1e-9 and np.median(gn) > 1 - 1e-9                     # M36: 1-липшицева
    # поверхность (g = 0) — внутри ограничивающей сферы: проекция точек на g = 0 вдоль -g∇g
    Q = P - np.array(o["f"])[:, None] * np.array(o["g"])
    assert np.max(np.linalg.norm(Q - np.array(shift), axis=1)) <= r0 + 1e-6
