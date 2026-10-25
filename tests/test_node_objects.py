"""DD-011, шаг 3б: составные объекты графа — гладкие ∪ ∩ − и сдвиг, поворот, масштаб (M36).

Эталон — независимая композиция на Python (jet.ops.smin и формулы M36), не выражение из C++:
- g и ∇g из C++ (`geo4d_gpu eval-graph`) = эталон (float64);
- M36: |∇g| ≤ 1 (1-липшицевость → g не больше расстояния до {g = 0}) для каждой операции;
- ограничивающая сфера (строки объекта, пропуск пустого) содержит {g ≤ 0};
- на GPU: гладкое объединение двух далёких шаров = два отдельных шара (кадр в пределах шума округления шага);
- сложная форма рисуется; смена разрешения у сцены-графа (редактор пишет .g4d с камерой) проходит в процессе."""
import json
import subprocess
from math import cos, log, radians, sin
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import gpu
from geo4d.jet import ops

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")
LN2 = log(2.0)


# ---- эталон M36 на Python ----
def ball(c, r):
    return lambda p: ops.length(*(p[i] - c[i] for i in range(4))) - r


def torus(c, R, r):
    return lambda p: ops.length(ops.length(p[0] - c[0], p[1] - c[1]) - R, p[2] - c[2], p[3] - c[3]) - r


def halfspace(axis, offset, sign):
    return lambda p: sign * (p[axis] - offset)


def union(a, b, k):
    return lambda p: ops.smin(a(p), b(p), k)


def intersect(a, b, k):
    return lambda p: -ops.smin(-a(p), -b(p), k) - k * LN2


def subtract(a, b, k):
    return lambda p: -ops.smin(-a(p), b(p), k) - k * LN2


def translate(g, t):
    return lambda p: g(tuple(p[i] - t[i] for i in range(4)))


def rotate(g, i, j, deg, c):
    ca, sa = cos(radians(deg)), sin(radians(deg))

    def f(p):
        q = [p[k] - c[k] for k in range(4)]
        a = [p[k] for k in range(4)]
        a[i] = ca * q[i] + sa * q[j] + c[i]
        a[j] = ca * q[j] - sa * q[i] + c[j]
        return g(tuple(a))
    return f


def scale(g, s, c):
    return lambda p: s * g(tuple((p[k] - c[k]) / s + c[k] for k in range(4)))


# ---- граф ----
def N(id, type, **params):
    return {"id": id, "type": type, "params": params}


def L(a, b):
    return {"from": f"{a}.out", "to": b}


def scene(objects, manifold=N("m", "mf.hyperplane"), camera=None):
    """objects: [(id_формы, [ноды], [связи])] → документ .g4d (материал — серый диффузный у всех)."""
    nodes = [manifold, N("scene", "out.scene"), N("grey", "mat.diffuse", albedo=[0.7, 0.6, 0.5])]
    links = [L(manifold["id"], "scene.manifold")]
    for k, (sid, ns, ls) in enumerate(objects):
        nodes += ns + [N(f"o{k}", "out.object", name=f"объект {k}")]
        links += ls + [L(sid, f"o{k}.shape"), L("grey", f"o{k}.material"), L(f"o{k}", "scene.objects")]
    return {"format": "geo4d", "version": 2, "graphs": {"main": {"nodes": nodes, "links": links}},
            "camera": camera or {"x": [0.0, -6.0, 0.6, 0.0], "f": [0.0, 1.0, -0.1, 0.0], "up": [0.0, 0.0, 1.0, 0.0],
                                 "fov": 70.0},
            "settings": {"speed": 1.0}}


CASES = {
    "union": ([N("a", "obj.ball4", c=[0, 0, 0, 0], r=0.8), N("b", "obj.ball4", c=[1.0, 0.3, 0, 0], r=0.6),
               N("u", "obj.union", k=0.3)], [L("a", "u.a"), L("b", "u.b")], "u",
              union(ball((0, 0, 0, 0), 0.8), ball((1.0, 0.3, 0, 0), 0.6), 0.3)),
    "subtract": ([N("a", "obj.ball4", c=[0, 0, 0, 0], r=1.0), N("b", "obj.ball4", c=[0, 0, 0, 0], r=0.5),
                  N("t", "xf.translate", t=[0.6, -0.2, 0.3, 0.0]), N("s", "obj.subtract", k=0.1)],
                 [L("b", "t.shape"), L("a", "s.a"), L("t", "s.b")], "s",
                 subtract(ball((0, 0, 0, 0), 1.0), translate(ball((0, 0, 0, 0), 0.5), (0.6, -0.2, 0.3, 0.0)), 0.1)),
    "intersect": ([N("a", "obj.ball4", c=[0.2, 0, 0, 0], r=1.0), N("h", "obj.halfspace", axis=2, offset=0.3, sign=1.0),
                   N("i", "obj.intersect", k=0.2)], [L("a", "i.a"), L("h", "i.b")], "i",
                  intersect(ball((0.2, 0, 0, 0), 1.0), halfspace(2, 0.3, 1.0), 0.2)),
    "rotate_scale": ([N("t", "obj.torus", c=[0.5, 0.0, 0.0, 0.0], R=0.7, r=0.2),
                      N("r", "xf.rotate", i=1, j=2, angle=35.0, center=[0.5, 0, 0, 0]),
                      N("s", "xf.scale", s=1.5, center=[0.0, 0.2, 0.0, 0.0])],
                     [L("t", "r.shape"), L("r", "s.shape")], "s",
                     scale(rotate(torus((0.5, 0, 0, 0), 0.7, 0.2), 1, 2, 35.0, (0.5, 0, 0, 0)), 1.5, (0.0, 0.2, 0, 0))),
}


def _eval(tmp_path, doc, pts):
    (tmp_path / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "in.json").write_text(json.dumps({"points": np.asarray(pts).tolist()}), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "eval-graph", str(tmp_path / "s.g4d"), str(tmp_path / "in.json"),
                        str(tmp_path / "out.json")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def _pts(n=400, seed=1, s=1.6):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 4)) * s
    P[:, 3] *= 0.2
    return P


@pytest.mark.parametrize("name", sorted(CASES))
def test_cpp_equals_reference(tmp_path, name):
    ns, ls, sid, ref = CASES[name]
    P = _pts()
    out = _eval(tmp_path, scene([(sid, ns, ls)]), P)
    o = out["objects"][0]
    assert o["kind"] == "graph"
    f = lambda p: ref((p[0], p[1], p[2], p[3]))
    Pd = jnp.asarray(P)
    g = np.asarray(jax.vmap(f)(Pd))
    G = np.asarray(jax.vmap(jax.grad(f))(Pd))
    assert np.max(np.abs(np.array(o["f"]) - g)) < 1e-12
    assert np.max(np.abs(np.array(o["g"]) - G)) < 1e-11


@pytest.mark.parametrize("name", sorted(CASES))
def test_m36_lipschitz_and_bound(tmp_path, name):
    ns, ls, sid, ref = CASES[name]
    P = _pts(4000, seed=3, s=1.3)
    f = lambda p: ref((p[0], p[1], p[2], p[3]))
    G = np.asarray(jax.vmap(jax.grad(f))(jnp.asarray(P)))
    assert np.max(np.linalg.norm(G, axis=1)) <= 1.0 + 1e-9                  # M36: 1-липшицевость
    out = _eval(tmp_path, scene([(sid, ns, ls)]), P[:1])
    o = out["objects"][0]
    inside = np.asarray(jax.vmap(f)(jnp.asarray(P))) <= 0
    if o["radius"] > 0:                                                      # ограниченная форма
        d = np.linalg.norm(P[inside] - np.asarray(o["center"]), axis=1)
        assert inside.any() and np.all(d <= o["radius"] + 1e-12), (d.max(), o["radius"])
    else:
        pytest.fail(f"{name}: форма должна быть ограничена (все случаи содержат шар или тор)")


def _render(tmp_path, doc, name, extra=""):
    d = tmp_path / name
    d.mkdir()
    (d / "s.g4d").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    np.savetxt(d / "script.txt", np.zeros((8, 6)), fmt="%.6f")
    (d / "extra.txt").write_text(f"hidden\nbench {gpu.qpath(d / 'script.txt')} {gpu.qpath(d / 'f.json')}\n"
                                 f"dump 7 {gpu.qpath(d)}\n" + extra, encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), str(d / "s.g4d"), "--res", "96x54", "--cube", "64", "--work-dir", str(d / "w"),
                        "--extra", str(d / "extra.txt")], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return {k: np.load(d / f"{k}_7.npy") for k in ("Frame", "Present", "AlbF", "NrmF")}, r.stdout


@pytest.mark.gpu
def test_far_union_equals_two_balls(tmp_path):
    """smin двух далёких шаров ≡ min (exp → 0): на GPU — те же пиксели, что у двух отдельных объектов."""
    a, b = N("a", "obj.ball4", c=[-1.2, 0, 0, 0], r=0.6), N("b", "obj.ball4", c=[1.2, 0, 0, 0], r=0.6)
    u, _ = _render(tmp_path, scene([("u", [a, b, N("u", "obj.union", k=0.01)], [L("a", "u.a"), L("b", "u.b")])]), "u")
    s, _ = _render(tmp_path, scene([("a", [a], []), ("b", [b], [])]), "sep")
    # первые попадания внутри силуэтов (альбедо и нормаль AOV) — те же поверхности. Кадр — в пределах шума путей:
    # у биссектрисы между шарами smin отличается от min на ≤ k·ln 2, шаги трассировки там другие, после отскоков пути
    # расходятся; ограничивающая сфера объединения больше — часть фона рядом считает трассировщик, а не режим A
    nu, ns_ = (np.linalg.norm(x["NrmF"][..., :3], axis=-1) > 0.3 for x in (u, s))
    m = nu & ns_
    m[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    m[[0, -1]] = False
    m[:, [0, -1]] = False
    assert m.sum() > 100
    for k in ("AlbF", "NrmF"):
        assert np.max(np.abs(u[k][..., :3] - s[k][..., :3])[m]) < 1e-5, k
    rgb = lambda p: np.stack([(p >> sh) & 255 for sh in (0, 8, 16)], -1).astype(float)
    mse = np.mean((rgb(u["Present"]) - rgb(s["Present"])) ** 2)
    assert 10 * np.log10(255 ** 2 / max(mse, 1e-12)) > 40


@pytest.mark.gpu
def test_csg_scene_renders_and_resizes(tmp_path):
    objs = [("u", CASES["union"][0], CASES["union"][1]),
            ("s", [dict(n, id="x" + n["id"]) for n in CASES["rotate_scale"][0]],
             [{"from": "x" + l["from"], "to": "x" + l["to"]} for l in CASES["rotate_scale"][1]])]
    objs[1] = ("xs", objs[1][1], objs[1][2])
    for n in objs[1][1]:
        if n["type"] == "obj.torus":
            n["params"]["c"] = [0.5, 1.5, 0.0, 0.0]
    fr, out = _render(tmp_path, scene(objs), "csg", extra="resize 3 128 72\n")
    graph = (tmp_path / "csg" / "w" / "edit_1.json").read_text(encoding="utf-8")
    assert json.loads(graph)["version"] == 2                         # редактор перегенерировал граф, не version 1
    assert out.count("standalone_build") == 2 and "rows_check 0" in out
    assert fr["Frame"].shape[:2] == (72, 128) and np.isfinite(fr["Frame"]).all()
    assert (fr["Present"] & 0xFFFFFF).std() > 0
