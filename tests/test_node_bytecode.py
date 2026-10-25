"""DD-011, шаг 5а: байткод выражений графа (gpu/src/bytecode.cpp) для интерпретатора в шейдерах.

- bc_check: программы, собранные C++ (`geo4d_gpu bytecode`), исполняет независимый Python (geo4d.bytecode.evaluate
  на jet.ops) — f, ∇f, H многообразия и g, ∇g составных объектов = вычислитель языка нод C++ (`eval-graph`) в float64;
- пределы: программа с > 32 регистрами — ошибка «только нативный путь»;
- GPU: интерпретатор (scenes.slang, prog_eval) даёт Jet7/HVP/Jet15 с той же точностью, что нативный заголовок графа."""
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import bytecode as B, gpu, gpu_pt as G
from test_node_manifolds import sheets_doc
from test_node_objects import CASES, N, scene
from test_node_slots import csg_doc

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")

FORMULA = "w - a * sin(x) * cosh(y / 3) + log(1 + z * z) / b + sqrt(x * x + 1) / (2 + exp(-y * y)) - smin(x, y, c)"


def formula_doc():
    return scene([], manifold=N("m", "mf.formula", formula=FORMULA, a=0.3, b=2.0, c=0.4, d=0.0, scale=1.0, s_max=50.0))


def csg_cases_doc():
    objs = []
    for k, name in enumerate(sorted(CASES)):
        ns, ls, sid, _ = CASES[name]
        ren = lambda i: f"c{k}{i}"
        objs.append((ren(sid), [dict(n, id=ren(n["id"])) for n in ns],
                     [{"from": ren(l["from"].split(".")[0]) + ".out", "to": ren(l["to"].split(".")[0]) + "." + l["to"].split(".")[1]}
                      for l in ls]))
    return scene(objs)


def _docs(tmp_path):
    docs = {"throats": sheets_doc(), "csg": csg_doc(), "csg_cases": csg_cases_doc(), "formula": formula_doc()}
    for p in sorted((ROOT / "scenes").glob("*.json")):
        docs[p.stem] = p
    return docs


def _cpp(tmp_path, name, doc, pts):
    d = tmp_path / name
    d.mkdir()
    if isinstance(doc, Path):
        src = doc
    else:
        src = d / "s.g4d"
        src.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    run = lambda *a: subprocess.run([str(gpu.EXE), *map(str, a)], capture_output=True, text=True, encoding="utf-8")
    r = run("bytecode", src, d / "bc.json")
    assert r.returncode == 0, r.stderr
    (d / "in.json").write_text(json.dumps({"points": pts.tolist()}), encoding="utf-8")
    r = run("eval-graph", src, d / "in.json", d / "ev.json")
    assert r.returncode == 0, r.stderr
    return (json.loads((d / f).read_text(encoding="utf-8")) for f in ("bc.json", "ev.json"))


def _points(n=300, seed=3):
    rng = np.random.default_rng(seed)
    P = rng.uniform(-6, 6, size=(n, 4))
    P[:, 3] = rng.uniform(-2, 2, size=n)
    return P


def _prog(p):
    """Программа: строки для GPU — округление до float32 строк с константами в double; исполняется версия double."""
    r64 = np.asarray(p["f64"], np.float64)
    assert np.array_equal(np.asarray(p["f32"], np.float32), r64.astype(np.float32))
    return lambda q: B.evaluate(r64, q[0], q[1], q[2], q[3])


@pytest.mark.parametrize("name", ["throats", "csg", "csg_cases", "formula"] + sorted(p.stem for p in (ROOT / "scenes").glob("*.json")))
def test_bc_check(tmp_path, name):
    doc = _docs(tmp_path)[name]
    P = _points()
    bc, ev = _cpp(tmp_path, name, doc, P)
    Pd = jnp.asarray(P)
    f = _prog(bc["manifold"])
    rel = lambda a, b: np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.maximum(1.0, np.abs(np.asarray(b))))
    assert rel(jax.vmap(f)(Pd), ev["f"]) < 1e-11
    assert rel(jax.vmap(jax.grad(f))(Pd), ev["g"]) < 1e-10
    assert rel(jax.vmap(jax.hessian(f))(Pd), ev["H"]) < 1e-9
    n = 0
    for rows, o in zip(bc["objects"], ev["objects"]):
        if rows is None:
            assert o["kind"] != "graph"
            continue
        n += 1
        g = _prog(rows)
        assert rel(jax.vmap(g)(Pd), o["f"]) < 1e-11, o["name"]
        assert rel(jax.vmap(jax.grad(g))(Pd), o["g"]) < 1e-10, o["name"]
    assert n == {"csg": 2, "csg_cases": len(CASES)}.get(name, 0)


def test_register_limit(tmp_path):
    """f₁·(f₂·(f₃·…)) — левые множители живы, пока считается правая часть: 40 множителей не помещаются в 32 регистра;
    20 — помещаются. Превышение — ошибка «только нативный путь» (нативный кодген такой сцены работает как раньше)."""
    def run(n):
        f = "w"
        for i in range(n, 0, -1):
            f = f"sin(x + {i}) * ({f})"
        doc = formula_doc()
        for nd in doc["graphs"]["main"]["nodes"]:
            if nd["id"] == "m":
                nd["params"]["formula"] = f
        d = tmp_path / f"n{n}.g4d"
        d.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return subprocess.run([str(gpu.EXE), "bytecode", str(d), str(tmp_path / "o.json")], capture_output=True,
                              text=True, encoding="utf-8")
    ok, bad = run(20), run(40)
    assert ok.returncode == 0, ok.stderr
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["manifold"]["f32"][0][1] <= 32
    assert bad.returncode != 0 and "регистров" in bad.stderr and "нативный" in bad.stderr


def _gpu_jets(spv, entry, bufs, n):
    out, _ = gpu.run(spv, entry, bufs + [("out", ("f4", (n * 8, 4)))], push=[("u32", n)], groups=gpu.groups_for(n))
    return out[len(bufs)].reshape(n, 8, 4).astype(np.float64)


def _errs(o, prog, P32, W32):
    """Отн. ошибки f, ∇f, ∇f·w, wᵀHw, H·w, H против float64 той же программы (как test_gpu_prog)."""
    f = _prog(prog)
    Pd, Wd = jnp.asarray(P32, jnp.float64), np.asarray(W32, np.float64)
    F = np.asarray(jax.vmap(f)(Pd))
    Gr = np.asarray(jax.vmap(jax.grad(f))(Pd))
    H = np.asarray(jax.vmap(jax.hessian(f))(Pd))
    Hw = np.einsum("nij,nj->ni", H, Wd)
    sc = np.max(np.abs(H), axis=(1, 2))[:, None] + 1.0
    wn = np.linalg.norm(Wd, axis=1)
    rel = lambda got, ref, s: float(np.max(np.abs(got - ref) / np.maximum(s, np.abs(ref))))
    return [rel(o[:, 0, 0], F, 1.0), rel(o[:, 1], Gr, 1.0), rel(o[:, 0, 1], np.sum(Gr * Wd, 1), 1.0),
            rel(o[:, 0, 2], np.einsum("ni,ni->n", Wd, Hw), wn ** 2 * sc[:, 0]), rel(o[:, 2], Hw, sc * wn[:, None]),
            rel(o[:, 4:].reshape(-1, 4, 4), H, sc[:, :, None])]


@pytest.mark.gpu
@pytest.mark.skipif(not gpu.available(), reason="нет geo4d_gpu или slangc")
@pytest.mark.parametrize("name", ["throats", "formula", "csg"])
def test_interpreter_gpu_jets(tmp_path, name):
    """Интерпретатор (prog_u1: SCENE_DYNAMIC + MF_PROG, программа с PROG_MF) — Jet7, HVP, Jet15 программ C++ с той же
    точностью float32, что нативный заголовок графа (jets_u1, SCENE_GRAPH) у многообразия; у объектов — та же граница."""
    doc = _docs(tmp_path)[name]
    n = 1000
    rng = np.random.default_rng(11)
    P = rng.uniform(-6, 6, size=(n, 4))
    P[:, 3] = rng.uniform(-2, 2, size=n)
    P32, W32 = gpu.f32(P), gpu.f32(rng.normal(size=P.shape))
    bc, ev = _cpp(tmp_path, name, doc, P[:2])
    spv = gpu.compile_shader("prog_u1.slang", "prog_u1", {"SCENE": gpu.SCENE_IDS["dynamic"], "MF_PROG": 1})
    progs = [("manifold", bc["manifold"])] + [(f"obj{k}", r) for k, r in enumerate(bc["objects"]) if r is not None]
    assert len(progs) == {"csg": 3}.get(name, 1)
    for label, prog in progs:
        region = G.prog_region(np.asarray(prog["f32"], np.float32))
        cam = np.zeros((G.PROG_BASE + len(region), 4), np.float32)
        cam[G.PROG_BASE:] = region
        e = _errs(_gpu_jets(spv, "prog_u1", [("in", cam), ("in", P32), ("in", W32)], n), prog, P32, W32)
        print(label, e)
        assert max(e) < 2e-5, (label, e)
    # многообразие: нативный путь на тех же точках — того же порядка
    (tmp_path / "h").mkdir()
    src = tmp_path / name / "s.g4d"
    r = subprocess.run([str(gpu.EXE), "eval-graph", str(src), str(tmp_path / name / "in.json"), str(tmp_path / "h" / "o.json")],
                       capture_output=True, text=True, encoding="utf-8")
    inc = json.loads((tmp_path / "h" / "o.json").read_text(encoding="utf-8"))["include"]
    spv0 = gpu.compile_shader("jets_u1.slang", "jets_u1", {"SCENE": gpu.SCENE_IDS["graph"]}, include=inc)
    e0 = _errs(_gpu_jets(spv0, "jets_u1", [("in", P32), ("in", W32)], n), bc["manifold"], P32, W32)
    print("native", e0)
    assert max(e0) < 2e-5
