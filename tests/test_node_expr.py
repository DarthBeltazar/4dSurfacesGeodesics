"""DD-011, шаг 1: язык выражений нод. C++ (`geo4d_gpu eval`, expr.cpp) и эталон Python (geo4d.nodes.expr):
- значения, градиенты и гессианы совпадают (float64) на формулах нынешних многообразий и на смешанных выражениях;
- вывод в Slang для нынешних видов — буква в букву текст `sdf` из gpu/shaders/scenes.slang (миграция без изменения
  SPIR-V, DD-011 §4);
- запреты контекста и ошибки разбора — одни и те же сообщения и позиции в C++ и Python."""
import json
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geo4d import gpu
from geo4d.nodes import expr as E

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not gpu.EXE.exists(), reason="нет geo4d_gpu")

# нынешние многообразия: формула языка, параметры, имена дефайнов scenes.slang
LEGACY = {
    "S3": ("length(x, y, z, w) - R", dict(R=1.3), dict(R="MF_R")),
    "spheritorus": ("length(length(x, y, z) - R, w) - r", dict(R=1.1, r=0.35), dict(R="MF_R", r="MF_RT")),
    "torisphere": ("length(length(x, y) - R, z, w) - r", dict(R=1.2, r=0.4), dict(R="MF_R", r="MF_RT")),
    "tiger": ("length(length(x, y) - R1, length(z, w) - R2) - r", dict(R1=1.2, R2=0.8, r=0.3),
              dict(R1="MF_R1", R2="MF_R2", r="MF_RT")),
    "ellis": ("length(x, y, z) - b * cosh(w * (1.0 / b))", dict(b=1.3), dict(b="MF_B")),
    "compact_wh": ("w * w - compact_h(length(x, y, z), b, rho0, W)", dict(b=1.0, rho0=3.0, W=1.5),
                   dict(b="MF_B", rho0="MF_RHO0", W="MF_W")),
    "hyperplane": ("w", {}, {}),
}
SCENE_KEY = {"S3": "S3", "spheritorus": "SPHERITORUS", "torisphere": "TORISPHERE", "tiger": "TIGER", "ellis": "ELLIS",
             "compact_wh": "COMPACT", "hyperplane": "HYPERPLANE"}
TWO = ("w * w - (compact_h(length(x - c1.x, y - c1.y, z - c1.z), b, rho0, W) + "
       "compact_h(length(x - c2.x, y - c2.y, z - c2.z), b, rho0, W) - W * W)")
TWO_P = {"c1.x": -4.0, "c1.y": 0.0, "c1.z": 0.0, "c2.x": 4.0, "c2.y": 0.0, "c2.z": 0.0, "b": 1.0, "rho0": 3.0, "W": 1.5}
MIXED = [
    ("smin(length(x, y, z) - 1.0, w * w - 0.25, 0.3) + 0.1 * exp(-x^2)", {}),
    ("-(x - a) * (y + 2) / (1 + z^2) + sinh(w / a) - log(2 + cos(x))", dict(a=1.7)),
    ("sqrt(1 + x*x + y*y) * (k / (2 + w*w)) - 3 * sin(k * z)", dict(k=0.6)),
    ("length(x * (1.0 / 0.8), y * (1.0 / 1.0), z * (1.0 / 1.3), w * (1.0 / 1.6)) - 1.0", {}),
]
ERRORS = [("abs(x) - 1", "негладкая"), ("min(x, y)", "негладкая"), ("length(x, y) - q", "неизвестное имя"),
          ("smin(x, y, z)", "не должен зависеть"), ("x^", "целое"), ("(x + 1", "скобки"), ("foo(x)", "неизвестная функция"),
          ("length(x)", "число аргументов"), ("x + 1 2", "лишний текст"), ("compact_h(1, 2, 3, 4)", "только для полей")]


def _run(cases, tmp_path):
    (tmp_path / "in.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    r = subprocess.run([str(gpu.EXE), "eval", str(tmp_path / "in.json"), str(tmp_path / "out.json")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))


def _points(n=40, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, 4)) * 2.0).tolist()


def _ref(formula, params, pts):
    e = E.parse(formula, list(params))
    f = lambda p: E.evaluate(e, params, (p[0], p[1], p[2], p[3]))
    P = jnp.asarray(pts, jnp.float64)
    return (np.asarray(jax.vmap(f)(P)), np.asarray(jax.vmap(jax.grad(f))(P)), np.asarray(jax.vmap(jax.hessian(f))(P)))


@pytest.mark.parametrize("name", sorted(LEGACY) + ["two_compact_wh"] + [f"mixed{k}" for k in range(len(MIXED))])
def test_cpp_equals_python(tmp_path, name):
    if name in LEGACY:
        formula, params, _ = LEGACY[name]
    elif name == "two_compact_wh":
        formula, params = TWO, TWO_P
    else:
        formula, params = MIXED[int(name[5:])]
    pts = _points()
    out = _run([dict(formula=formula, params=params, points=pts)], tmp_path)[0]
    assert out["ok"], out
    f, g, H = _ref(formula, params, pts)
    sc = 1.0 + np.abs(f)
    assert np.max(np.abs(np.array(out["f"]) - f) / sc) < 1e-12
    assert np.max(np.abs(np.array(out["g"]) - g) / (1 + np.abs(g))) < 1e-10
    assert np.max(np.abs(np.array(out["H"]) - H) / (1 + np.abs(H))) < 1e-9


def _scenes_sdf(key):
    """Текст `return …;` функции sdf ветви SCENE_<key> в scenes.slang."""
    src = (ROOT / "gpu" / "shaders" / "scenes.slang").read_text(encoding="utf-8")
    m = re.search(rf"#elif SCENE == SCENE_{key}\n(.*?)(?=\n#elif|\n#endif)", src, re.S)
    body = m.group(1)
    return re.search(r"T sdf\(T x, T y, T z, T w\)\s*\{\s*return (.*?);\s*\}", body, re.S).group(1)


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_slang_matches_scenes(tmp_path, name):
    formula, params, defs = LEGACY[name]
    out = _run([dict(formula=formula, params=params, slang_params=defs or {"_": "_"})], tmp_path)[0]
    assert out["ok"], out
    assert out["slang"] == _scenes_sdf(SCENE_KEY[name])


def test_errors_match(tmp_path):
    cases = [dict(formula=f, params={}) for f, _ in ERRORS]
    out = _run(cases, tmp_path)
    for (formula, word), r in zip(ERRORS, out):
        assert not r["ok"] and word in r["error"], (formula, r)
        with pytest.raises(E.ExprError) as ex:
            E.parse(formula, [])
        assert str(ex.value) == r["error"] and ex.value.pos == r["pos"], (formula, str(ex.value), r)
