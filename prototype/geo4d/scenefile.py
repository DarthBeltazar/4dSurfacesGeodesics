"""Файл сцены (DD-008): JSON ↔ сцена режима B (`pathtrace.Scene`) + стартовая камера.

    {"manifold": {"kind": "two_compact_wh", "params": {"b": 1, "rho0": 3, "W": 1.5, "c1": [-4, 0, 0], "c2": [4, 0, 0]}},
     "env": "procedural" | "none",
     "objects": [{"name": "…", "kind": "ball4", "center": [x, y, z, w], "r": 0.8, "on_M": false,
                  "material": {"kind": "diffuse" | "metal" | "emit", "albedo": [r, g, b], "rough": 0.3, "emit": [r, g, b]}},
                 {"kind": "torus", "center": […], "R": 1.2, "r": 0.3, …}, {"kind": "plane", "axis": 2, "offset": -3.5, "sign": 1, …}],
     "camera": {"x": […], "f": […], "up": […], "right": […] (необязательно), "fov": 75}, "speed": 2.0}

NEE-источники не задаются — правило M34 (`pathtrace.nee_lights`). Центры объектов с on_M и камера проецируются на M
(M3); так при смене многообразия из окна (хост сохраняет JSON с текущей камерой) сцена остаётся согласованной."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np

from . import gpu, shapes as S
from . import pathtrace as PT
from .objects import Material, ball4, torus_in_sheet, halfspace

SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"

# вид многообразия → (конструктор geo4d.shapes, параметры по умолчанию)
MANIFOLDS = {
    "hyperplane": (S.hyperplane, {}),
    "S3": (S.sphere3, {"R": 1.0}),
    "spheritorus": (S.spheritorus, {"R": 1.0, "r": 0.4}),
    "torisphere": (S.torisphere, {"R": 1.0, "r": 0.4}),
    "tiger": (S.tiger, {"R1": 1.2, "R2": 0.8, "r": 0.3}),
    "ellis": (S.ellis, {"b": 1.0}),
    "compact_wh": (S.compact_wormhole, {"b": 1.0, "rho0": 3.0, "W": 1.5}),
    "two_compact_wh": (S.two_compact_wormholes, {"c1": [-4.0, 0.0, 0.0], "c2": [4.0, 0.0, 0.0], "b": 1.0, "rho0": 3.0,
                                                 "W": 1.5}),
}
COMPACT_SPACES = ("S3", "spheritorus", "torisphere", "tiger")
ELLIS_ELL_MAX = 12.0


def check_params(kind, p):
    """Параметры, для которых формулы фигуры верны; иначе ValueError (окно покажет сообщение и оставит старую сцену)."""
    pos = lambda *ks: all(p[k] > 0 for k in ks)
    ok, why = True, ""
    if kind == "S3":
        ok, why = pos("R"), "R > 0"
    elif kind in ("spheritorus", "torisphere"):
        ok, why = pos("R", "r") and p["r"] < p["R"], "0 < r < R"
    elif kind == "tiger":
        ok, why = pos("R1", "R2", "r") and p["r"] < p["R1"] and p["r"] < p["R2"], "0 < r < R1, R2"
    elif kind == "ellis":
        ok, why = pos("b"), "b > 0"
    elif kind in ("compact_wh", "two_compact_wh"):
        ok, why = pos("b", "rho0", "W") and p["b"] < p["rho0"], "0 < b < ρ₀, W > 0"
        if ok and kind == "two_compact_wh":
            ok, why = np.linalg.norm(np.subtract(p["c1"], p["c2"])) > 2 * p["rho0"], "|c₁ − c₂| > 2ρ₀"
    if not ok:
        raise ValueError(f"{kind}: недопустимые параметры {p} (нужно {why})")


def make_shape(kind, params=None):
    ctor, dflt = MANIFOLDS[kind]
    p = {**dflt, **(params or {})}
    check_params(kind, p)
    p = {k: tuple(float(a) for a in v) if isinstance(v, (list, tuple)) else float(v) for k, v in p.items()}
    return ctor(**p), p


def _material(m):
    return Material(m.get("kind", "diffuse"), tuple(m.get("albedo", (0.8, 0.8, 0.8))), float(m.get("rough", 0.3)),
                    tuple(m.get("emit", (0.0, 0.0, 0.0))))


def _project(shape, P):
    return np.asarray(S.project_points(shape, np.atleast_2d(np.asarray(P, float))))


def make_object(o, shape):
    mat, name, kind = _material(o.get("material", {})), o.get("name", o["kind"]), o["kind"]
    on_M = bool(o.get("on_M", False))
    if kind == "plane":
        ob = halfspace(int(o["axis"]), float(o["offset"]), mat, float(o.get("sign", 1.0)), name=name)
    else:
        c = np.asarray(o["center"], float)
        if on_M:
            c = _project(shape, c)[0]
        if kind == "ball4":
            ob = ball4(c, float(o["r"]), mat, name=name)
        elif kind == "torus":
            ob = torus_in_sheet(c, float(o["R"]), float(o["r"]), mat, name=name)
        else:
            raise ValueError(f"нет вида объекта {kind}")
    ob.on_M = on_M
    return ob


def build(d, mips=None):
    """dict сцены → (Scene, камера dict(x, f, up, right, fov), скорость). mips — кубмапа фона (иначе процедурная 64)."""
    kind = d["manifold"]["kind"]
    shape, p = make_shape(kind, d["manifold"].get("params"))
    objs = [make_object(o, shape) for o in d.get("objects", [])]
    env = d.get("env", "none" if kind in COMPACT_SPACES else "procedural")
    if env == "none":
        mips = PT.constant_env(0.0)
    else:
        mips = mips or PT.procedural_env()
    from .wormholes import compact_exit, ellis_exit, ellis_escape
    flat, escape, exit_fn = None, None, compact_exit
    meta = dict(scene_id=gpu.SCENE_IDS[shape.name], exit=2)
    if kind == "hyperplane":
        flat = PT.FlatZones((0.0,))
    elif kind == "compact_wh":
        flat = PT.FlatZones((p["W"], -p["W"]), np.zeros((1, 3)), p["rho0"])
    elif kind == "two_compact_wh":
        flat = PT.FlatZones((p["W"], -p["W"]), np.array([p["c1"], p["c2"]], float), p["rho0"])
    elif kind == "ellis":
        exit_fn, escape = partial(ellis_exit, b=p["b"]), ellis_escape(ELLIS_ELL_MAX, p["b"])
        meta.update(exit=1, ell_max=ELLIS_ELL_MAX)
    elif kind in COMPACT_SPACES:
        size = p.get("R", p.get("R1", 1.2) / 1.2)        # длина пути — от размера пространства (8π у фигур по умолчанию)
        meta.update(ell_max=1e9, s_max=8 * np.pi * size)
    sc = PT.Scene(shape, objs, mips, exit_fn, flat, escape=escape, lights=PT.nee_lights(objs, flat), meta=meta)
    c = d.get("camera", {})
    x = _project(shape, c.get("x", [0, 0, 0, 0]))[0]
    cam = dict(x=x, f=np.asarray(c.get("f", [0, 1, 0, 0]), float), up=np.asarray(c.get("up", [0, 0, 1, 0]), float),
               right=np.asarray(c["right"], float) if "right" in c else None, fov=float(c.get("fov", 75.0)))
    return sc, cam, float(d.get("speed", 1.0))


def load(path, mips=None):
    return build(json.loads(Path(path).read_text(encoding="utf-8")), mips)


def resolve(name_or_path):
    """Имя сцены из scenes/ или путь к JSON."""
    p = Path(name_or_path)
    return p if p.suffix == ".json" else SCENES_DIR / f"{name_or_path}.json"


def _l(a):
    return [float(v) for v in np.asarray(a, float).ravel()]


def manifold_of(shape):
    """Фигура → (вид, параметры) файла сцены."""
    kind = {v[0](**v[1]).name: k for k, v in MANIFOLDS.items()}[shape.name]
    return kind, {k: (_l(v) if isinstance(v, (list, tuple, np.ndarray)) else float(v)) for k, v in shape.params.items()
                  if k in MANIFOLDS[kind][1]}


def to_dict(sc: PT.Scene, cam, speed=1.0, env=None):
    """Сцена прототипа → dict файла сцены (объекты — по spec; многообразие — по Shape.name/params)."""
    kind, params = manifold_of(sc.shape)
    objs = []
    for o in sc.objs:
        sp, m = o.spec, o.mat
        e = dict(name=o.name, kind=sp[0])
        if sp[0] == "ball4":
            e.update(center=_l(sp[1]), r=sp[2])
        elif sp[0] == "torus":
            e.update(center=_l(sp[1]), R=sp[2], r=sp[3])
        else:
            e.update(axis=sp[1], offset=sp[2], sign=sp[3])
        e.update(on_M=bool(o.on_M), material=dict(kind=m.kind, albedo=_l(m.albedo), rough=float(m.rough), emit=_l(m.emit)))
        objs.append(e)
    c = dict(x=_l(cam["x"]), f=_l(cam["f"]), up=_l(cam["up"]), fov=float(cam.get("fov", 75.0)))
    if cam.get("right") is not None:
        c["right"] = _l(cam["right"])
    env = env or ("none" if kind in COMPACT_SPACES else "procedural")
    return dict(manifold=dict(kind=kind, params=params), env=env, objects=objs, camera=c, speed=float(speed))


def save(path, d):
    """JSON с отступами; векторы чисел — в одну строку (файл правят руками)."""
    import re
    text = json.dumps(d, indent=1, ensure_ascii=False)
    text = re.sub(r"\[\s*([-0-9.eE+,\s]+?)\s*\]", lambda m: "[" + ", ".join(m.group(1).split()).replace(",,", ",") + "]",
                  text)
    Path(path).write_text(text + "\n", encoding="utf-8")
