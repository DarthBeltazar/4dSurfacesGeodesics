"""Байткод многообразия для интерпретатора в шейдерах режима редактирования — проба перед DD-011
(docs/design/concept-standalone-nodes.md, п. 3). Шейдер — scenes.slang, SCENE_DYNAMIC с MF_PROG.

Программа — строки float4 в хвосте буфера камеры с PROG_BASE: строка 0 — (число инструкций, число регистров, 0, 0),
затем по строке на инструкцию: (op + 256·dst + 65536·a, b + 256·c, k, 0). Целые хранятся точными значениями float
(< 2²⁴). Регистры 0…3 на входе — x, y, z, w; результат — регистр 0 после последней инструкции.

`Prog` строит программу в SSA и распределяет регистры по времени жизни значений; `evaluate` исполняет программу
операциями jet.ops (массивы jnp, Jet7, Jet15 …) — эталон для сверки с формулой фигуры."""
import numpy as np

from .jet import ops

OPS = {"const": 1, "add": 2, "sub": 3, "mul": 4, "neg": 6, "addk": 7, "mulk": 8, "recip": 9, "sqrt": 10,
       "exp": 11, "log": 12, "cosh": 13, "sin": 14, "cos": 15, "sqrtg": 16, "selgt": 17, "sq": 19, "sinh": 20}
NARGS = {"const": 0, "add": 2, "sub": 2, "mul": 2, "neg": 1, "addk": 1, "mulk": 1, "recip": 1, "sqrt": 1,
         "exp": 1, "log": 1, "cosh": 1, "sin": 1, "cos": 1, "sqrtg": 1, "selgt": 3, "sq": 1, "sinh": 1}
LENGTH_EPS2 = 1e-36                # как jlength* в jet.slang


class Prog:
    """Программа в SSA: значения — целые id, 0…3 — x, y, z, w."""

    def __init__(self):
        self.code = []             # (op, (аргументы…), k)

    def _emit(self, op, args=(), k=0.0):
        assert len(args) == NARGS[op]
        self.code.append((op, tuple(args), float(k)))
        return 3 + len(self.code)

    def __getattr__(self, op):
        if op not in OPS:
            raise AttributeError(op)
        return lambda *args, k=0.0: self._emit(op, args, k)

    # Деления и smin в шейдере нет: раскладываются здесь (в шейдере только унарные функции, a ± b, a·b и выбор)
    def div(self, a, b):
        return self.mul(a, self.recip(b))

    def smin(self, a, b, k):
        """M6, как jsmin в jet.slang: lo − k·log(1 + exp((lo − hi)/k))."""
        d = self.sub(a, b)
        lo, hi = self.selgt(b, a, d, k=0.0), self.selgt(a, b, d, k=0.0)
        t = self.exp(self.mulk(self.sub(lo, hi), k=1.0 / k))
        return self.sub(lo, self.mulk(self.log(self.addk(t, k=1.0)), k=k))

    def length(self, *v):
        s = self.sq(v[0])
        for a in v[1:]:
            s = self.add(s, self.sq(a))
        return self.sqrtg(s)

    def assemble(self, result):
        """Строки float32 (1 + n, 4); регистр результата — 0 (последняя инструкция копирует его при необходимости)."""
        code = list(self.code)
        last = {}                  # последнее использование значения
        for i, (_, args, _) in enumerate(code):
            for a in args:
                last[a] = i
        reg, free, nreg = {v: v for v in range(4)}, [], 4
        for v in range(4):
            if v not in last:
                free.append(v)
        rows = []
        for i, (op, args, k) in enumerate(code):
            src = [reg[a] for a in args]
            for a in set(args):    # регистры аргументов освобождаются до выбора регистра результата
                if last[a] == i and a != result:
                    free.append(reg[a])
            v = 4 + i
            if free:
                free.sort()
                reg[v] = free.pop(0)
            else:
                reg[v], nreg = nreg, nreg + 1
            src += [0] * (3 - len(src))
            rows.append((OPS[op] + 256 * reg[v] + 65536 * src[0], src[1] + 256 * src[2], k, 0.0))
            if v not in last and v != result:
                free.append(reg[v])
        if reg[result] != 0:
            rows.append((OPS["mulk"] + 256 * 0 + 65536 * reg[result], 0, 1.0, 0.0))
        R = np.zeros((1 + len(rows), 4))
        R[0, :2] = len(rows), nreg
        R[1:] = rows
        assert np.all(R[1:, :2] < 2 ** 24)
        return R.astype(np.float32)


def evaluate(rows, x, y, z, w):
    """Исполняет программу из строк (как шейдер), операциями jet.ops."""
    rows = np.asarray(rows, dtype=np.float64)
    n, nreg = int(rows[0, 0]), int(rows[0, 1])
    r = [x, y, z, w] + [0.0] * max(nreg - 4, 0)
    names = {v: k for k, v in OPS.items()}
    for ins in rows[1:1 + n]:
        e0, e1, k = int(ins[0]), int(ins[1]), float(ins[2])
        op, d, a, b, c = names[e0 & 255], (e0 >> 8) & 255, e0 >> 16, e1 & 255, e1 >> 8
        A, B, C = r[a], r[b], r[c]
        r[d] = {
            "const": lambda: k + 0.0 * ops.value(x),
            "add": lambda: A + B, "sub": lambda: A - B, "mul": lambda: A * B,
            "neg": lambda: -A, "addk": lambda: A + k, "mulk": lambda: A * k, "recip": lambda: 1.0 / A,
            "sqrt": lambda: ops.sqrt(A), "exp": lambda: ops.exp(A), "log": lambda: ops.log(A),
            "cosh": lambda: ops.cosh(A), "sinh": lambda: ops.sinh(A), "sin": lambda: ops.sin(A), "cos": lambda: ops.cos(A),
            "sqrtg": lambda: ops.sqrt(ops.where(ops.value(A) < LENGTH_EPS2, LENGTH_EPS2, A)),
            "selgt": lambda: ops.where(ops.value(C) > k, A, B),
            "sq": lambda: A * A,
        }[op]()
    return r[0]


def compact_h(P: Prog, rho, one, b, rho0, W):
    """M11: W²·ψ((ρ − b)/(ρ₀ − b)) — как shapes.compact_h и compact_h в scenes.slang."""
    q = P.addk(P.mulk(rho, k=-1.0 / (rho0 - b)), k=1.0 + b / (rho0 - b))
    qs = P.selgt(q, one, q, k=1e-2)
    e = P.exp(P.addk(P.neg(P.recip(qs)), k=1.0))
    psi = P.selgt(P.addk(P.neg(e), k=1.0), one, q, k=1e-2)
    return P.mulk(psi, k=W * W)


def two_compact_program(c1=(-4.0, 0.0, 0.0), c2=(4.0, 0.0, 0.0), b=1.0, rho0=3.0, W=1.5):
    """M12 (shapes.two_compact_wormholes) в байткоде: f = w² − (h(r₁) + h(r₂) − W²)."""
    P = Prog()
    one = P.const(k=1.0)
    hs = []
    for c in (c1, c2):
        r = P.length(*(P.addk(i, k=-float(ci)) for i, ci in enumerate(c)))
        hs.append(compact_h(P, r, one, b, rho0, W))
    f = P.sub(P.sq(3), P.addk(P.add(*hs), k=-W * W))
    return P.assemble(f)


def s3_program(R=1.0):
    """S³ (shapes.sphere3) в байткоде: f = |p| − R — короткая программа для замера цены инструкции."""
    P = Prog()
    return P.assemble(P.addk(P.length(0, 1, 2, 3), k=-R))
