"""Язык выражений нод (DD-011 §2) — эталонный разбор для тестов. Зеркало gpu/src/expr.cpp: та же грамматика, те же
функции и запреты контекстов, те же сообщения об ошибках с позицией. Вычисление — jet.ops (массивы jnp, Jet7 …),
поэтому производные берутся jax.grad / jax.hessian или jet-числами прототипа.

Грамматика: сумма := произведение (('+'|'-') произведение)*; произведение := унарное (('*'|'/') унарное)*;
унарное := '-' унарное | степень; степень := первичное ('^' целое)?; первичное := число | имя | имя.(x|y|z|w) |
имя '(' аргументы ')' | '(' сумма ')'. Имена: x y z w (координаты), pi, параметры ноды."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

from ..jet import ops
from .. import shapes

GEOMETRY, TEXTURE = "geometry", "texture"

# имя → (мин. и макс. число аргументов, индексы однородных аргументов, только для полей)
FUNCTIONS = {"sqrt": (1, 1, (), False), "exp": (1, 1, (), False), "log": (1, 1, (), False), "sin": (1, 1, (), False),
             "cos": (1, 1, (), False), "cosh": (1, 1, (), False), "sinh": (1, 1, (), False),
             "length": (2, 4, (), False), "smin": (3, 3, (2,), True), "compact_h": (4, 4, (1, 2, 3), True)}
NON_SMOOTH = ("abs", "min", "max", "floor", "ceil", "fract", "mod", "step", "smoothstep", "mix", "clamp", "sign", "round")


class ExprError(Exception):
    def __init__(self, msg, pos):
        super().__init__(msg)
        self.pos = pos


@dataclass
class Expr:
    kind: str                       # num | param | coord | neg | add | sub | mul | div | call
    pos: int
    args: list = field(default_factory=list)
    value: float = 0.0
    text: str = ""
    coord: int = 0
    field: bool = False


class _Parser:
    def __init__(self, s, params, ctx):
        self.s, self.params, self.ctx, self.i = s, list(params), ctx, 0

    def fail(self, msg, at):
        raise ExprError(msg, at)

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def peek(self, c):
        self.ws()
        return self.i < len(self.s) and self.s[self.i] == c

    def eat(self, c):
        if not self.peek(c):
            return False
        self.i += 1
        return True

    def node(self, kind, pos, args=()):
        e = Expr(kind, pos, list(args))
        e.field = any(a.field for a in e.args)
        return e

    def parse(self):
        e = self.additive()
        self.ws()
        if self.i != len(self.s):
            self.fail("лишний текст в формуле", self.i)
        return e

    def additive(self):
        a = self.multiplicative()
        while True:
            self.ws()
            p = self.i
            if self.eat("+"):
                a = self.node("add", p, [a, self.multiplicative()])
            elif self.eat("-"):
                a = self.node("sub", p, [a, self.multiplicative()])
            else:
                return a

    def multiplicative(self):
        a = self.unary()
        while True:
            self.ws()
            p = self.i
            if self.eat("*"):
                a = self.node("mul", p, [a, self.unary()])
            elif self.eat("/"):
                a = self.node("div", p, [a, self.unary()])
            else:
                return a

    def unary(self):
        self.ws()
        p = self.i
        if self.eat("-"):
            return self.node("neg", p, [self.unary()])
        return self.power()

    def power(self):
        a = self.primary()
        self.ws()
        p = self.i
        if not self.eat("^"):
            return a
        self.ws()
        q = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        if q == self.i:
            self.fail("после ^ — целое неотрицательное число (степень)", q)
        n = int(self.s[q:self.i])
        if n == 0:
            return Expr("num", p, value=1.0, text="1.0")
        r = a
        for _ in range(n - 1):
            r = self.node("mul", p, [r, a])
        return r

    def primary(self):
        self.ws()
        p = self.i
        s = self.s
        if self.i >= len(s):
            self.fail("формула оборвана", self.i)
        c = s[self.i]
        if c.isdigit() or c == ".":
            q = self.i
            while self.i < len(s) and (s[self.i].isdigit() or s[self.i] == "."):
                self.i += 1
            if self.i < len(s) and s[self.i] in "eE":
                self.i += 1
                if self.i < len(s) and s[self.i] in "+-":
                    self.i += 1
                while self.i < len(s) and s[self.i].isdigit():
                    self.i += 1
            text = s[q:self.i]
            try:
                return Expr("num", p, value=float(text), text=text)
            except ValueError:
                self.fail(f"неверное число «{text}»", q)
        if c.isalpha() or c == "_":
            q = self.i
            while self.i < len(s) and (s[self.i].isalnum() or s[self.i] == "_"):
                self.i += 1
            name = s[q:self.i]
            if (self.i + 1 < len(s) and s[self.i] == "." and s[self.i + 1] in "xyzw"
                    and (self.i + 2 >= len(s) or not s[self.i + 2].isalnum())):
                name += s[self.i:self.i + 2]
                self.i += 2
            self.ws()
            if self.i < len(s) and s[self.i] == "(":
                return self.call(name, p)
            if len(name) == 1 and name in "xyzw":
                return Expr("coord", p, coord="xyzw".index(name), text=name, field=True)
            if name == "pi":
                return Expr("num", p, value=pi, text="3.14159265358979323846")
            if name in self.params:
                return Expr("param", p, text=name)
            self.fail(f"неизвестное имя «{name}»", q)
        if self.eat("("):
            e = self.additive()
            if not self.eat(")"):
                self.fail("нет закрывающей скобки", self.i)
            return e
        self.fail(f"неожиданный символ «{c}»", self.i)

    def call(self, name, p):
        if name in NON_SMOOTH:
            if self.ctx == GEOMETRY:
                self.fail(f"«{name}» негладкая — в геометрии запрещена (M6); используйте smin, length", p)
            self.fail("функции текстур ещё не реализованы (DD-011, шаг 7)", p)
        if name not in FUNCTIONS:
            self.fail(f"неизвестная функция «{name}»", p)
        self.eat("(")
        args = []
        if not self.peek(")"):
            while True:
                args.append(self.additive())
                if not self.eat(","):
                    break
        if not self.eat(")"):
            self.fail(f"нет закрывающей скобки вызова «{name}»", self.i)
        nmin, nmax, uni, field_only = FUNCTIONS[name]
        if not nmin <= len(args) <= nmax:
            self.fail(f"«{name}»: неверное число аргументов ({len(args)})", p)
        for k in uni:
            if args[k].field:
                self.fail(f"«{name}»: аргумент {k + 1} не должен зависеть от x, y, z, w", args[k].pos)
        e = self.node("call", p, args)
        e.text = name
        if field_only and not e.field:
            self.fail(f"«{name}» — только для полей (зависящих от x, y, z, w)", p)
        return e


def parse(src, params=(), ctx=GEOMETRY) -> Expr:
    return _Parser(src, params, ctx).parse()


def evaluate(e: Expr, params: dict, p):
    """Значение выражения в точке p = (x, y, z, w) (числа, массивы jnp или jet-числа прототипа)."""
    k = e.kind
    if k == "num":
        return e.value
    if k == "param":
        return params[e.text]
    if k == "coord":
        return p[e.coord]
    a = [evaluate(x, params, p) for x in e.args]
    if k == "neg":
        return -a[0]
    if k == "add":
        return a[0] + a[1]
    if k == "sub":
        return a[0] - a[1]
    if k == "mul":
        return a[0] * a[1]
    if k == "div":
        return a[0] / a[1]
    n = e.text
    if n == "length":
        return ops.length(*a)
    if n == "smin":
        return ops.smin(a[0], a[1], a[2])
    if n == "compact_h":
        return shapes.compact_h(a[0], a[1], a[2], a[3])
    if not e.field:
        import math
        return getattr(math, n)(a[0])
    return getattr(ops, n)(a[0])
