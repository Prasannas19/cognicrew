"""TASK 4 (v11, Group 2): backend-authored math operations — parameter only.

SAFETY CONTRACT (work order STEP 4B):
- The LLM NEVER executes code and NEVER writes code that runs here. It may
  only EXTRACT structured parameters (coordinates, angles, factors) from the
  verified evidence; this module computes everything itself through fixed,
  backend-authored SymPy calls.
- No eval(), no exec(), no dynamic code construction from model output.
- Every public function accepts plain numbers / [x, y] pairs ONLY and returns
  exact SymPy results (plus decimal approximations) as plain dicts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

Point = tuple[sympy.Expr, sympy.Expr]


class MathOpError(ValueError):
    """Raised when extracted parameters are malformed (caller falls back)."""


def _to_xy(value: Any, name: str) -> Point:
    """Validate ONE point parameter: [x, y] pair of finite real numbers."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            x = sympy.Rational(str(value[0]))
            y = sympy.Rational(str(value[1]))
        except (TypeError, ValueError, sympy.SympifyError) as exc:
            raise MathOpError(f"{name} is not a numeric [x, y] pair: {value!r}") from exc
        return x, y
    raise MathOpError(f"{name} must be an [x, y] pair, got {value!r}")


def _to_num(value: Any, name: str) -> sympy.Expr:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise MathOpError(f"{name} must be a number, got {value!r}")
    try:
        return sympy.Rational(str(value))
    except (TypeError, ValueError, sympy.SympifyError) as exc:
        raise MathOpError(f"{name} must be a number, got {value!r}") from exc


def _pt_str(p: Point) -> str:
    x, y = p
    return f"({x}, {y})"


def _exact_str(expr) -> str:
    """Compact exact string: 6*sqrt(2), -1, 9/2 ..."""
    s = sympy.sstr(sympy.nsimplify(expr))
    return s


def _fmt(expr) -> dict[str, Any]:
    """Standard result payload: exact + unicode pretty + decimal."""
    pretty = sympy.pretty(expr, use_unicode=True).strip()
    dec = sympy.N(expr, 10)
    return {"exact": _exact_str(expr), "pretty": pretty, "decimal": float(dec) if dec.is_number else None}


# ---------------------------------------------------------------------------
# Operations (work order STEP 4B list)
# ---------------------------------------------------------------------------

def compute_distance(p1: Any, p2: Any) -> dict[str, Any]:
    a, b = _to_xy(p1, "p1"), _to_xy(p2, "p2")
    d = sympy.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return {"operation": "distance", "p1": _pt_str(a), "p2": _pt_str(b),
            "formula": "d = √((x₂−x₁)² + (y₂−y₁)²)", "distance": _fmt(d)}


def compute_slope(p1: Any, p2: Any) -> dict[str, Any]:
    a, b = _to_xy(p1, "p1"), _to_xy(p2, "p2")
    if b[0] == a[0]:
        return {"operation": "slope", "p1": _pt_str(a), "p2": _pt_str(b),
                "formula": "m = (y₂−y₁)/(x₂−x₁)",
                "slope": {"exact": "undefined (vertical line)", "pretty": "undefined", "decimal": None}}
    m = sympy.Rational(b[1] - a[1], b[0] - a[0])
    return {"operation": "slope", "p1": _pt_str(a), "p2": _pt_str(b),
            "formula": "m = (y₂−y₁)/(x₂−x₁)", "slope": _fmt(m)}


def compute_line_equation(p1: Any, p2: Any) -> dict[str, Any]:
    """Two-point form → ax + by + c = 0 (integer coefficients) + y = mx + k."""
    a, b = _to_xy(p1, "p1"), _to_xy(p2, "p2")
    x, y = sympy.symbols("x y")
    if b[0] == a[0]:  # vertical line x = k
        k = a[0]
        return {"operation": "line_equation", "p1": _pt_str(a), "p2": _pt_str(b),
                "formula": "x = k (vertical line: x₂ = x₁)",
                "standard_form": f"x − {_exact_str(k)} = 0",
                "a": int(k), "b": 0, "c": int(-k),
                "slope_intercept": None,
                "slope": {"exact": "undefined", "pretty": "undefined", "decimal": None}}
    m = sympy.Rational(b[1] - a[1], b[0] - a[0])
    # y = m(x − x₁) + y₁  →  m·x − y + (y₁ − m·x₁) = 0
    c_const = sympy.expand(a[1] - m * a[0])
    coeff_a, coeff_b, coeff_c = m, sympy.Integer(-1), sympy.Rational(c_const)
    # scale to integer coefficients when exact rationals
    if all(v.q == 1 for v in (coeff_a, coeff_b, coeff_c)):  # type: ignore[attr-defined]
        ints = [int(coeff_a), int(coeff_b), int(coeff_c)]
        g = sympy.igcd(abs(ints[0]), abs(ints[1]), abs(ints[2])) or 1
        ints = [i // g for i in ints]
        if ints[0] < 0 or (ints[0] == 0 and ints[1] < 0):
            ints = [-i for i in ints]
        std_expr = sympy.Integer(ints[0]) * x + sympy.Integer(ints[1]) * y + sympy.Integer(ints[2])
        std = {"a": ints[0], "b": ints[1], "c": ints[2], "string": _uni(f"{sympy.sstr(std_expr)} = 0")}
    else:
        std = {"a": _exact_str(coeff_a), "b": _exact_str(coeff_b), "c": _exact_str(coeff_c),
               "string": _uni(f"{_exact_str(coeff_a)}x + {_exact_str(coeff_b)}y + {_exact_str(coeff_c)} = 0")}
    intercept = a[1] - m * a[0]
    return {"operation": "line_equation", "p1": _pt_str(a), "p2": _pt_str(b),
            "formula": "y − y₁ = m(x − x₁),  m = (y₂−y₁)/(x₂−x₁)",
            "standard_form": std,
            "slope_intercept": _uni(f"y = {_linear_term(m)} + {_exact_str(intercept)}"),
            "slope": _fmt(m)}


def _uni(s: str) -> str:
    """Display form: unicode minus (STEP 4D notation); exact keys stay ASCII."""
    return s.replace("-", "−")


def _linear_term(m) -> str:
    """'x', '−x', '2x', '(1/2)x' … for y = <term> + k display."""
    if m == 1:
        return "x"
    if m == -1:
        return "−x"
    return _uni(f"{_exact_str(m)}x")


def compute_centroid(a: Any, b: Any, c: Any) -> dict[str, Any]:
    A, B, C = _to_xy(a, "a"), _to_xy(b, "b"), _to_xy(c, "c")
    gx = (A[0] + B[0] + C[0]) / 3
    gy = (A[1] + B[1] + C[1]) / 3
    return {"operation": "centroid", "a": _pt_str(A), "b": _pt_str(B), "c": _pt_str(C),
            "formula": "G = ((x₁+x₂+x₃)/3, (y₁+y₂+y₃)/3)",
            "centroid": {"x": _fmt(gx), "y": _fmt(gy),
                         "point": f"({_exact_str(gx)}, {_exact_str(gy)})"}}


def compute_collinear(p1: Any, p2: Any, p3: Any) -> dict[str, Any]:
    a, b, c = _to_xy(p1, "p1"), _to_xy(p2, "p2"), _to_xy(p3, "p3")
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return {"operation": "collinear", "p1": _pt_str(a), "p2": _pt_str(b), "p3": _pt_str(c),
            "formula": "½|x₁(y₂−y₃) + x₂(y₃−y₁) + x₃(y₁−y₂)| = 0 ⇔ collinear",
            "area_signed_half": _fmt(sympy.Abs(area2) / 2),
            "collinear": bool(area2 == 0)}


def translate_point(p: Any, dx: Any, dy: Any) -> dict[str, Any]:
    P = _to_xy(p, "p")
    tx, ty = _to_num(dx, "dx"), _to_num(dy, "dy")
    return {"operation": "translate_point", "p": _pt_str(P), "dx": _exact_str(tx), "dy": _exact_str(ty),
            "formula": "(x, y) → (x + dx, y + dy)",
            "result": _fmt_point((P[0] + tx, P[1] + ty))}


def translate_triangle(a: Any, b: Any, c: Any, dx: Any, dy: Any) -> dict[str, Any]:
    res = [translate_point(v, dx, dy) for v in (a, b, c)]
    return {"operation": "translate_triangle", "dx": res[0]["dx"], "dy": res[0]["dy"],
            "formula": "T(dx, dy): (x, y) → (x + dx, y + dy)",
            "a": res[0]["result"], "b": res[1]["result"], "c": res[2]["result"]}


def reflect_point(p: Any, axis: str) -> dict[str, Any]:
    P = _to_xy(p, "p")
    ax = str(axis).strip().lower()
    if ax in ("x", "x-axis", "x axis"):
        return {"operation": "reflect_point", "axis": "x", "p": _pt_str(P),
                "formula": "reflection through x-axis: (x, y) → (x, −y)",
                "result": _fmt_point((P[0], -P[1]))}
    if ax in ("y", "y-axis", "y axis"):
        return {"operation": "reflect_point", "axis": "y", "p": _pt_str(P),
                "formula": "reflection through y-axis: (x, y) → (−x, y)",
                "result": _fmt_point((-P[0], P[1]))}
    if ax in ("origin", "0"):
        return {"operation": "reflect_point", "axis": "origin", "p": _pt_str(P),
                "formula": "reflection through origin: (x, y) → (−x, −y)",
                "result": _fmt_point((-P[0], -P[1]))}
    raise MathOpError(f"axis must be 'x', 'y' or 'origin', got {axis!r}")


def rotate_point(p: Any, angle_deg: Any, center: Any = (0, 0)) -> dict[str, Any]:
    P = _to_xy(p, "p")
    O = _to_xy(center if center is not None else (0, 0), "center")
    ang = _to_num(angle_deg, "angle_deg")
    rad = sympy.rad(ang)
    cos, sin = sympy.cos(rad), sympy.sin(rad)
    rx = O[0] + (P[0] - O[0]) * cos - (P[1] - O[1]) * sin
    ry = O[1] + (P[0] - O[0]) * sin + (P[1] - O[1]) * cos
    return {"operation": "rotate_point", "p": _pt_str(P), "center": _pt_str(O),
            "angle_deg": _exact_str(ang),
            "formula": "x' = cx + (x−cx)cosθ − (y−cy)sinθ;  y' = cy + (x−cx)sinθ + (y−cy)cosθ",
            "result": _fmt_point((sympy.nsimplify(rx), sympy.nsimplify(ry)))}


def scale_point(p: Any, factor: Any) -> dict[str, Any]:
    P = _to_xy(p, "p")
    k = _to_num(factor, "factor")
    return {"operation": "scale_point", "p": _pt_str(P), "factor": _exact_str(k),
            "formula": "uniform scaling: (x, y) → (k·x, k·y)",
            "result": _fmt_point((P[0] * k, P[1] * k))}


def _fmt_point(p: Point) -> dict[str, Any]:
    return {"x": _fmt(p[0]), "y": _fmt(p[1]), "point": f"({_exact_str(p[0])}, {_exact_str(p[1])})"}


# ---------------------------------------------------------------------------
# Dispatch: fixed op -> fixed function with fixed parameter names (STEP 4C).
# The model can only NAME one of these; there is no way to run anything else.
# ---------------------------------------------------------------------------

_OPERATIONS: dict[str, tuple[Callable[..., dict[str, Any]], tuple[str, ...], tuple[str, ...]]] = {
    "distance": (compute_distance, ("p1", "p2"), ()),
    "slope": (compute_slope, ("p1", "p2"), ()),
    "line_equation": (compute_line_equation, ("p1", "p2"), ()),
    "centroid": (compute_centroid, ("a", "b", "c"), ()),
    "collinear": (compute_collinear, ("p1", "p2", "p3"), ()),
    "translate_point": (translate_point, ("p", "dx", "dy"), ()),
    "translate_triangle": (translate_triangle, ("a", "b", "c", "dx", "dy"), ()),
    "reflect_point": (reflect_point, ("p", "axis"), ()),
    "rotate_point": (rotate_point, ("p", "angle_deg"), ("center",)),
    "scale_point": (scale_point, ("p", "factor"), ()),
}


def run_operation(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a whitelisted backend operation with validated parameters.

    This is the ONLY entry point the extraction phase may reach. `operation`
    must be a key of _OPERATIONS and every named parameter is validated by
    the function itself — model text never reaches an interpreter.
    """
    if not isinstance(operation, str) or not isinstance(params, dict):
        raise MathOpError("operation must be a string and params a dict")
    entry = _OPERATIONS.get(operation.strip().lower())
    if entry is None:
        raise MathOpError(f"unknown operation {operation!r}")
    fn, required, optional = entry
    missing = [k for k in required if k not in params]
    if missing:
        raise MathOpError(f"operation {operation!r} missing parameters: {missing}")
    extra = set(params) - set(required) - set(optional)
    if extra:
        raise MathOpError(f"operation {operation!r} got unexpected parameters: {sorted(extra)}")
    return fn(**params)
