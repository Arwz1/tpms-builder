"""Sandboxed evaluator for user-typed ``f(x, y, z)`` expressions.

Expression-driven grading lets a user write ``4 + 3*sin(x/10)`` in the UI and have it
drive cell size across the part. That string arrives from a text box, so it is compiled
against an AST allowlist rather than handed to bare ``eval``: only literals, names we
publish, arithmetic, comparisons and calls to the whitelisted functions survive. No
attribute access, no subscripting, no imports, no dunders, no lambdas.

Everything is vectorised over numpy arrays, so one compiled expression evaluates a whole
slab at once.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Mapping

import numpy as np


class ExpressionError(ValueError):
    """Raised for an expression that will not compile or will not evaluate."""


# Callables an expression may use. Every one is elementwise over arrays.
SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "cbrt": np.cbrt,
    "abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil, "round": np.round,
    "min": np.minimum, "max": np.maximum,
    "clamp": lambda v, lo, hi: np.clip(v, lo, hi),
    "clip": np.clip,
    "lerp": lambda a, b, t: a + (b - a) * t,
    "smoothstep": lambda e0, e1, v: (
        lambda t: t * t * (3.0 - 2.0 * t)
    )(np.clip((v - e0) / np.where((e1 - e0) == 0, 1e-12, (e1 - e0)), 0.0, 1.0)),
    "hypot": np.hypot,
    "pow": np.power,
    "mod": np.mod,
    "where": np.where,
}

SAFE_CONSTANTS: dict[str, float] = {
    "pi": float(np.pi),
    "e": float(np.e),
    "tau": float(2.0 * np.pi),
}

# Variables the grading feature supplies. Documented in the UI tooltip.
VARIABLES = ("x", "y", "z", "r", "d", "u", "v", "w")

VARIABLE_HELP = {
    "x": "world X coordinate",
    "y": "world Y coordinate",
    "z": "world Z coordinate",
    "r": "distance from the domain centre axis",
    "d": "distance to the part surface (negative inside)",
    "u": "normalised X across the bounding box, 0 to 1",
    "v": "normalised Y across the bounding box, 0 to 1",
    "w": "normalised Z across the bounding box, 0 to 1",
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.Not,
    ast.Tuple,
)


def _validate(tree: ast.AST, allowed_names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"'{type(node).__name__}' is not allowed in an expression"
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only direct calls to known functions are allowed")
            if node.func.id not in SAFE_FUNCTIONS:
                raise ExpressionError(f"unknown function '{node.func.id}'")
            if node.keywords:
                raise ExpressionError("keyword arguments are not supported")

        elif isinstance(node, ast.Name):
            if node.id not in allowed_names:
                known = ", ".join(sorted(allowed_names))
                raise ExpressionError(f"unknown name '{node.id}'. Available: {known}")

        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise ExpressionError("only numeric constants are allowed")


def compile_expression(
    source: str, variables: tuple[str, ...] = VARIABLES
) -> Callable[..., np.ndarray]:
    """Compile ``source`` into a callable taking the named variables as keywords.

    Raises :class:`ExpressionError` with a message fit to show in the UI.
    """
    if not source or not source.strip():
        raise ExpressionError("expression is empty")

    allowed_names = set(variables) | set(SAFE_FUNCTIONS) | set(SAFE_CONSTANTS)

    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"syntax error: {exc.msg}") from exc

    _validate(tree, allowed_names)

    try:
        code = compile(tree, filename="<expression>", mode="eval")
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - parse catches most
        raise ExpressionError(f"could not compile: {exc}") from exc

    # No builtins at all. Every name the expression can reach is in this dict.
    base_globals: dict[str, Any] = {"__builtins__": {}}
    base_globals.update(SAFE_FUNCTIONS)
    base_globals.update(SAFE_CONSTANTS)

    def evaluator(**kwargs: Any) -> np.ndarray:
        missing = [v for v in variables if v not in kwargs]
        if missing:
            raise ExpressionError(f"missing values for: {', '.join(missing)}")
        try:
            # numpy warnings are suppressed rather than allowed to fire. Emitting one
            # sends numpy through the warnings machinery, which reaches for
            # ``__builtins__`` — empty here by design — and the resulting
            # KeyError('__import__') masks the real problem. Silencing them lets
            # log(0) return -inf and sqrt(-1) return nan, which the finiteness check
            # below reports accurately.
            with np.errstate(all="ignore"):
                result = eval(code, base_globals, dict(kwargs))  # noqa: S307 - AST-gated
        except ZeroDivisionError as exc:
            raise ExpressionError("division by zero") from exc
        except Exception as exc:
            raise ExpressionError(f"evaluation failed: {exc}") from exc

        return np.asarray(result, dtype=np.float32)

    evaluator.source = source  # type: ignore[attr-defined]
    return evaluator


def evaluate(source: str, **variables: Any) -> np.ndarray:
    """Compile and evaluate in one call. For one-shot use and tests."""
    fn = compile_expression(source, tuple(variables.keys()))
    return fn(**variables)


def validate_expression(source: str, variables: tuple[str, ...] = VARIABLES) -> str | None:
    """Check an expression without running it. Returns an error message, or ``None``.

    The UI calls this on every keystroke to colour the input box, so it must be cheap
    and must never raise.
    """
    try:
        fn = compile_expression(source, variables)
        # Compile-time validation misses runtime problems such as sqrt of a negative,
        # so probe with a scalar of each variable.
        probe = {v: np.float32(1.0) for v in variables}
        result = fn(**probe)
        if not np.all(np.isfinite(result)):
            return "expression produced a non-finite value"
    except ExpressionError as exc:
        return str(exc)
    except Exception as exc:  # pragma: no cover - defensive, UI must not crash
        return f"invalid expression: {exc}"
    return None
