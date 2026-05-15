from __future__ import annotations

import json
import time
import sympy as sp

from agent.tools.decorator import tool


@tool(
    "math_tool",
    (
        "Solve math expressions and equations with SymPy. "
        "Input formats: "
        "'eval: <expr>' (default), "
        "'simplify: <expr>', "
        "'factor: <expr>', "
        "'expand: <expr>', "
        "'solve: <equation or expr>', "
        "'diff: <expr>, <var>', "
        "'integrate: <expr>, <var>'."
    ),
)
def math_tool(tool_input: str) -> str:
    started = time.perf_counter()

    def ok_payload(op: str, expr: str, value, **extra) -> str:
        payload = {
            "ok": True,
            "tool": "math_tool",
            "input": {"raw": tool_input},
            "data": {"op": op, "expr": expr, "val": value, **extra},
            "meta": {"latency_ms": int((time.perf_counter() - started) * 1000)},
            "error": None,
        }
        return json.dumps(payload, ensure_ascii=False)

    def err_payload(code: str, message: str) -> str:
        payload = {
            "ok": False,
            "tool": "math_tool",
            "input": {"raw": tool_input},
            "data": None,
            "meta": {"latency_ms": int((time.perf_counter() - started) * 1000)},
            "error": {"code": code, "message": message},
        }
        return json.dumps(payload, ensure_ascii=False)

    raw = (tool_input or "").strip()
    if not raw:
        return err_payload("EMPTY_INPUT", "Empty input. Example: `solve: x^2 - 5*x + 6 = 0; x`")

    # Keep parser safe and predictable: only expose a small math surface.
    symbols = {n: sp.Symbol(n) for n in ("x", "y", "z", "t", "a", "b", "c")}
    safe_locals = {
        **symbols,
        "pi": sp.pi,
        "e": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "abs": sp.Abs,
    }

    def to_expr(text: str):
        return sp.sympify(text.strip().replace("^", "**"), locals=safe_locals)

    def parse_mode_and_body(text: str) -> tuple[str, str]:
        if ":" in text:
            mode, body = text.split(":", 1)
            return mode.strip().lower(), body.strip()
        return "eval", text

    def parse_solve_body(body: str):
        # Supported:
        #   solve: x^2 - 5*x + 6 = 0; x
        #   solve: x^2 - 5*x + 6 = 0
        #   solve: x^2 - 5*x + 6; x
        eq_part, var_part = body, ""
        if ";" in body:
            eq_part, var_part = body.split(";", 1)
            eq_part, var_part = eq_part.strip(), var_part.strip()

        if "=" in eq_part:
            lhs, rhs = eq_part.split("=", 1)
            equation = sp.Eq(to_expr(lhs), to_expr(rhs))
        else:
            equation = sp.Eq(to_expr(eq_part), 0)

        target_var = None
        if var_part:
            target_var = sp.Symbol(var_part)
        return equation, target_var

    try:
        mode, body = parse_mode_and_body(raw)

        if mode == "eval":
            expr = to_expr(body)
            return ok_payload("eval", str(expr), str(sp.N(expr, 15)))

        if mode == "simplify":
            expr = to_expr(body)
            return ok_payload("simplify", str(expr), str(sp.simplify(expr)))

        if mode == "factor":
            expr = to_expr(body)
            return ok_payload("factor", str(expr), str(sp.factor(expr)))

        if mode == "expand":
            expr = to_expr(body)
            return ok_payload("expand", str(expr), str(sp.expand(expr)))

        if mode == "solve":
            eq, target_var = parse_solve_body(body)
            free_syms = sorted(eq.free_symbols, key=lambda s: s.name)

            if not free_syms:
                residual = sp.simplify(eq.lhs - eq.rhs)
                return ok_payload("solve", str(eq), str(residual), note="No variable to solve")

            if target_var is not None:
                solutions = sp.solve(eq, target_var, dict=True)
            else:
                solutions = sp.solve(eq, *free_syms, dict=True)
            if not solutions:
                return ok_payload("solve", str(eq), [], vars=[str(s) for s in free_syms])
            norm_solutions = [{str(k): str(v) for k, v in row.items()} for row in solutions]
            return ok_payload(
                "solve",
                str(eq),
                norm_solutions,
                vars=[str(s) for s in free_syms],
                var=(str(target_var) if target_var is not None else None),
            )

        if mode in {"diff", "integrate"}:
            # Format: "<expr>, <var>" (var optional; defaults to x)
            parts = [p.strip() for p in body.split(",", 1)]
            expr = to_expr(parts[0])
            var = sp.Symbol(parts[1]) if len(parts) == 2 and parts[1] else symbols["x"]
            val = sp.diff(expr, var) if mode == "diff" else sp.integrate(expr, var)
            return ok_payload(mode, str(expr), str(val), var=str(var))

        return err_payload("UNKNOWN_MODE", "Use eval|simplify|factor|expand|solve|diff|integrate.")

    except Exception as e:
        return err_payload("MATH_EXECUTION_FAILED", str(e))
