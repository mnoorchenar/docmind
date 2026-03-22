import ast, math, operator, re

_SAFE_OPS = {
    ast.Add:  operator.add,  ast.Sub: operator.sub,
    ast.Mult: operator.mul,  ast.Div: operator.truediv,
    ast.Pow:  operator.pow,  ast.USub: operator.neg,
    ast.Mod:  operator.mod,  ast.FloorDiv: operator.floordiv,
}
_SAFE_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "int": int, "float": float})


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    if isinstance(node, ast.Call):
        func = node.func.id if isinstance(node.func, ast.Name) else None
        if func in _SAFE_NAMES:
            return _SAFE_NAMES[func](*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    raise ValueError(f"Unsafe expression: {ast.dump(node)}")


def calculate(expr: str) -> str:
    try:
        expr  = re.sub(r"[^0-9+\-*/().,%^ \t\na-zA-Z_]", "", expr).strip()
        tree  = ast.parse(expr, mode="eval")
        val   = _safe_eval(tree.body)
        return f"Result: {val}"
    except Exception as exc:
        return f"Calculation error: {exc}"