import io, contextlib, builtins

_SAFE_BUILTINS = {
    k: getattr(builtins, k)
    for k in (
        "print","range","len","sum","max","min","abs","round","sorted",
        "list","dict","set","tuple","str","int","float","bool","enumerate",
        "zip","map","filter","isinstance","type","repr","chr","ord"
    )
    if hasattr(builtins, k)
}


def run_code(code: str) -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": _SAFE_BUILTINS}, {})
        out = buf.getvalue()
        return out.strip() if out.strip() else "✅ Code executed successfully (no output)."
    except Exception as exc:
        return f"❌ Error: {exc}"