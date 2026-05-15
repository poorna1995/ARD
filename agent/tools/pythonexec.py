import sys
import io
import traceback

@tool("python_exec", "function to execute Python code. Input: raw Python source string.")
def python_exec(code: str) -> str:
    """
    Executes Python in a restricted namespace.
    Returns stdout + stderr, or a traceback on failure.
    """
    namespace = {
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "json": __import__("json"),
        "math": __import__("math"),
        # Explicitly block dangerous builtins
        "__builtins__": {
            k: __builtins__[k]
            for k in ("print", "len", "range", "enumerate",
                      "zip", "map", "filter", "sorted",
                      "min", "max", "sum", "abs", "round",
                      "isinstance", "type", "str", "int",
                      "float", "list", "dict", "set", "tuple")
        },
    }
    stdout_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        exec(compile(code, "<agent>", "exec"), namespace)
    except Exception:
        return f"Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = sys.__stdout__

    output = stdout_capture.getvalue().strip()
    return output or "(code ran successfully, no output)"