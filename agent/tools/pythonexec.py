from __future__ import annotations

import json
import os
import subprocess
import tempfile

from agent.tools.decorator import tool

# Maximum wall-clock seconds a script may run.
_TIMEOUT_SEC = int(os.environ.get("PYTHON_EXEC_TIMEOUT_SEC", "30"))

# Maximum characters returned from stdout / stderr.
_MAX_OUTPUT = 3000

# Patterns that are always blocked regardless of context.
# These prevent the most dangerous operations in a research environment.
_BLOCKED_PATTERNS: list[str] = [
    "os.system(",
    "os.popen(",
    "os.remove(",
    "os.unlink(",
    "os.rmdir(",
    "shutil.rmtree(",
    "shutil.move(",
    "subprocess.run(",
    "subprocess.Popen(",
    "subprocess.call(",
    "__import__('os')",
    '__import__("os")',
    "sys.exit(",
    "open('/etc",
    'open("/etc',
    "open('/proc",
    'open("/proc',
]


def _check_blocked(code: str) -> str | None:
    """Return the first blocked pattern found, or None if code is safe."""
    for pattern in _BLOCKED_PATTERNS:
        if pattern in code:
            return pattern
    return None


@tool(
    "python_exec",
    (
        "Execute a Python code snippet and return its printed output. "
        "Use for: data processing, arithmetic, sorting, filtering, "
        "working with files already loaded into variables, string manipulation. "
        "Available imports: pandas, numpy, math, statistics, json, csv, re, "
        "datetime, collections, itertools, functools, pathlib. "
        "Always use print() to show results — only stdout is captured. "
        "Timeout: 30 seconds. "
        "Do NOT use for: web requests, file deletion, system commands. "
        "Example: python_exec[import pandas as pd\\ndf=pd.read_csv('/path/file.csv')\\nprint(df.describe())]"
    ),
)
def python_exec(code: str) -> str:
    code = (code or "").strip()
    if not code:
        return json.dumps({"ok": False, "error": "No code provided."})

    blocked = _check_blocked(code)
    if blocked:
        return json.dumps(
            {
                "ok": False,
                "error": (
                    f"Blocked operation detected: '{blocked}'. "
                    "File system modifications and shell commands are not permitted."
                ),
            }
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SEC,
            env={
                **os.environ,
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                # Prevent the subprocess from spawning interactive sessions
                "TERM": "dumb",
            },
        )

        if result.returncode == 0:
            output = result.stdout[:_MAX_OUTPUT]
            return json.dumps(
                {
                    "ok": True,
                    "output": output,
                    "stderr": result.stderr[:500] if result.stderr.strip() else None,
                },
                ensure_ascii=False,
            )
        else:
            return json.dumps(
                {
                    "ok": False,
                    "error": result.stderr[:_MAX_OUTPUT] or "Script exited with non-zero status.",
                    "stdout": result.stdout[:500] if result.stdout.strip() else None,
                }
            )

    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "ok": False,
                "error": f"Script timed out after {_TIMEOUT_SEC} seconds. "
                "Simplify the computation or break it into smaller steps.",
            }
        )
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass