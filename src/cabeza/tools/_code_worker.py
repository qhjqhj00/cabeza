from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import pprint
import re
import signal
import sys
import threading
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional


class UnsafeCodeError(Exception):
    pass


class _ExecutionTimeout(Exception):
    pass


UNSAFE_PATTERNS = [
    r"import\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|socket)",
    r"from\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|socket)\s+import",
    r"(?<!\w)(input|eval|exec|exit|quit|__import__)\s*\(",
    r"os\.(system|popen|fork|kill|remove|rmdir)",
    r"subprocess\.",
]


def _register_default_modules(namespace: dict[str, Any], work_dir: str) -> None:
    namespace.update(
        {
            "__name__": "__main__",
            "__builtins__": __builtins__,
            "math": math,
            "Path": Path,
            "WORK_DIR": work_dir,
        }
    )

    optional_imports = (
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sympy", "sp"),
    )
    for module_name, alias in optional_imports:
        try:
            module = __import__(module_name)
        except Exception:
            continue
        namespace[module_name] = module
        namespace[alias] = module

    if "sp" in namespace:
        try:
            x, y, z = namespace["sp"].symbols("x y z")
            namespace.update({"x": x, "y": y, "z": z})
        except Exception:
            pass

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        namespace["matplotlib"] = matplotlib
        namespace["plt"] = plt
    except Exception:
        pass


def _validate_user_code(code_piece: str) -> None:
    if not isinstance(code_piece, str):
        raise TypeError("Code must be provided as a string.")

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, code_piece):
            raise UnsafeCodeError(
                "Your process is not safe. Execution of potentially unsafe code was blocked."
            )


def _compile_user_code(code: str) -> tuple[Optional[Any], Optional[Any]]:
    module = ast.parse(code, mode="exec")
    expr_code = None

    if module.body and isinstance(module.body[-1], ast.Expr):
        expr_node = ast.Expression(module.body[-1].value)
        expr_code = compile(ast.fix_missing_locations(expr_node), "<code_interpreter>", "eval")
        module.body = module.body[:-1]

    exec_code = None
    if module.body:
        exec_code = compile(ast.fix_missing_locations(module), "<code_interpreter>", "exec")

    return exec_code, expr_code


def _stringify_result(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return pprint.pformat(value, sort_dicts=False)
    except Exception:
        return repr(value)


def _collect_matplotlib_outputs(work_dir: str) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    image_blocks: list[str] = []
    fig_nums = list(plt.get_fignums())
    for index, fig_num in enumerate(fig_nums, start=1):
        try:
            fig = plt.figure(fig_num)
            image_path = os.path.join(work_dir, f"{uuid.uuid4()}.png")
            fig.savefig(image_path, format="png", bbox_inches="tight")
            image_blocks.append(f"![fig-{index:03d}]({image_path})")
        finally:
            with contextlib.suppress(Exception):
                plt.close(fig_num)
    return image_blocks


def _render_sections(sections: list[tuple[str, str]], image_blocks: list[str]) -> str:
    parts: list[str] = []
    for label, text in sections:
        cleaned = text.rstrip()
        if not cleaned:
            continue
        parts.append(f"{label}:\n\n```\n{cleaned}\n```")
    parts.extend(image_blocks)
    if not parts:
        return "Finished execution."
    return "\n\n".join(parts)


@contextlib.contextmanager
def _working_directory(path: str):
    old_cwd = os.getcwd()
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextlib.contextmanager
def _soft_time_limit(timeout_seconds: Optional[int]):
    if (
        not timeout_seconds
        or timeout_seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "setitimer")
    ):
        yield
        return

    def _raise_timeout(_sig_num, _frame):
        raise _ExecutionTimeout

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _execute_user_code(
    code: str,
    namespace: dict[str, Any],
    *,
    work_dir: str,
    timeout_seconds: Optional[int],
) -> str:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    sections: list[tuple[str, str]] = []

    try:
        _validate_user_code(code)
        code = (code or "").strip()
        if not code:
            return ""
        exec_code, expr_code = _compile_user_code(code)
        with _working_directory(work_dir):
            with _soft_time_limit(timeout_seconds):
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    if exec_code is not None:
                        exec(exec_code, namespace, namespace)
                    if expr_code is not None:
                        value = eval(expr_code, namespace, namespace)
                        rendered = _stringify_result(value)
                        if rendered:
                            sections.append(("execute_result", rendered))
    except _ExecutionTimeout:
        sections.append(("error", "Timeout: Code execution exceeded the time limit."))
    except Exception:
        sections.append(("error", traceback.format_exc()))

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    if stdout_text:
        sections.insert(0, ("stdout", stdout_text))
    if stderr_text:
        insert_idx = 1 if stdout_text else 0
        sections.insert(insert_idx, ("stderr", stderr_text))

    image_blocks = _collect_matplotlib_outputs(work_dir)
    return _render_sections(sections, image_blocks)


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "result": "error:\n\n```\nMissing work directory.\n```"}))
        return 1

    work_dir = sys.argv[1]
    namespace: dict[str, Any] = {}
    _register_default_modules(namespace, work_dir)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except Exception:
            response = {"ok": False, "result": "error:\n\n```\nInvalid worker payload.\n```"}
            print(json.dumps(response), flush=True)
            continue

        action = payload.get("action")
        if action == "shutdown":
            print(json.dumps({"ok": True, "result": "Finished execution."}), flush=True)
            break

        if action != "execute":
            print(json.dumps({"ok": False, "result": "error:\n\n```\nUnknown worker action.\n```"}), flush=True)
            continue

        result = _execute_user_code(
            payload.get("code", ""),
            namespace,
            work_dir=work_dir,
            timeout_seconds=payload.get("timeout"),
        )
        print(json.dumps({"ok": True, "result": result}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
