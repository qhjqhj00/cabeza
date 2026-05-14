"""
Local persistent code interpreter tool.

This module keeps a dedicated Python worker process alive across tool calls so
variables and imports persist within the same tool instance, similar to a
notebook-style code interpreter. It does not depend on Docker, qwen-agent, or
Jupyter packages.
"""

from __future__ import annotations

import ast
import asyncio
import atexit
import contextlib
import io
import json
import math
import os
import pprint
import re
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional

from cabeza.tools.base import BaseTool


class UnsafeCodeError(Exception):
    """Raised when submitted user code contains blocked operations."""


class _ExecutionTimeout(Exception):
    """Raised inside the worker when a soft execution timeout fires."""


UNSAFE_PATTERNS = [
    r"import\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|socket)",
    r"from\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread|socket)\s+import",
    r"(?<!\w)(input|eval|exec|exit|quit|__import__)\s*\(",
    r"os\.(system|popen|fork|kill|remove|rmdir)",
    r"subprocess\.",
]

_CODE_INTERPRETER_DESCRIPTION = "Python code sandbox, which can be used to execute Python code."
_DEFAULT_TIMEOUT_SECONDS = 60
_ACTIVE_SESSIONS: "weakref.WeakSet[_PersistentInterpreterSession]" = weakref.WeakSet()


def _preview_code_for_log(code: str, max_chars: int = 160) -> str:
    if not isinstance(code, str):
        return f"<non-string code: {type(code).__name__}>"

    compact_lines = [line.strip() for line in code.splitlines() if line.strip()]
    preview = " | ".join(compact_lines[:2]) or "<empty>"
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3] + "..."
    return preview


def _escape_ansi(line: str) -> str:
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line or "")


def _summarize_execution_status(result: str) -> str:
    lowered = (result or "").lower()
    if "timeout: code execution exceeded the time limit." in lowered:
        return "timeout"
    if "\nerror:\n" in lowered or lowered.startswith("error:\n"):
        return "error"
    return "Done"


def _build_python_tool_parameters(file_process: bool = False) -> dict:
    code_description = "The python code."
    if file_process:
        code_description = (
            "The python code. Relative file access is rooted at the tool work "
            "directory; prefer absolute paths when referring to external files."
        )

    return {
        "type": "object",
        "properties": {
            "code": {
                "description": code_description,
                "type": "string",
                "examples": [
                    "x = 5\nprint(x * 2)",
                    "import sympy as sp\nx = sp.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sp.factor(expr))",
                    "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})\nprint(df.head())",
                ],
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    }


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
        sections.append(("error", _escape_ansi(traceback.format_exc())))

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    if stdout_text:
        sections.insert(0, ("stdout", stdout_text))
    if stderr_text:
        insert_idx = 1 if stdout_text else 0
        sections.insert(insert_idx, ("stderr", stderr_text))

    image_blocks = _collect_matplotlib_outputs(work_dir)
    return _render_sections(sections, image_blocks)


def _worker_main(conn, work_dir: str) -> None:
    namespace: dict[str, Any] = {}
    _register_default_modules(namespace, work_dir)

    try:
        while True:
            try:
                payload = conn.recv()
            except EOFError:
                break

            action = payload.get("action")
            if action == "shutdown":
                break

            if action != "execute":
                conn.send({"ok": False, "result": "error:\n\n```\nUnknown worker action.\n```"})
                continue

            result = _execute_user_code(
                payload.get("code", ""),
                namespace,
                work_dir=work_dir,
                timeout_seconds=payload.get("timeout"),
            )
            conn.send({"ok": True, "result": result})
    finally:
        with contextlib.suppress(Exception):
            conn.close()


class _PersistentInterpreterSession:
    def __init__(self, *, work_dir: str):
        self.work_dir = work_dir
        self._lock = threading.Lock()
        self._process = None
        self._stdin = None
        self._stdout = None

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        self.shutdown_locked()
        os.makedirs(self.work_dir, exist_ok=True)
        project_root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            project_root
            if not existing_pythonpath
            else f"{project_root}:{existing_pythonpath}"
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "cabeza.tools._code_worker",
                self.work_dir,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        self._process = process
        self._stdin = process.stdin
        self._stdout = process.stdout

    def execute(self, code: str, timeout_seconds: Optional[int]) -> str:
        with self._lock:
            self._start_locked()
            return self._execute_locked(code, timeout_seconds)

    def _execute_locked(self, code: str, timeout_seconds: Optional[int]) -> str:
        if self._stdin is None or self._stdout is None or self._process is None:
            return "error:\n\n```\nThe code interpreter is not available.\n```"

        payload = {
            "action": "execute",
            "code": code,
            "timeout": timeout_seconds,
        }

        try:
            self._stdin.write(json.dumps(payload) + "\n")
            self._stdin.flush()
        except Exception:
            self._start_locked()
            if self._stdin is None:
                return "error:\n\n```\nFailed to start the code interpreter.\n```"
            self._stdin.write(json.dumps(payload) + "\n")
            self._stdin.flush()

        hard_timeout = float(timeout_seconds or _DEFAULT_TIMEOUT_SECONDS) + 2.0
        response_line: list[str] = []
        reader_error: list[BaseException] = []

        def _reader() -> None:
            try:
                line = self._stdout.readline()
                response_line.append(line)
            except BaseException as exc:
                reader_error.append(exc)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(hard_timeout)

        if thread.is_alive():
            self.shutdown_locked()
            return "error:\n\n```\nTimeout: Code execution exceeded the time limit.\n```"

        if reader_error or not response_line or not response_line[0]:
            self.shutdown_locked()
            return "error:\n\n```\nThe code interpreter terminated unexpectedly.\n```"

        try:
            response = json.loads(response_line[0])
        except Exception:
            self.shutdown_locked()
            return "error:\n\n```\nThe code interpreter returned an invalid response.\n```"

        return str(response.get("result") or "Finished execution.")

    def shutdown(self) -> None:
        with self._lock:
            self.shutdown_locked()

    def shutdown_locked(self) -> None:
        if self._stdin is not None:
            with contextlib.suppress(Exception):
                self._stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                self._stdin.flush()
            with contextlib.suppress(Exception):
                self._stdin.close()
            self._stdin = None

        if self._stdout is not None:
            with contextlib.suppress(Exception):
                self._stdout.close()
            self._stdout = None

        if self._process is not None:
            with contextlib.suppress(Exception):
                if self._process.poll() is None:
                    self._process.terminate()
                    self._process.wait(timeout=1)
                else:
                    self._process.wait(timeout=0.1)
            self._process = None


def _shutdown_active_sessions() -> None:
    for session in list(_ACTIVE_SESSIONS):
        with contextlib.suppress(Exception):
            session.shutdown()


atexit.register(_shutdown_active_sessions)


def execute_python_code_sync(
    code: str,
    *,
    timeout_length: int = _DEFAULT_TIMEOUT_SECONDS,
    max_concurrency: int = 8,
) -> str:
    del max_concurrency
    work_dir = os.path.join(tempfile.gettempdir(), "gizmo_code_interpreter", str(uuid.uuid4()))
    session = _PersistentInterpreterSession(work_dir=work_dir)
    try:
        return session.execute(code, timeout_length)
    finally:
        session.shutdown()


async def execute_python_code(
    code: str,
    *,
    timeout_length: int = _DEFAULT_TIMEOUT_SECONDS,
    max_concurrency: int = 8,
) -> str:
    return await asyncio.to_thread(
        execute_python_code_sync,
        code,
        timeout_length=timeout_length,
        max_concurrency=max_concurrency,
    )


class PythonTool(BaseTool):
    """Notebook-like local code interpreter with per-instance persistent state."""

    def __init__(
        self,
        *,
        file_process: bool = False,
        timeout_length: int = _DEFAULT_TIMEOUT_SECONDS,
        max_concurrency: int = 8,
        name: str = "code_interpreter",
        work_dir: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            description=_CODE_INTERPRETER_DESCRIPTION,
            parameters=_build_python_tool_parameters(file_process=file_process),
        )
        self.timeout_length = timeout_length
        self.max_concurrency = max_concurrency
        self.file_process = file_process
        self.work_dir = os.path.abspath(
            work_dir
            or os.path.join(
                tempfile.gettempdir(),
                "gizmo_code_interpreter",
                str(uuid.uuid4()),
            )
        )
        self._session = _PersistentInterpreterSession(work_dir=self.work_dir)
        _ACTIVE_SESSIONS.add(self._session)

    def execute(self, code: str) -> str:
        line_count = len(code.splitlines()) if isinstance(code, str) else 0
        preview = _preview_code_for_log(code)
        print(f"[PythonTool] Executing code ({line_count} lines): {preview}")
        result = self._session.execute(code, self.timeout_length)
        status = _summarize_execution_status(result)
        print(f"[PythonTool] Finished with status: {status}")
        return result if result.strip() else "Finished execution."

    def shutdown(self) -> None:
        self._session.shutdown()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.shutdown()


CodeInterpreterTool = PythonTool
