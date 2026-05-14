"""Run every cabeza test suite in sequence and print one combined summary."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


SUITES = [
    ("strategies_unit", "test_strategies_unit"),
    ("live",            "test_live"),
    ("strategies_live", "test_strategies_live"),
    ("teams",           "test_teams"),
    ("runner",          "test_runner"),
]


def main() -> int:
    started = time.time()
    failures: list[tuple[str, int]] = []
    for label, modname in SUITES:
        print(f"\n{'#' * 64}\n# suite: {label}\n{'#' * 64}")
        mod = importlib.import_module(modname)
        code = mod.main([])
        if code != 0:
            failures.append((label, code))

    print(f"\n{'=' * 64}")
    print(f"OVERALL ({time.time() - started:.1f}s)")
    print("=" * 64)
    if failures:
        for label, code in failures:
            print(f"  [FAIL] {label}  exit={code}")
        return 1
    for label, _ in SUITES:
        print(f"  [PASS] {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
