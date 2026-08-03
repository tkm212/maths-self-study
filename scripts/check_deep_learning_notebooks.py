#!/usr/bin/env python3
"""Validate Deep Learning marimo notebooks are not corrupted."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    p
    for p in (ROOT / "notebooks/deep-learning").rglob("*.py")
    if not p.name.startswith("ch")
]


def _cell_functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _is_empty_cell(fn: ast.FunctionDef) -> bool:
    body = [s for s in fn.body if not isinstance(s, (ast.Pass, ast.Expr))]
    # allow mo.md(...) then return
    meaningful = []
    for s in fn.body:
        if isinstance(s, ast.Pass):
            continue
        if isinstance(s, ast.Return) and s.value is None:
            continue
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            meaningful.append(s)
            continue
        meaningful.append(s)
    return len(meaningful) == 0


def _returns_name(fn: ast.FunctionDef, name: str) -> bool:
    for stmt in fn.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Tuple):
            return any(isinstance(e, ast.Name) and e.id == name for e in stmt.value.elts)
    return False


def main() -> int:
    failed = False
    for path in sorted(NOTEBOOKS):
        tree = ast.parse(path.read_text())
        funcs = _cell_functions(tree)
        rel = path.relative_to(ROOT)

        if len(funcs) < 3:
            print(f"FAIL {rel}: too few cells")
            failed = True
            continue

        if not _returns_name(funcs[0], "mo"):
            print(f"FAIL {rel}: cell 1 must return (mo,)")
            failed = True
        if not _returns_name(funcs[1], "helpers"):
            print(f"FAIL {rel}: cell 2 must return (helpers,)")
            failed = True

        empties = sum(1 for fn in funcs if _is_empty_cell(fn))
        if empties:
            print(f"FAIL {rel}: {empties} empty cell(s)")
            failed = True

        src = path.read_text()
        if "hide_code" in src:
            print(f"FAIL {rel}: hide_code still present")
            failed = True
        if "from pathlib import Path" in src or "import sys" in src:
            print(f"FAIL {rel}: sys/pathlib bootstrap — use maths_self_study.deep_learning")
            failed = True

    if failed:
        return 1
    print(f"OK: {len(NOTEBOOKS)} deep-learning notebooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
