"""Shared pytest wiring.

* ``needs(name, ...)`` marks a test that reads a data set of ``configs/expected.json:needs``: it
  is skipped, with the reason, unless every file listed there is present with its pinned size
  (and sha256 where given) and every counted directory holds exactly that many files,
  excluding Python bytecode caches. With
  ``OCTTTA_TEST_REQUIRE_DATA=1`` it runs anyway and fails on what is missing, so a machine that
  should have the data cannot pass by skipping. ``needs_root(name)`` is where that data set lives.
* A collection-time lint over every ``test_*.py`` under ``tests/``: tests skip only through
  ``needs()``; a bare ``pytest.skip`` / ``skipif`` / ``importorskip`` is refused.
* The ``slow`` marker; the per-step runs deselect it with ``-m "not slow"``.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPECTED = REPO / "configs" / "expected.json"

# ------------------------------------------------------------------------------ needs()

@functools.lru_cache(maxsize=1)
def _expected() -> dict:
    return json.loads(EXPECTED.read_text())


def _root(name: str) -> Path:
    spec = _expected()["roots"][name]
    value = os.environ.get(spec["env"]) or re.sub(
        r"\$\{(\w+)\}", lambda m: str(_root(m.group(1))), spec["default"])
    return Path(os.path.expanduser(value))


def needs_root(name: str) -> Path:
    """The directory the data set ``name`` of configs/expected.json:needs lives in."""
    return _root(_expected()["needs"][name]["root"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def _problem(name: str) -> str | None:
    """Why the data set ``name`` is not usable here (None: it is)."""
    entry = _expected()["needs"][name]
    base = needs_root(name)
    for rel, pin in (entry.get("files") or {}).items():
        path = base / rel
        if not path.is_file():
            return f"{rel} missing under root '{entry['root']}'"
        if path.stat().st_size != pin["bytes"]:
            return f"{rel}: size differs from the pin"
        if "sha256" in pin and _sha256(path) != pin["sha256"]:
            return f"{rel}: content differs from the pin"
    for rel, n in (entry.get("count") or {}).items():
        path = base / rel
        got = sum(1 for p in path.rglob("*") if p.is_file() and not (
            p.suffix == ".pyc" and "__pycache__" in p.relative_to(path).parts)) if path.is_dir() else 0
        if got != n:
            return f"{rel}: {got} files, expected {n}"
    return None


def needs(*names: str):
    """Mark a test that reads the named data sets of configs/expected.json:needs."""
    unknown = [n for n in names if n not in _expected()["needs"]]
    if not names or unknown:
        raise ValueError(f"needs(): unknown data set(s) {unknown or '(none given)'}; "
                         f"configs/expected.json lists {sorted(_expected()['needs'])}")
    problems = [f"needs({n}): {p}" for n in names if (p := _problem(n))]
    require = os.environ.get("OCTTTA_TEST_REQUIRE_DATA") == "1"
    return pytest.mark.skipif(bool(problems) and not require,
                              reason="; ".join(problems) or "data present")


# ------------------------------------------------------------------------ the skip lint

_SKIPS = {"pytest.skip", "pytest.importorskip", "pytest.mark.skip", "pytest.mark.skipif"}


def _dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


def skip_lint(path: Path) -> list[str]:
    """Every bare skip in ``path``: in a lean file a test skips only through needs()."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [f"{path.name}:{node.lineno}: {_dotted(node)}" for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and _dotted(node) in _SKIPS]


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: long-running; the per-step runs deselect it "
                                       "with -m 'not slow'")


def pytest_collection_modifyitems(session, config, items) -> None:
    test_files = sorted((REPO / "tests").rglob("test_*.py"))
    problems = [p for path in test_files for p in skip_lint(path)]
    if problems:
        raise pytest.UsageError("skip lint: every tests/test_*.py file skips only through needs() "
                                "(tests/conftest.py):\n  " + "\n  ".join(problems))
