"""Release hygiene: the modules import, entry points start, paths named in the tree exist, and
the tree carries no CJK text, no cluster path and no AI-READI participant id.

Data-free; seconds, most of them spent importing torch in the child processes. The full
static checker of every import, attribute, call signature and path (shell heredocs,
``python -c`` strings and pathlib joins included) runs outside this suite, at every step of
the restructuring that produced this tree; this file keeps the lasting part.

Each baseline below freezes what the release base (a5da238) already had. A found set must
equal its baseline: a new finding fails, and a finding that goes away must leave the baseline
in the same change, so the baselines only shrink.

Every message masks standalone four-digit runs (AI-READI participant ids are four-digit
numbers); a baseline entry that holds one is stored as ``sha256:<first 16 hex>``.
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from conftest import skip_lint




REPO = Path(__file__).resolve().parents[1]
SELF = "tests/test_release_hygiene.py"

#: The ``python -m`` entry points: what the shells, the container and the docs launch.
ENTRY_POINTS = ("octtta.infer", "octtta.runenv_status", "octtta.train")

#: The names the teardown repository imports, at the module paths it imports them from.
TEARDOWN_NAMES = {
    "octtta.data.dataset": ("normalize_image",),
    "octtta.engine": ("load_inference_model",),
    "octtta.fusion": ("alpha_for_image", "fuse_probs", "shape_class"),
    "octtta.infer": ("_gaussian_patch_weight", "labels_from_probs", "load_fusion_partner",
                     "patch_starts", "plan_from_config", "probs_for_image"),
    "octtta.surface": ("expected_boundaries", "boundaries_from_labels"),
}

#: ``<top>/...`` in code or docs names a repo path when <top> is one of these. A fixed set, so
#: a path into a deleted top directory is still recognised.
PATH_TOPS = ("configs", "data", "docker", "docs", "external", "octtta", "scripts", "tests",
             "tools", "submission", ".githooks")
#: Recorded data and the organisers' vendored kit are not read for paths.
NOT_PATH_CHECKED = ("tests/data/", "external/")
CODE_AND_DOCS = (".py", ".sh", ".sbatch", ".bash", ".def", ".yaml", ".yml", ".json", ".cfg",
                 ".toml", ".md", ".txt", ".rst")

#: Paths named in code or docs that do not exist at the release base.
MISSING_PATHS: frozenset[str] = frozenset()

#: Files holding an AI-READI-id-shaped token in an id context at the release base, with the
#: count (never the token): the lists and old tests the restructuring deletes, and collisions.
ID_CONTEXT_BASELINE: dict[str, int] = {}

_RUN = re.compile(r"(?<!\d)\d{4}(?!\d)")
_BINARY = (".gz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npz", ".npy", ".pt",
           ".pth", ".zip", ".tar", ".pkl", ".safetensors", ".dcm", ".bin", ".h5", ".so", ".pyc",
           ".whl", ".xz", ".bz2")
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def mask(text) -> str:
    return _RUN.sub("####", str(text))


def baseline_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16] if _RUN.search(text) else text


def _fmt(items, limit: int = 60) -> str:
    items = sorted(items)
    more = f"\n  ... and {len(items) - limit} more" if len(items) > limit else ""
    return "\n  " + "\n  ".join(items[:limit]) + more


def tree_files() -> list[str]:
    """Repo-relative files: git's tracked plus untracked-not-ignored, else a directory walk."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True, check=True,
                             timeout=120).stdout.decode("utf-8", "surrogateescape")
        rels = sorted(r for r in set(out.split("\0")) if r and (REPO / r).is_file())
        if rels:
            return rels
    except (OSError, subprocess.SubprocessError):
        pass
    rels = []
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        rels += [os.path.relpath(os.path.join(dirpath, f), REPO).replace(os.sep, "/")
                 for f in filenames if not f.endswith((".pyc", ".pyo"))]
    return sorted(rels)


FILES = tree_files()


def text_files():
    for rel in FILES:
        if not rel.endswith(_BINARY):
            try:
                yield rel, (REPO / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue


def _is_shell(rel: str, text: str) -> bool:
    first = text.split("\n", 1)[0]
    return rel.endswith((".sh", ".sbatch", ".bash")) or (
        "." not in rel.rsplit("/", 1)[-1] and first.startswith("#!")
        and bool(re.search(r"\b(ba|da|z)?sh\b", first)))


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1"}


# ------------------------------------------------------------------ the tree runs

_IMPORT_ALL = r"""
import importlib, json, pathlib, sys
root, bad = pathlib.Path(sys.argv[1]).resolve(), {}
for name in sys.argv[2:]:
    try:
        mod = importlib.import_module(name)
    except BaseException as exc:
        bad[name] = f"{type(exc).__name__}: {exc}"
        continue
    where = [getattr(mod, "__file__", None), *getattr(mod, "__path__", [])]
    if not all(pathlib.Path(f).resolve().is_relative_to(root) for f in where if f):
        bad[name] = "imported from outside the tree"
print(json.dumps(bad))
"""


def test_every_octtta_module_imports() -> None:
    modules = sorted(r[:-3].replace("/", ".").removesuffix(".__init__") for r in FILES
                     if r.startswith("octtta/") and r.endswith(".py"))
    assert len(modules) >= 20, modules
    p = subprocess.run([sys.executable, "-c", _IMPORT_ALL, str(REPO), *modules], cwd=REPO,
                       env=_env(), capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, mask(p.stderr[-2000:])
    bad = json.loads(p.stdout.strip().splitlines()[-1])
    assert not bad, "modules that do not import:" + _fmt(f"{k}: {mask(v)}" for k, v in bad.items())


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_every_entry_point_answers_help(module: str) -> None:
    p = subprocess.run([sys.executable, "-m", module, "--help"], cwd=REPO, env=_env(),
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "usage" in p.stdout.lower(), mask((p.stdout + p.stderr)[-2000:])


def test_the_tree_launches_only_these_entry_points() -> None:
    """Every ``-m octtta.<module>`` in a shell script, a container file or a doc is listed."""
    launched = defaultdict(set)
    for rel, text in text_files():
        if (_is_shell(rel, text) or "dockerfile" in rel.lower() or rel.endswith(".def")
                or rel.endswith((".md", ".txt", ".rst"))):
            for m in re.finditer(r"(?<![\w-])-m[ \t]+(octtta(?:\.\w+)+)", text):
                launched[m.group(1)].add(rel)
    assert launched, "no entry point launched anywhere"
    extra = {k: v for k, v in launched.items() if k not in ENTRY_POINTS}
    assert not extra, "launched but not in ENTRY_POINTS:" + _fmt(
        f"{k} <- {sorted(v)}" for k, v in extra.items())


def test_shell_files_parse() -> None:
    shells = [rel for rel, text in text_files()
              if not rel.startswith(NOT_PATH_CHECKED) and _is_shell(rel, text)]
    assert shells, "no shell scripts found"
    bad = []
    for rel in shells:
        p = subprocess.run(["bash", "-n", str(REPO / rel)], capture_output=True, text=True,
                           timeout=60)
        if p.returncode:
            bad.append(mask(f"{rel}: {p.stderr.strip()[-300:]}"))
    assert not bad, _fmt(bad)


@pytest.mark.parametrize("module", sorted(TEARDOWN_NAMES))
def test_teardown_names_are_importable_at_their_paths(module: str) -> None:
    mod = importlib.import_module(module)
    missing = [n for n in TEARDOWN_NAMES[module] if not callable(getattr(mod, n, None))]
    assert not missing, f"{module} lacks {missing}: the teardown repository imports them"


# ------------------------------------------------------------------ the paths it names exist

def _gitignored() -> list[str]:
    path = REPO / ".gitignore"
    lines = path.read_text().splitlines() if path.is_file() else []
    return [ln.strip().rstrip("/") for ln in lines if ln.strip() and ln[0] not in "#!"]


def missing_paths() -> tuple[dict[str, set[str]], int]:
    """``({missing path: files naming it}, number of path tokens checked)``."""
    dirs = {r.rsplit("/", 1)[0] if "/" in r else "" for r in FILES}
    dirs |= {"/".join(d.split("/")[:i]) for d in list(dirs) for i in range(1, d.count("/") + 1)}
    known = set(FILES) | dirs
    ignored = _gitignored()
    tops = "|".join(re.escape(t) for t in sorted(PATH_TOPS, key=len, reverse=True))
    bare = re.compile(r"(?<![\w.$/~@+-])(?:%s)/[\w.+@/-]*" % tops)
    anchored = re.compile(r"\$\{?(?:REPO|REPO_ROOT|OCTTTA_REPO)\}?/((?:%s)(?![\w.+@-])[\w.+@/-]*)"
                          % tops)
    found: dict[str, set[str]] = defaultdict(set)
    n = 0
    for rel, text in text_files():
        if (rel == SELF or rel.startswith(NOT_PATH_CHECKED)
                or not (rel.endswith(CODE_AND_DOCS) or "dockerfile" in rel.lower()
                        or _is_shell(rel, text))):
            continue
        tokens = [(m.group(0), m.end()) for m in bare.finditer(text)]
        tokens += [(m.group(1), m.end()) for m in anchored.finditer(text)]
        for tok, end in tokens:
            partial = end < len(text) and text[end] in "${<*?%[("
            path = tok.rstrip(".-")
            last = path.rstrip("/").rsplit("/", 1)[-1]
            if not path or "//" in path or (not partial and last.isdigit()):
                continue                              # "octtta/expected/1" is a schema id
            n += 1
            if partial and not path.endswith("/"):
                head, _, stem = path.rpartition("/")
                ok = head in known and any(k.startswith(f"{head}/{stem}") for k in known)
            else:
                candidate = path.rstrip("/")
                ok = candidate in known
            parts = path.rstrip("/").split("/")
            if not ok and not any(fnmatch.fnmatch(c, g)
                                  for c in [path.rstrip("/"), *parts] for g in ignored):
                found[path.rstrip("/") + ("*" if partial and not path.endswith("/") else "")
                      ].add(rel)
    return found, n


def test_every_repo_path_named_in_code_and_docs_exists() -> None:
    found, n = missing_paths()
    assert n >= 100, f"only {n} path tokens checked"
    keys = {baseline_key(p): p for p in found}
    new = [mask(f"{keys[k]} <- {sorted(found[keys[k]])}") for k in set(keys) - MISSING_PATHS]
    gone = [mask(k) for k in MISSING_PATHS - set(keys)]
    assert not new, "paths named in code or docs that do not exist:" + _fmt(new)
    assert not gone, "no longer missing, delete from MISSING_PATHS:" + _fmt(gone)


# ------------------------------------------------------------------ what the tree carries

_CJK = re.compile("[\u2e80-\u2fdf\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
                  "\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
_CLUSTER = re.compile("/" + "gpfs/|/" + "vast/")


def test_no_cjk_text() -> None:
    bad = [f"{mask(rel)}: {len(_CJK.findall(t))}" for rel, t in text_files() if _CJK.search(t)]
    assert not bad, "CJK characters (the release is English):" + _fmt(bad)


def test_no_absolute_cluster_path() -> None:
    bad = [mask(rel) for rel, t in text_files() if _CLUSTER.search(t)]
    assert not bad, "absolute cluster paths (spell them from ${OCTTTA_SCRATCH}):" + _fmt(bad)


def test_no_aireadi_metadata_file() -> None:
    names = {rel.rsplit("/", 1)[-1].lower() for rel in FILES}
    assert not names & {"participants.tsv", "manifest.tsv"}


#: AI-READI participant ids are four-digit numbers whose first digit is 1, 4 or 7 (test
#: fixtures use 9xxx). A token counts when no hex digit touches it, it is not part of a decimal
#: or of a hex literal, and it sits in an id context: a line naming a person-like word, or a
#: file-stem position (between separators, before a device or eye, after "Subject_").
_ID_SHAPED = re.compile(r"(?<![0-9A-Fa-f.])[147]\d{3}(?![0-9A-Fa-f]|\.\d)")
_ID_WORDS = re.compile(r"person|participant|pid|subject|patient|donor|people|eye", re.I)
_ID_NEXT = re.compile(r"_(?:Heidelberg|Topcon|Zeiss|Spectralis|Maestro2|Triton|Cirrus|L|R)"
                      r"(?![a-z])")


def id_context_hits(line: str) -> int:
    if re.fullmatch(r'\s*"persons": \d+,?\s*', line):
        return 0
    n = 0
    for m in _ID_SHAPED.finditer(line):
        s, e = m.span()
        before, after = line[s - 1:s], line[e:e + 1]
        if re.search(r"0[xX][0-9A-Fa-f_]*$", line[:s]):
            continue
        if (_ID_WORDS.search(line) or (before and after and before in "/_" and after in "/_")
                or _ID_NEXT.match(line, e) or line[:s].endswith("Subject_")):
            n += 1
    return n


def id_context_counts() -> dict[str, int]:
    counts = Counter()
    for rel, text in text_files():
        for line in text.splitlines():
            counts[rel] += id_context_hits(line)
    return {rel: n for rel, n in counts.items() if n}


def test_skip_lint_rejects_each_bare_skip_form(tmp_path: Path) -> None:
    probe = tmp_path / "test_skip_lint_probe.py"
    dotted = ["pytest" + ".skip", "pytest" + ".importorskip",
              "pytest.mark." + "skip", "pytest.mark." + "skipif"]
    probe.write_text("import pytest\n" + "\n".join(f"{call}(True)" for call in dotted))
    findings = skip_lint(probe)
    assert {item.split(": ", 1)[1] for item in findings} == {
        "pytest.skip", "pytest.importorskip", "pytest.mark.skip", "pytest.mark.skipif"}


def test_no_aireadi_id_shaped_token_in_an_id_context() -> None:
    assert id_context_hits("participant " + "12" + "34") == 1
    assert id_context_hits('"person_id": '+"12"+"34") == 1
    assert id_context_hits('"persons": '+"12"+"34") == 0
    assert id_context_hits("x_" + "92" + "34_L") == 0             # a 9xxx placeholder
    assert id_context_hits("0x2B_" + "71" + "74_") == 0           # a hex literal
    got = id_context_counts()
    bad = [f"{mask(rel)}: {n} (baseline {ID_CONTEXT_BASELINE.get(rel, 0)})"
           for rel, n in got.items() if n != ID_CONTEXT_BASELINE.get(rel, 0)]
    bad += [f"{mask(rel)}: 0 (baseline {n}): delete it from ID_CONTEXT_BASELINE"
            for rel, n in ID_CONTEXT_BASELINE.items() if rel not in got]
    assert not bad, ("id-shaped tokens in an id context (use 9xxx placeholders; never a real "
                     "participant id):" + _fmt(bad))


def test_needs_count_ignores_bytecode_but_requires_source_files(tmp_path, monkeypatch):
    import conftest
    source = tmp_path / "kit"
    source.mkdir()
    (source / "scorer.py").write_text("pass\n")
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "scorer.cpython-311.pyc").write_bytes(b"cache")
    monkeypatch.setattr(conftest, "_expected", lambda: {
        "needs": {"fake_kit": {"root": "data", "count": {"kit": 1}}}})
    monkeypatch.setattr(conftest, "needs_root", lambda name: tmp_path)
    conftest._problem.cache_clear()
    try:
        assert conftest._problem("fake_kit") is None
        (source / "scorer.py").unlink()
        conftest._problem.cache_clear()
        assert "files, expected" in conftest._problem("fake_kit")
    finally:
        conftest._problem.cache_clear()
