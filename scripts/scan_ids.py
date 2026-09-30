#!/usr/bin/env python3
"""Scan a release git tree for licensed participant identifiers.

Only aggregate counts leave this process. Acknowledgements contain file paths, hashes of
reviewed text, and short reasons, never identifiers; they must live outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ID_TOKEN = re.compile(r"(?<![0-9A-Fa-f])([0-9]{4})(?![0-9A-Fa-f])")
ID_CONTEXT = re.compile(
    r"person|participant|\bpid\b|subject|patient|donor|people|\beye\b|filepath|filename|\bstem\b|\bpath\b",
    re.IGNORECASE,
)
STEM_CONTEXT = re.compile(r"__t__|/t/|/t_|_t_|(?:^|[/_])t_[A-Za-z0-9]+|Subject_t", re.I)
DATA_EXTENSIONS = (".csv", ".tsv", ".csv.gz", ".jsonl")
TEXT_EXTENSIONS = {
    ".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".tsv", ".csv", ".html", ".css", ".js", ".svg", ".rst",
    ".cfg", ".ini", ".lock", ".bat", ".sbatch", ".gitignore", ".gitattributes",
    ".githooks", ".dockerignore", ".license",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
WEIGHT_EXTENSIONS = {".pt", ".pth", ".safetensors"}
IDENTITY_FIELD = re.compile(
    r"^(?:person|participant|subject|patient|donor)(?:_ids?|_number)?$|^(?:pid|people_ids?|ids?)$",
    re.IGNORECASE,
)
PATH_FIELD = re.compile(r"(?:^|_)(?:path|filepath|filename|stem)$", re.IGNORECASE)
DIGEST_FIELD = re.compile(r"(?:^|_)(?:sha256|fingerprint|fp)$", re.IGNORECASE)
DIGEST_VALUE = re.compile(r"(?:[0-9a-f]{16}|[0-9a-f]{64})", re.IGNORECASE)
TECHNICAL_CONTEXT = re.compile(
    r"\b(?:seconds?|budget|reserve|floor|finetune|meminfo|bytes?|height|width|"
    r"pixels?|columns?|frames?|raster|nm|wavelength|histogram|hist_max|"
    r"rng|seed|channels?|num_chs|shapes?|steps?|weights?|score|loss|alpha|"
    r"extreme_frac|samples_per_epoch|ru_maxrss|thickness|resolution)\b|"
    r"\b(?:Triton|Cirrus|Spectralis|Maestro2)\b|\\times|_HIST_MAX|"
    r"\bBUDGET_[A-Z_]+\b",
    re.IGNORECASE,
)


def participant_ids(path: Path) -> set[str]:
    """Read the licensed four-digit set without copying it into logs or artifacts."""
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            value = (row.get("person_id") or "").strip()
            if re.fullmatch(r"[0-9]{4}", value):
                ids.add(value)
    if not ids:
        raise ValueError("licensed participant metadata contains no four-digit IDs")
    return ids


def _git(*args: str, cwd: Path, stdout=None) -> bytes:
    run = subprocess.run(["git", *args], cwd=cwd, stdout=stdout or subprocess.PIPE,
                         stderr=subprocess.DEVNULL, check=True)
    return run.stdout if stdout is None else b""


def tree_blobs(repo: Path, tree: str):
    for line in _git("ls-tree", "-r", "-z", tree, cwd=repo).split(b"\0"):
        if not line:
            continue
        head, name = line.split(b"\t", 1)
        mode, kind, oid = head.split()
        if kind == b"blob":
            yield name.decode("utf-8", "surrogateescape"), oid.decode("ascii")


def worktree_files(repo: Path):
    """Visit tracked and untracked files, including edits not yet committed."""
    for raw in _git("ls-files", "-c", "-o", "--exclude-standard", "-z", cwd=repo).split(b"\0"):
        if raw:
            path = raw.decode("utf-8", "surrogateescape")
            if (repo / path).exists() or (repo / path).is_symlink():
                yield path


def _blob(repo: Path, oid: str) -> bytes:
    return _git("cat-file", "blob", oid, cwd=repo)


def _line_hash(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8", "surrogateescape")).hexdigest()


def _load_ack(path: Path | None, repo: Path) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    if path.resolve().is_relative_to(repo.resolve()):
        raise ValueError("acknowledgement file must be outside the repository")
    ack = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if (len(parts) != 3 or not re.fullmatch(r"[0-9a-f]{64}", parts[1])
                or not re.fullmatch(r"[A-Za-z][A-Za-z _/-]{4,100}", parts[2])):
            raise ValueError("acknowledgement file has an invalid record")
        ack.add((parts[0], parts[1]))
    return ack


def _data_path(name: str) -> bool:
    low = name.lower()
    fixture_roots = tuple("/".join(("tests", part, "")) for part in ("data", "golden"))
    return low.endswith(DATA_EXTENSIONS) or low.startswith(fixture_roots) or low in (
        "configs/aireadi_exclusions.json", "configs/expected.json")


def _hits(value: str, ids: set[str]) -> bool:
    return any(match.group(1) in ids for match in ID_TOKEN.finditer(value))


def _png_metadata(data: bytes) -> bytes:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid PNG")
    pos, fields = 8, []
    while pos + 12 <= len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + size
        if end > len(data):
            raise ValueError("truncated PNG")
        payload = data[pos + 8:pos + 8 + size]
        if kind in (b"tEXt", b"eXIf"):
            fields.append(payload)
        elif kind == b"iTXt":
            try:
                key, rest = payload.split(b"\0", 1)
                compressed, method = rest[:2]
                rest = rest[2:]
                language, rest = rest.split(b"\0", 1)
                translated, value = rest.split(b"\0", 1)
                fields.append(b"\n".join((key, language, translated,
                                           zlib.decompress(value) if compressed else value)))
            except (ValueError, IndexError, zlib.error):
                raise ValueError("invalid PNG text metadata") from None
        elif kind == b"zTXt":
            sep = payload.find(b"\0")
            if sep >= 0 and len(payload) > sep + 2:
                fields.append(payload[:sep] + b"\n" + zlib.decompress(payload[sep + 2:]))
        pos = end
        if kind == b"IEND":
            break
    return b"\n".join(fields)


def _jpeg_metadata(data: bytes) -> bytes:
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:
        values = list(image.info.values()) + list(image.getexif().values())
    return "\n".join(str(v) for v in values).encode("utf-8", "surrogateescape")


def _weight_text(path: Path) -> str:
    if path.suffix == ".safetensors":
        with path.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > 100_000_000:
                raise ValueError("invalid safetensors header")
            return fh.read(n).decode("utf-8")
    import torch
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    seen: set[int] = set()
    tokens: list[str] = []

    def walk(obj, depth: int = 0):
        if depth > 32 or len(tokens) > 100_000:
            raise ValueError("weight metadata exceeds scanner bounds")
        if isinstance(obj, torch.Tensor):
            return
        if isinstance(obj, (dict, list, tuple, set)):
            ident = id(obj)
            if ident in seen:
                return
            seen.add(ident)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(key, str) and isinstance(value, (str, int, float, bool, bytes)):
                        scalar = (value.decode("utf-8", "replace") if isinstance(value, bytes)
                                  else str(value))
                        tokens.append(f"{key}\t{scalar}")
                    else:
                        walk(key, depth + 1)
                        walk(value, depth + 1)
            else:
                for value in obj:
                    walk(value, depth + 1)
            return
        if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
            tokens.append(obj.decode("utf-8", "replace") if isinstance(obj, bytes) else str(obj))
        else:
            tokens.append(type(obj).__name__)

    walk(payload)
    return "\n".join(tokens)


def _classify_text(name: str, text: str, ids: set[str], ack: set[tuple[str, str]],
                   counts: dict[str, int]) -> None:
    def technical_number(line: str, match: re.Match[str]) -> bool:
        if ID_CONTEXT.search(line) or STEM_CONTEXT.search(line):
            return False
        start, end = match.span(1)
        before, after = line[:start], line[end:]
        if re.search(r"\d\.$", before):
            return True  # decimal precision, not a person identifier
        if (re.fullmatch(r"20\d{2}", match.group(1))
                and re.match(r"-\d{2}-\d{2}(?!\d)", after)):
            return True  # ISO calendar date
        if re.search(r"0x[0-9a-f_]+$", before, re.IGNORECASE):
            return True  # hexadecimal stream constant
        if before.endswith(("/", "_")) or after.startswith(("/", "_")):
            return False  # path or filename-like token needs review
        if re.match(r"x\d+\b", after) or re.search(r"\d+x$", before):
            return True  # image raster dimension
        return bool(TECHNICAL_CONTEXT.search(line))

    for line in text.splitlines():
        candidates = [match for match in ID_TOKEN.finditer(line)
                      if match.group(1) in ids and not technical_number(line, match)]
        if not candidates:
            continue
        counts["matched_lines"] += 1
        if _data_path(name) or ID_CONTEXT.search(line) or STEM_CONTEXT.search(line):
            counts["errors"] += 1
        elif (name, _line_hash(line)) in ack:
            counts["acknowledged"] += 1
        else:
            counts["reviews"] += 1


def _configuration_schema(document: object) -> bool:
    """Recognize released config/metadata shapes, regardless of their file location."""
    if not isinstance(document, dict):
        return False
    keys = set(document)
    return (bool({"runtime", "server_budget"} <= keys)
            or bool({"config", "run", "source_run"} <= keys)
            or bool({"files", "scrubbed_paths", "retrained_register", "A", "B"} <= keys)
            or bool({"aireadi", "pools", "needs", "published_weights"} <= keys)
            or bool({"entries", "scans", "schema"} <= keys)
            or bool({"schema", "fingerprint", "manifest_rows", "lists"} <= keys))


def _classify_structured(name: str, document: object, ids: set[str],
                         ack: set[tuple[str, str]], counts: dict[str, int]) -> None:
    numeric_metadata = _configuration_schema(document)

    def classify(value: str, field: str, *, numeric: bool = False) -> None:
        if not _hits(value, ids):
            return
        if DIGEST_FIELD.search(field) and DIGEST_VALUE.fullmatch(value):
            return
        if numeric and numeric_metadata and not IDENTITY_FIELD.fullmatch(field):
            return
        counts["matched_lines"] += 1
        if (IDENTITY_FIELD.fullmatch(field) or PATH_FIELD.search(field)
                or ID_CONTEXT.search(value) or STEM_CONTEXT.search(value)):
            counts["errors"] += 1
        elif (name, _line_hash(value)) in ack:
            counts["acknowledged"] += 1
        else:
            counts["reviews"] += 1

    def walk(value: object, field: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key = str(key)
                classify(key, field)
                walk(child, key)
        elif isinstance(value, list):
            for child in value:
                walk(child, field)
        elif isinstance(value, str):
            classify(value, field)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            classify(str(value), field, numeric=True)

    walk(document)


def scan_blob(name: str, data: bytes, ids: set[str], ack: set[tuple[str, str]],
              counts: dict[str, int]) -> None:
    if _hits(name, ids):
        counts["errors"] += 1
    low = name.lower()
    suffix = Path(low).suffix
    try:
        if low.endswith(".gz"):
            text = gzip.decompress(data).decode("utf-8")
        elif suffix == ".png":
            text = _png_metadata(data).decode("utf-8", "replace")
        elif suffix in (".jpg", ".jpeg"):
            text = _jpeg_metadata(data).decode("utf-8", "replace")
        elif (suffix in TEXT_EXTENSIONS or Path(low).name in TEXT_EXTENSIONS
              or Path(low).name in ("license", "makefile", "dockerfile")
              or suffix == ""):
            text = data.decode("utf-8")
            if "\0" in text:
                raise ValueError("binary text file")
        else:
            counts["unknown_binary"] += 1
            counts["errors"] += 1
            return
    except (OSError, UnicodeError, ValueError, zlib.error):
        counts["errors"] += 1
        return
    if suffix in (".json", ".jsonl", ".yaml", ".yml"):
        try:
            if suffix == ".jsonl":
                documents = [json.loads(line) for line in text.splitlines() if line.strip()]
            elif suffix in (".yaml", ".yml"):
                import yaml
                documents = [yaml.safe_load(text)]
            else:
                documents = [json.loads(text)]
            for document in documents:
                _classify_structured(name, document, ids, ack, counts)
        except Exception:
            counts["errors"] += 1
    else:
        _classify_text(name, text, ids, ack, counts)


def _scan_weight_blob(repo: Path, name: str, oid: str, ids: set[str],
                      ack: set[tuple[str, str]], counts: dict[str, int]) -> None:
    if _hits(name, ids):
        counts["errors"] += 1
    with tempfile.TemporaryDirectory(prefix="octtta_idscan_") as directory:
        path = Path(directory) / Path(name).name
        with path.open("wb") as fh:
            _git("cat-file", "blob", oid, cwd=repo, stdout=fh)
        try:
            text = _weight_text(path)
        except Exception:
            counts["errors"] += 1
            return
    _classify_text(name, text, ids, ack, counts)
    counts["weights"] += 1


def scan(repo: Path, tree: str, participants: Path, ack_path: Path | None = None,
         weights: tuple[Path, ...] = ()) -> dict[str, int]:
    ids = participant_ids(participants)
    ack = _load_ack(ack_path, repo)
    counts = {"files": 0, "matched_lines": 0, "errors": 0, "reviews": 0,
              "acknowledged": 0, "unknown_binary": 0, "weights": 0}
    for name, oid in tree_blobs(repo, tree):
        counts["files"] += 1
        if Path(name.lower()).suffix in WEIGHT_EXTENSIONS:
            _scan_weight_blob(repo, name, oid, ids, ack, counts)
        else:
            scan_blob(name, _blob(repo, oid), ids, ack, counts)
    for path in weights:
        counts["files"] += 1
        try:
            text = _weight_text(path)
        except Exception:
            counts["errors"] += 1
            continue
        _classify_text(path.name, text, ids, ack, counts)
        counts["weights"] += 1
    return counts


def scan_worktree(repo: Path, participants: Path, ack_path: Path | None = None,
                  weights: tuple[Path, ...] = ()) -> dict[str, int]:
    ids = participant_ids(participants)
    ack = _load_ack(ack_path, repo)
    counts = {"files": 0, "matched_lines": 0, "errors": 0, "reviews": 0,
              "acknowledged": 0, "unknown_binary": 0, "weights": 0}
    for name in worktree_files(repo):
        path = repo / name
        counts["files"] += 1
        if path.is_symlink() or not path.is_file():
            counts["errors"] += 1
        elif path.suffix.lower() in WEIGHT_EXTENSIONS:
            if _hits(name, ids):
                counts["errors"] += 1
            try:
                text = _weight_text(path)
            except Exception:
                counts["errors"] += 1
            else:
                _classify_text(name, text, ids, ack, counts)
                counts["weights"] += 1
        else:
            try:
                scan_blob(name, path.read_bytes(), ids, ack, counts)
            except OSError:
                counts["errors"] += 1
    for path in weights:
        counts["files"] += 1
        try:
            text = _weight_text(path)
        except Exception:
            counts["errors"] += 1
            continue
        _classify_text(path.name, text, ids, ack, counts)
        counts["weights"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--participants", type=Path, required=True)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--tree", default="HEAD")
    source.add_argument("--worktree", action="store_true")
    ap.add_argument("--ack", type=Path)
    ap.add_argument("--weight", type=Path, action="append", default=[])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    repo = Path.cwd()
    if args.ack is None:
        if os.environ.get("OCTTTA_ID_SCAN_ACK"):
            args.ack = Path(os.environ["OCTTTA_ID_SCAN_ACK"])
        elif os.environ.get("OCTTTA_DATA"):
            args.ack = Path(os.environ["OCTTTA_DATA"]) / "release_id_scan" / "ack.tsv"
        elif len(args.participants.resolve().parents) >= 3:
            args.ack = args.participants.resolve().parents[2] / "release_id_scan" / "ack.tsv"
    try:
        if args.worktree:
            counts = scan_worktree(repo, args.participants, args.ack, tuple(args.weight))
        else:
            counts = scan(repo, args.tree, args.participants, args.ack, tuple(args.weight))
    except Exception as exc:
        print(f"ID_SCAN ERROR: scan could not complete ({type(exc).__name__})")
        return 2
    verdict = "PASS" if counts["errors"] == counts["reviews"] == 0 else "FAIL"
    report = {"verdict": verdict, **counts}
    if args.report:
        args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("ID_SCAN " + json.dumps(report, sort_keys=True))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
