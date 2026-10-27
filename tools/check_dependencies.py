"""Dependency policy check (S1-IMP-19, S1-IMP-20).

    python tools/check_dependencies.py [--root .]

Fails (exit 1) when any of the following holds:

1. **Cycles** — the import-time module graph has a cycle. Imports inside
   functions are lazy by design and only reported (``lazy-cycle``, a warning).
2. **Forbidden imports** — a module imports a downstream package
   (``apgi_system``): the dependency direction is framework → implementation →
   system, never back.
3. **Unpinned packages** — a dependency in ``pyproject.toml`` or
   ``requirements.txt`` has no upper bound or exact pin.
4. **Copied model files** — two source files have identical normalised content
   (comments and blank lines ignored, at least ``MIN_DUP_LINES`` lines), or a
   vendored ``apgi_framework`` tree is present.
5. **Deprecation drift** — ``adapters.version_adapter.DEPRECATIONS`` and the
   ``## Deprecations`` table of ``DEPENDENCY-MATRIX.md`` disagree, or a listed
   adapter module does not exist. A duplicate without a deprecation entry fails.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Finding", "run_checks", "main"]

#: Directories scanned for source modules.
SKIP_DIRS = {
    ".git",
    "build",
    "dist",
    ".venv",
    "venv",
    "__pycache__",
    "tests",
    "examples",
    "node_modules",
}
FORBIDDEN_IMPORTS = ("apgi_system",)
VENDORED_DIRS = ("apgi_framework", "apgi_system")
MIN_DUP_LINES = 15


@dataclass(frozen=True)
class Finding:
    """One policy finding. ``severity`` is ``error`` or ``warning``."""

    check: str
    severity: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


def _source_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        files.append(p)
    return files


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _ImportCollector(ast.NodeVisitor):
    def __init__(self, module: str, is_pkg: bool):
        self.module = module
        self.is_pkg = is_pkg
        self.eager: set[str] = set()
        self.lazy: set[str] = set()
        self._depth = 0

    def _target(self, name: str) -> set[str]:
        return {name}

    def _add(self, name: str) -> None:
        (self.lazy if self._depth else self.eager).add(name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_If(self, node: ast.If) -> None:
        # `if TYPE_CHECKING:` imports never execute.
        test = ast.unparse(node.test)
        if "TYPE_CHECKING" in test:
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            base = self.module.split(".")
            if not self.is_pkg:
                base = base[:-1]
            base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
            prefix = ".".join(base)
            mod = f"{prefix}.{node.module}" if node.module else prefix
        else:
            mod = node.module or ""
        if not mod:
            return
        self._add(mod)
        for alias in node.names:
            self._add(f"{mod}.{alias.name}")


def build_graph(root: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Path]]:
    """Return (eager edges, lazy edges, module → file) restricted to local modules."""
    files = _source_files(root)
    modules = {_module_name(root, p): p for p in files}
    eager: dict[str, set[str]] = {m: set() for m in modules}
    lazy: dict[str, set[str]] = {m: set() for m in modules}
    for mod, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        col = _ImportCollector(mod, path.name == "__init__.py")
        col.visit(tree)
        for kind, target in ((eager, col.eager), (lazy, col.lazy)):
            for name in target:
                resolved = _resolve(name, modules)
                if resolved and resolved != mod:
                    kind[mod].add(resolved)
    return eager, lazy, modules


def _resolve(name: str, modules: dict[str, Path]) -> str | None:
    """Longest local-module prefix of ``name``."""
    parts = name.split(".")
    while parts:
        cand = ".".join(parts)
        if cand in modules:
            return cand
        parts.pop()
    return None


def _sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    """Strongly connected components with more than one node (Tarjan, iterative)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for start in graph:
        if start in index:
            continue
        work = [(start, iter(sorted(graph[start])))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on.add(start)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on.add(nxt)
                    work.append((nxt, iter(sorted(graph[nxt]))))
                    advanced = True
                    break
                if nxt in on:
                    low[node] = min(low[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1:
                    out.append(sorted(comp))
    return out


def check_cycles(root: Path) -> list[Finding]:
    eager, lazy, _ = build_graph(root)
    findings = [
        Finding("cycle", "error", "import-time cycle: " + " <-> ".join(c)) for c in _sccs(eager)
    ]
    combined = {m: eager[m] | lazy[m] for m in eager}
    eager_sets = {frozenset(c) for c in _sccs(eager)}
    for comp in _sccs(combined):
        if frozenset(comp) not in eager_sets:
            findings.append(
                Finding(
                    "lazy-cycle", "warning", "cycle only via lazy imports: " + " <-> ".join(comp)
                )
            )
    return findings


def check_forbidden_imports(root: Path) -> list[Finding]:
    findings = []
    for path in _source_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in FORBIDDEN_IMPORTS:
                    findings.append(
                        Finding(
                            "forbidden-import", "error", f"{path.relative_to(root)} imports {n}"
                        )
                    )
    return findings


_REQ = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\])?\s*(.*)$")


def _pinned(spec: str) -> bool:
    """True if the specifier has an exact pin or an upper bound."""
    return bool(re.search(r"==|~=|<(?!=)|<=", spec.split(";")[0]))


def _load_toml(path: Path) -> dict | None:  # type: ignore[type-arg]
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            return None
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _requirement_lines(root: Path) -> list[tuple[str, str]]:
    """Runtime and optional dependency specifiers (build-system requires excluded)."""
    out: list[tuple[str, str]] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = _load_toml(pyproject)
        if data is not None:
            project = data.get("project", {})
            out.extend(("pyproject.toml", r) for r in project.get("dependencies", []))
            for reqs in project.get("optional-dependencies", {}).values():
                out.extend(("pyproject.toml", r) for r in reqs)
    req = root / "requirements.txt"
    if req.is_file():
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                out.append(("requirements.txt", line))
    return out


def check_pins(root: Path) -> list[Finding]:
    findings = []
    for origin, req in _requirement_lines(root):
        m = _REQ.match(req)
        if not m:
            continue
        if not _pinned(m.group(3)):
            findings.append(
                Finding("unpinned", "error", f"{origin}: {req!r} has no upper bound or exact pin")
            )
    return findings


def _normalised(path: Path) -> str:
    lines = []
    in_doc = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s.startswith(('"""', "'''")):
            if not (len(s) > 3 and s.endswith(s[:3])):
                in_doc = not in_doc
            continue
        if in_doc or not s or s.startswith("#"):
            continue
        lines.append(s)
    return "\n".join(lines)


def check_copied_models(root: Path) -> list[Finding]:
    findings = []
    seen: dict[str, Path] = {}
    for path in _source_files(root):
        if path.name == "__init__.py":
            continue
        body = _normalised(path)
        if body.count("\n") + 1 < MIN_DUP_LINES:
            continue
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest in seen:
            findings.append(
                Finding(
                    "copied-file",
                    "error",
                    f"{path.relative_to(root)} duplicates {seen[digest].relative_to(root)}",
                )
            )
        else:
            seen[digest] = path
    for d in VENDORED_DIRS:
        if (root / d).exists():
            findings.append(Finding("vendored", "error", f"vendored tree present: {d}/"))
    return findings


_ROW = re.compile(r"^\|(.+)\|\s*$")


def parse_matrix_deprecations(text: str) -> list[dict[str, str]]:
    """Rows of the ``## Deprecations`` table in DEPENDENCY-MATRIX.md."""
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "## deprecations")
    except StopIteration:
        return []
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        m = _ROW.match(line.strip())
        if m:
            rows.append([c.strip().strip("`") for c in m.group(1).split("|")])
    if len(rows) < 2:
        return []
    header = [h.lower() for h in rows[0]]
    return [dict(zip(header, r, strict=True)) for r in rows[2:] if len(r) == len(header)]


def check_deprecations(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    matrix = root / "DEPENDENCY-MATRIX.md"
    if not matrix.is_file():
        return [Finding("deprecation", "error", "DEPENDENCY-MATRIX.md is missing")]
    sys.path.insert(0, str(root))
    try:
        from adapters.version_adapter import DEPRECATIONS
    except ImportError as exc:
        return [Finding("deprecation", "error", f"cannot import adapters.version_adapter: {exc}")]
    finally:
        sys.path.pop(0)
    rows = {
        r.get("legacy name", ""): r
        for r in parse_matrix_deprecations(matrix.read_text(encoding="utf-8"))
    }
    declared = {d.old: d for d in DEPRECATIONS}
    for old, d in declared.items():
        row = rows.get(old)
        if row is None:
            findings.append(
                Finding(
                    "deprecation",
                    "error",
                    f"{old!r} is deprecated in code but absent from DEPENDENCY-MATRIX.md",
                )
            )
            continue
        for col, want in (
            ("canonical name", d.new),
            ("deprecated in", d.deprecated_in),
            ("removal release", d.removal_in),
            ("canonical owner", d.owner),
        ):
            if row.get(col) != want:
                findings.append(
                    Finding(
                        "deprecation",
                        "error",
                        f"{old!r}: matrix {col}={row.get(col)!r}, code says {want!r}",
                    )
                )
        adapter = row.get("adapter", "")
        mod_path = root / (adapter.replace(".", "/") + ".py")
        if not mod_path.is_file():
            findings.append(
                Finding("deprecation", "error", f"{old!r}: adapter module {adapter!r} not found")
            )
    for old in rows:
        if old and old not in declared:
            findings.append(
                Finding(
                    "deprecation",
                    "error",
                    f"{old!r} is in DEPENDENCY-MATRIX.md but has no DEPRECATIONS entry",
                )
            )
    return findings


def run_checks(root: Path) -> list[Finding]:
    """Run every check and return all findings."""
    root = root.resolve()
    return (
        check_cycles(root)
        + check_forbidden_imports(root)
        + check_pins(root)
        + check_copied_models(root)
        + check_deprecations(root)
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args(argv)
    findings = run_checks(Path(args.root))
    for f in findings:
        print(f)
    errors = [f for f in findings if f.severity == "error"]
    print(f"{len(errors)} error(s), {len(findings) - len(errors)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
