#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_PATTERNS = [
    r"\bDEA\s+improves\b",
    r"\bDEA\s+reduces\s+false\s+alarms\b",
    r"\bDEA\s+solves\b",
    r"\bDEA\s+achieves\b",
    r"\bDEA\s+outperforms\b",
    r"\bDEA\s+is\s+effective\b",
    r"\bDEA\s+demonstrates\s+superior\b",
    r"\bfull\s+DEA\s+is\s+effective\b",
    r"\bFull\s+DEA\s+results\b",
    r"\bproposed\s+DEA\s+method\b",
    r"\bDEA-lite\s+validates\s+(the\s+)?full\s+DEA\b",
    r"\bDEA-lite\s+proves\s+(the\s+)?DEA\b",
    r"\bDEA-lite\s+is\s+(the\s+)?(proposed\s+)?full\s+DEA\b",
    r"\bDEA-lite\s+universally\s+improves\b",
    r"\buniversally\s+improves\s+IRSTD\b",
    r"\bDEA-lite\s+is\s+AAAI-ready\b",
    r"\bDEA-lite\s+is\s+our\s+main\s+method\b",
    r"\bDEA-lite\s+serves\s+as\s+the\s+full\s+DEA\b",
    r"DEA\s*显著提升",
    r"DEA\s*降低虚警",
    r"DEA\s*有效",
    r"DEA-lite\s*验证了\s*DEA",
    r"DEA-lite\s*就是\s*完整\s*DEA",
    r"DEA-lite\s*作为\s*主方法",
]

NEGATION_HINTS = [
    "do not claim",
    "do not write",
    "do not describe",
    "forbidden",
    "not allowed",
    "must not",
    "should not",
    "cannot claim",
    "不要",
    "不能",
    "禁止",
    "不应",
]

SUPPRESSED_SECTION_HINTS = [
    "forbidden claims",
    "forbidden_next_steps",
    "non-goals",
    "no-go",
    "do not claim",
    "it should not be described as",
    "full dea must not be implemented as merely",
    "不要",
    "不能",
    "禁止",
    "不应",
]

DEFAULT_SCAN_TARGETS = [
    "README.md",
    "docs",
]


def is_negated_context(lines: list[str], idx: int) -> bool:
    start = max(0, idx - 6)
    context = "\n".join(lines[start : idx + 1]).lower()
    return any(hint in context for hint in NEGATION_HINTS)


def is_suppressed_example_context(lines: list[str], idx: int) -> bool:
    start = max(0, idx - 20)
    context = "\n".join(lines[start : idx + 1]).lower()
    return any(hint in context for hint in SUPPRESSED_SECTION_HINTS)


def iter_markdown_files(root: Path, targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        p = (root / target).resolve()
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst"}:
            files.append(p)
        elif p.is_dir():
            files.extend(
                q
                for q in p.rglob("*")
                if q.is_file() and q.suffix.lower() in {".md", ".txt", ".rst"}
            )
    return sorted(set(files))


def scan_file(path: Path, root: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    violations: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                if is_negated_context(lines, idx) or is_suppressed_example_context(lines, idx):
                    continue
                violations.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": idx + 1,
                        "pattern": pat,
                        "text": line.strip(),
                    }
                )

    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ly/DEA")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_SCAN_TARGETS,
        help="Files or directories relative to root.",
    )
    parser.add_argument(
        "--output",
        default="docs/internal/dea_lite/DEA_LITE_FULL_DEA_CLAIM_GUARD.json",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = iter_markdown_files(root, args.targets)
    violations: list[dict[str, Any]] = []
    for path in files:
        violations.extend(scan_file(path, root))

    result: dict[str, Any] = {
        "guard": "check_no_full_dea_claim_from_dea_lite",
        "root": str(root),
        "scanned_files": [str(p.relative_to(root)) for p in files],
        "pass": len(violations) == 0,
        "violations": violations,
        "decision": "PASS_NO_FULL_DEA_CLAIM_FROM_DEA_LITE"
        if not violations
        else "FAIL_FULL_DEA_CLAIM_FROM_DEA_LITE_FOUND",
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
