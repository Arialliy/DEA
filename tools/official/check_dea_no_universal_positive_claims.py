#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"DEA-lite\s+0\.005\s+improves\s+all",
    r"universally\s+improves",
    r"universally\s+reduces\s+false\s+alarms",
    r"solves\s+false\s+alarms",
    r"robust\s+across\s+all\s+datasets",
    r"NUAA\s+.*positive",
]

DEFAULT_SCAN_DIRS = ["docs", "repro_runs"]


def mask_forbidden_example_blocks(text: str) -> str:
    """Avoid flagging explicitly forbidden examples inside audit documents."""
    lines = text.splitlines(keepends=True)
    masked = []
    in_skip = False
    fence_count = 0
    for line in lines:
        lower = line.lower()
        if "forbidden claims" in lower or "do not claim" in lower:
            in_skip = True
            fence_count = 0
            masked.append("\n")
            continue
        if in_skip:
            if line.lstrip().startswith("```"):
                fence_count += 1
                if fence_count >= 2:
                    in_skip = False
            masked.append("\n")
            continue
        masked.append(line)
    return "".join(masked)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ly/DEA")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    violations = []
    for rel in DEFAULT_SCAN_DIRS:
        d = root / rel
        if not d.exists():
            continue
        for path in d.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            text = mask_forbidden_example_blocks(text)
            for pat in FORBIDDEN_PATTERNS:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    line = text[: m.start()].count("\n") + 1
                    violations.append({"file": str(path), "line": line, "pattern": pat})

    result = {"pass": len(violations) == 0, "violations": violations}
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
