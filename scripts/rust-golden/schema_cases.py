#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "test-json-schema-to-grammar.cpp"
TARGET = ROOT / "rust" / "llama-schema" / "tests" / "golden" / "cases.json"

LITERAL = r'(?:R"""\((.*?)\)"""|"((?:[^"\\]|\\.)*)")'
CASE = re.compile(
    r'test\(\{\s*(SUCCESS|FAILURE),\s*"((?:[^"\\]|\\.)*)",\s*'
    + LITERAL
    + r',\s*'
    + LITERAL
    + r'\s*\}\)',
    re.DOTALL,
)


def literal(raw: str | None, plain: str | None) -> str:
    return raw if raw is not None else json.loads(f'"{plain}"')


def main() -> int:
    text = SOURCE.read_text()
    cases = []
    for m in CASE.finditer(text):
        cases.append(
            {
                "name": m.group(2),
                "success": m.group(1) == "SUCCESS",
                "schema": literal(m.group(3), m.group(4)),
                "grammar": literal(m.group(5), m.group(6)),
            }
        )
    declared = len(re.findall(r"test\(\{\s*(?:SUCCESS|FAILURE),", text))
    if declared != len(cases):
        print(f"extracted {len(cases)} of {declared} cases", file=sys.stderr)
        return 1
    TARGET.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
