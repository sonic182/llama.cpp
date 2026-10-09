#!/usr/bin/env python3
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "rust" / "llama-schema" / "src" / "grisu_table.rs"

MIN_DEC_EXP = -300
STEP = 8
COUNT = 79


def entry(k: int):
    value = Fraction(10) ** k
    e = value.numerator.bit_length() - value.denominator.bit_length() - 64
    while True:
        scaled = value / (Fraction(2) ** e)
        f = round(scaled)
        if f >= 1 << 64:
            e += 1
        elif f < 1 << 63:
            e -= 1
        else:
            return f, e


def main() -> int:
    rows = []
    for i in range(COUNT):
        k = MIN_DEC_EXP + STEP * i
        f, e = entry(k)
        rows.append((f, e, k))
    assert rows[0] == (0xAB70FE17C79AC6CA, -1060, -300), rows[0]
    assert rows[1] == (0xFF77B1FCBEBCDC4F, -1034, -292), rows[1]
    lines = ["pub struct CachedPower {", "    pub f: u64,", "    pub e: i32,", "    pub k: i32,", "}", "",
             f"pub const CACHED_POWERS: [CachedPower; {COUNT}] = ["]
    for f, e, k in rows:
        lines.append(f"    CachedPower {{ f: 0x{f:016X}, e: {e}, k: {k} }},")
    lines.append("];")
    TARGET.write_text("\n".join(lines) + "\n")
    print(f"{COUNT} entries -> {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
