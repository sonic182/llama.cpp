#!/usr/bin/env python3
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "rust" / "llama-schema" / "tests" / "golden" / "cases.json"

NAMES = ["a", "b", "name", "id", "x-y", "a b", "ñ", "q\"uote", "*", "a-b", "ref", "string", "root", "space", "é"]
PATTERN_ATOMS = [
    "a", "b", "abc", "[a-z]", "[0-9]", "[^a-z]", "\\d", "\\w", ".", "(a|b)", "(?:ab)", "(?=a)", "\\.", "\\\"", '"',
    "a*", "b+", "c?", "[a-c]{2,3}", "a{2}", "x{1,}", "x{,3}", "x{ 3}", "{", "}", "]", "\\", "\\x41", "\\u00e9", "é",
    "(a(b(c)))", "a|b|c", "[\\]]", "[\\d]", "^", "$", "\\n", "\\t", "(", ")", "|", "+", "?", "{2,1}", "{99999999999}",
]
FORMATS = ["date", "time", "date-time", "uuid", "uuid1", "uuid5", "uuid6", "email", "ipv4", ""]
SCALARS = [None, True, False, 0, 1, -1, 2**31, 2**63, 2**64, -(2**63), 1.0, 0.5, -0.0, 1e20, 1e-5, 1.5e-7,
           123456.789, 0.1, 1e15, 1e16, 3.141592653589793, "", "a", "a b", "é", "a\"b", "a\\b", "a/b", "\n\t\r",
           "\u0001", "\u007f", "😀", " ", "*", "1.0"]


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    def pick(self, items):
        return self.r.choice(items)

    def chance(self, p):
        return self.r.random() < p

    def int_bound(self):
        r = self.r
        kind = r.random()
        if kind < 0.35:
            return r.randint(-20, 20)
        if kind < 0.6:
            return r.randint(-1000, 100000)
        if kind < 0.8:
            return r.choice([10, 99, 100, 101, 999, 1000, 9, 19, 90, 1234, 5000, 99999])
        if kind < 0.9:
            return r.randint(-(2**40), 2**40)
        if kind < 0.95:
            return r.choice([2**31, 2**63 - 1, -(2**63), 2**63, 2**64])
        return r.choice([0.5, -0.5, 3.0, 2.9999, 1e3, 1e30, -1e30])

    def scalar(self):
        r = self.r
        if self.chance(0.25):
            kind = r.random()
            if kind < 0.3:
                return r.uniform(-1e3, 1e3)
            if kind < 0.6:
                return r.random() * 10.0 ** r.randint(-30, 30)
            if kind < 0.8:
                return float(r.randint(-(10**6), 10**6))
            return float(r.randint(-(10**18), 10**18)) * 10.0 ** r.randint(-5, 5)
        return self.pick(SCALARS)

    def value(self, depth=0):
        r = self.r
        if depth < 2 and self.chance(0.2):
            return [self.value(depth + 1) for _ in range(r.randint(0, 3))]
        if depth < 2 and self.chance(0.2):
            return {self.pick(NAMES): self.value(depth + 1) for _ in range(r.randint(0, 3))}
        return self.scalar()

    def pattern(self):
        atoms = [self.pick(PATTERN_ATOMS) for _ in range(self.r.randint(0, 5))]
        body = "".join(atoms)
        roll = self.r.random()
        if roll < 0.85:
            return "^" + body + "$"
        if roll < 0.92:
            return body
        return "^" + body

    def schema(self, depth=0, defs=None):
        r = self.r
        defs = defs if defs is not None else []
        if depth > 3:
            return self.leaf()
        roll = r.random()
        if roll < 0.08 and defs:
            return {"$ref": "#/$defs/" + self.pick(defs)}
        if roll < 0.14:
            key = self.pick(["anyOf", "oneOf"])
            return {key: [self.schema(depth + 1, defs) for _ in range(r.randint(1, 3))]}
        if roll < 0.2:
            return {"allOf": [self.schema(depth + 1, defs) for _ in range(r.randint(1, 3))]}
        if roll < 0.23:
            return {"const": self.value()}
        if roll < 0.28:
            return {"enum": [self.value() for _ in range(r.randint(1, 4))]}
        if roll < 0.31:
            types = r.sample(["string", "integer", "number", "boolean", "null", "object", "array"], r.randint(1, 3))
            s = {"type": types}
            if self.chance(0.4):
                s["minimum"] = self.int_bound()
            return s
        return self.typed(depth, defs)

    def leaf(self):
        r = self.r
        kind = r.random()
        if kind < 0.3:
            return {"type": "string"}
        if kind < 0.5:
            return {"type": "integer"}
        if kind < 0.6:
            return {"type": "number"}
        if kind < 0.7:
            return {"type": "boolean"}
        if kind < 0.75:
            return {"type": "null"}
        if kind < 0.8:
            return {}
        return {"const": self.scalar()}

    def typed(self, depth, defs):
        r = self.r
        t = self.pick(["object", "object", "array", "array", "string", "string", "integer", "integer", "number",
                       "boolean", "null", None, None, "bogus", 5])
        s = {} if t is None else {"type": t}
        if t == "integer" or (t is None and self.chance(0.1)):
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                if self.chance(0.35):
                    s[key] = self.int_bound()
        if t == "string" or (t is None and self.chance(0.2)):
            if self.chance(0.4):
                s["pattern"] = self.pattern()
            if self.chance(0.3):
                s["format"] = self.pick(FORMATS)
            if self.chance(0.3):
                s["minLength"] = r.choice([0, 1, 2, 5, -1, 2**32 + 3])
            if self.chance(0.3):
                s["maxLength"] = r.choice([0, 1, 3, 10, -2])
            if self.chance(0.1):
                s["allOf"] = [self.schema(depth + 1, defs)]
        if t == "array" or (t is None and self.chance(0.2)):
            roll = r.random()
            if roll < 0.4:
                s["items"] = self.schema(depth + 1, defs)
            elif roll < 0.55:
                s["items"] = [self.schema(depth + 1, defs) for _ in range(r.randint(0, 3))]
            elif roll < 0.7:
                s["prefixItems"] = [self.schema(depth + 1, defs) for _ in range(r.randint(1, 3))]
            if self.chance(0.4):
                s["minItems"] = r.choice([0, 1, 2, 5])
            if self.chance(0.4):
                s["maxItems"] = r.choice([0, 1, 2, 5, 10])
        if t == "object" or (t is None and self.chance(0.25)):
            if self.chance(0.8):
                names = r.sample(NAMES, r.randint(0, 4))
                s["properties"] = {n: self.schema(depth + 1, defs) for n in names}
                if names and self.chance(0.6):
                    s["required"] = r.sample(names, r.randint(0, len(names)))
            roll = r.random()
            if roll < 0.2:
                s["additionalProperties"] = True
            elif roll < 0.4:
                s["additionalProperties"] = False
            elif roll < 0.5:
                s["additionalProperties"] = self.schema(depth + 1, defs)
            elif roll < 0.53:
                s["additionalProperties"] = 3
            if self.chance(0.08):
                s["allOf"] = [self.schema(depth + 1, defs) for _ in range(r.randint(1, 2))]
        return s

    def document(self):
        defs = {}
        names = [f"d{i}" for i in range(self.r.randint(0, 3))]
        root = self.schema(0, names)
        for n in names:
            defs[n] = self.schema(1, names)
        if defs:
            if self.chance(0.5) and isinstance(root, dict):
                root["$defs"] = defs
            else:
                root = {"$ref": "#/$defs/" + names[0], "$defs": defs}
        if self.chance(0.03):
            root = self.pick([[], 5, "x", None, {"$ref": "#/nope"}, {"$ref": "http://x"}, {"$ref": 3}])
        return root


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = sys.stdout
    for case in json.loads(CASES.read_text()):
        parsed = _parse(case["schema"])
        if parsed is not _INVALID:
            out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
    gen = Gen(seed)
    for _ in range(count):
        out.write(json.dumps(gen.document(), ensure_ascii=False) + "\n")
    return 0


_INVALID = object()


def _parse(text):
    try:
        return json.loads(text)
    except ValueError:
        return _INVALID


if __name__ == "__main__":
    sys.exit(main())
