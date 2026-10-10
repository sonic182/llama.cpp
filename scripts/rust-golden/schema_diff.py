#!/usr/bin/env python3
import subprocess
import sys


def run(driver: str, corpus: str, out_path: str) -> None:
    lines = open(corpus, encoding="utf-8").read().splitlines()
    results = []
    errors = b""
    start = 0
    while start < len(lines):
        proc = subprocess.run(
            driver.split(),
            input=("\n".join(lines[start:]) + "\n").encode("utf-8"),
            capture_output=True,
        )
        errors += proc.stderr
        done = proc.stdout.decode("ascii").splitlines()
        results.extend(done)
        start += len(done)
        if start < len(lines) and proc.returncode != 0:
            results.append("crash")
            start += 1
        elif start < len(lines):
            raise SystemExit(f"driver stopped at line {start} with code {proc.returncode}")
    with open(out_path, "w") as f:
        f.write("\n".join(results) + "\n")
    with open(out_path + ".err", "wb") as f:
        f.write(errors)


def decode(entry: str) -> str:
    kind, _, payload = entry.partition(" ")
    return f"{kind} {bytes.fromhex(payload).decode('utf-8', 'replace')}" if payload else kind


def _utf8(entry: str) -> bool:
    try:
        bytes.fromhex(entry.partition(" ")[2]).decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def compare(cpp_path: str, rust_path: str, corpus: str, limit: int = 5) -> int:
    cpp = open(cpp_path).read().splitlines()
    rust = open(rust_path).read().splitlines()
    lines = open(corpus, encoding="utf-8").read().splitlines()
    assert len(cpp) == len(rust) == len(lines), (len(cpp), len(rust), len(lines))
    stats = {"same": 0, "cpp_crash": 0, "cpp_invalid_utf8": 0, "diff": 0}
    shown = 0
    for i, (a, b) in enumerate(zip(cpp, rust)):
        if a == b:
            stats["same"] += 1
        elif a == "crash" and b.startswith("err"):
            stats["cpp_crash"] += 1
        elif a.startswith("ok ") and b.startswith("ok ") and not _utf8(a):
            stats["cpp_invalid_utf8"] += 1
        else:
            stats["diff"] += 1
            if shown < limit:
                shown += 1
                print(f"--- line {i + 1}: {lines[i][:400]}")
                da, db = decode(a), decode(b)
                for la, lb in zip(da.splitlines(), db.splitlines()):
                    if la != lb:
                        print(f"cpp : {la[:500]}")
                        print(f"rust: {lb[:500]}")
                        break
                else:
                    print(f"cpp : {da[:300]}")
                    print(f"rust: {db[:300]}")
    warnings_equal = open(cpp_path + ".err", "rb").read() == open(rust_path + ".err", "rb").read()
    print(stats, "warnings_equal" if warnings_equal else "WARNINGS DIFFER")
    return 1 if stats["diff"] or not warnings_equal else 0


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3], sys.argv[4]))
