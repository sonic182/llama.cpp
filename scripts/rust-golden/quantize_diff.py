#!/usr/bin/env python3
"""Run llama-quantize over a fixed argument matrix and record or compare the results.

The llama-imatrix next to the given binary also loads, converts and merges the
imatrix fixtures, which covers the Rust imatrix loader as tools/imatrix uses it.
Only its exit code and output files are compared: llama-imatrix logs through the
asynchronous common_log thread and sometimes loses its last lines at exit.

  quantize_diff.py fetch
  quantize_diff.py run  <llama-quantize> <out.json> [--lib-dir DIR]
  quantize_diff.py diff <expected.json> <actual.json>
  quantize_diff.py check <llama-quantize>   (run, then diff against quantize_golden.json)

stderr keeps only the lines llama-quantize and common/ write; the libllama log is
dropped, its effect is covered by the sha256 of the files each case produces.
stderr is compared as a multiset of lines: the C++ tool logged imatrix errors from
the asynchronous common_log thread, so their order against direct writes varied.
The stories260K F32 model is downloaded to $LLAMA_QUANT_FIXTURES (default
~/.cache/llama-quant-test). quantize/im.gguf and quantize/im.dat are imatrix files
that llama-imatrix produced from that model, one per format.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

FIXTURES = Path(os.environ.get("LLAMA_QUANT_FIXTURES", Path.home() / ".cache" / "llama-quant-test"))
MODEL = "stories260K-f32.gguf"
MODEL_URL = "https://huggingface.co/ggml-org/test-model-stories260K/resolve/main/stories260K-f32.gguf"
MODEL_SHA256 = "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d"
IMATRIX_DIR = Path(__file__).resolve().parent / "quantize"
IMATRIX_FILES = ["im.gguf", "im.dat"]
TEXT = "there was a little girl named lily she liked to play in the park\n"

TENSOR_TYPES = "attn_q=q8_0\n  ffn_down=Q5_0 output=f16\n"

M = MODEL
CASES = [
    [],
    ["--help"],
    [M],
    [M, "Q4_0"],
    [M, "out.gguf", "q4_0", "2"],
    [M, "out.gguf", "2", "2"],
    [M, "out.gguf", "15", "2"],
    [M, "out.gguf", " 7x", "2"],
    [M, "out.gguf", "0", "2"],
    [M, "out.gguf", "q3_k", "2"],
    [M, "out.gguf", "BAD"],
    [M, "out.gguf"],
    [M, "out.gguf", "COPY"],
    [M, "out.gguf", "Q4_0", "abc"],
    [M, "out.gguf", "Q4_0", "99999999999"],
    [M, "out.gguf", "Q4_0", "-3"],
    [M, M, "Q4_0"],
    [M, "./" + M, "Q4_0"],
    [M, "out.gguf", "Q1_0", "2"],
    ["--imatrix", "im.gguf", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "im.dat", M, "out.gguf", "IQ4_NL", "2"],
    ["--imatrix", "im.gguf", M, "out.gguf", "IQ4_NL", "2"],
    ["--imatrix", "im.gguf", "--include-weights", "attn", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "im.gguf", "--include-weights", "attn_q", "--include-weights", "blk.1.", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "im.dat", "--exclude-weights", "ffn_up", "--exclude-weights", "output", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "im.gguf", "--exclude-weights", "blk", M, "out.gguf", "Q4_0", "2"],
    ["--include-weights", "a", "--exclude-weights", "b", M, "out.gguf", "Q4_0"],
    ["--include-weights", "attn", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "missing.gguf", M, "out.gguf", "Q4_0"],
    ["--imatrix", "text.txt", M, "out.gguf", "Q4_0"],
    ["--imatrix"],
    ["--tensor-type", "attn_q=q8_0", "--tensor-type", "FFN_DOWN=Q5_0", M, "out.gguf", "Q4_0", "2"],
    ["--tensor-type", "noeq", M, "out.gguf", "Q4_0"],
    ["--tensor-type", "=q8_0", M, "out.gguf", "Q4_0"],
    ["--tensor-type", "attn_q=", M, "out.gguf", "Q4_0"],
    ["--tensor-type", "attn_q=zz", M, "out.gguf", "Q4_0"],
    ["--tensor-type-file", "tt.txt", M, "out.gguf", "Q4_0", "2"],
    ["--tensor-type-file", "missing.txt", M, "out.gguf", "Q4_0"],
    ["--output-tensor-type", "q8_0", "--token-embedding-type", "F16", M, "out.gguf", "Q4_0", "2"],
    ["--output-tensor-type", "nope", M, "out.gguf", "Q4_0"],
    ["--token-embedding-type"],
    ["--leave-output-tensor", "--pure", M, "out.gguf", "Q4_K_M", "2"],
    ["--allow-requantize", M, "out.gguf", "Q5_K_S", "2"],
    ["--prune-layers", "1,0,1", M, "out.gguf", "Q4_0", "2"],
    ["--prune-layers", "a", M, "out.gguf", "Q4_0"],
    ["--prune-layers", "1,,2", M, "out.gguf", "Q4_0"],
    ["--prune-layers", " 2x", M, "out.gguf", "Q4_0", "2"],
    ["--override-kv", "general.name=str:foo", "--override-kv", "test.i=int:42x", "--override-kv", "test.f=float:1.5e1",
     "--override-kv", "test.b=bool:true", M, "out.gguf", "Q4_0", "2"],
    ["--override-kv", "general.name=str:x", "--imatrix", "im.gguf", M, "out.gguf", "Q4_0", "2"],
    ["--override-kv", "x=bool:maybe", M, "out.gguf", "Q4_0"],
    ["--override-kv", "noeq", M, "out.gguf", "Q4_0"],
    ["--override-kv", "k=zz:1", M, "out.gguf", "Q4_0"],
    ["--override-kv", "k=str:" + "v" * 128, M, "out.gguf", "Q4_0"],
    ["--dry-run", M, "Q4_K"],
    ["--dry-run", M, "out.gguf", "Q4_0", "2"],
    ["--keep-split", M, "out.gguf", "Q4_0", "2"],
    ["--keep-split", M, "Q8_0", "2"],
    ["--max-buffer-size", "1", M, "out.gguf", "Q4_0", "2"],
    ["--max-buffer-size", "0", M, "out.gguf", "Q4_0"],
    ["--max-buffer-size", "abc", M, "out.gguf", "Q4_0"],
    ["--max-buffer-size"],
    ["--bogus", M, "out.gguf", "Q4_0"],
    ["-x", M, "out.gguf", "Q4_0"],
]

TRACE_CASES = [
    ["--imatrix", "im.gguf", M, "out.gguf", "Q4_0", "2"],
    ["--imatrix", "im.dat", M, "out.gguf", "Q4_0", "2"],
]

IMATRIX_CASES = [
    ["--in-file", "im.dat", "-o", "out.gguf"],
    ["--in-file", "im.gguf", "-o", "out.dat", "--output-format", "dat"],
    ["--in-file", "im.gguf", "-o", "out.gguf"],
    ["--in-file", "im.gguf", "--in-file", "im.dat", "-o", "out.gguf"],
    ["--in-file", "text.txt", "-o", "out.gguf"],
    ["--in-file", "missing.gguf", "-o", "out.gguf"],
]

TIME_RE = re.compile(r"^.*time = .*$", re.M)

LIBLLAMA_RE = re.compile(
    r"^(\[|llama_model_loader: |llama_model_quantize_impl: |llama_tensor_get_type: |validate_override: "
    r"|warning: |converting to |====== |\s|$)"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    model = FIXTURES / MODEL
    if not model.exists():
        urllib.request.urlretrieve(MODEL_URL, model)
    if sha256(model) != MODEL_SHA256:
        sys.exit(f"{model}: unexpected sha256")


def normalize(text, binary):
    text = text.replace(binary, "BIN")
    text = TIME_RE.sub("TIME", text)
    text = re.sub(r"^version: .*$", "version: V", text, flags=re.M)
    text = re.sub(r"^built with .*$", "built with C", text, flags=re.M)
    return text


def own_lines(text):
    return "".join(line for line in text.splitlines(keepends=True) if not LIBLLAMA_RE.match(line))


def run_case(binary, env, work, args, imatrix=False):
    before = set(os.listdir(work))
    cmd = [binary, "-m", MODEL, *args] if imatrix else [binary, *args]
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True)
    produced = {}
    for name in sorted(set(os.listdir(work)) - before):
        produced[name] = sha256(work / name)
        (work / name).unlink()
    stdout = normalize(proc.stdout.decode("utf-8", "replace"), binary)
    stderr = normalize(proc.stderr.decode("utf-8", "replace"), binary)
    return {
        "args": (["llama-imatrix"] if imatrix else []) + args,
        "rc": proc.returncode,
        "stdout": "" if imatrix else stdout,
        "stderr": "" if imatrix else own_lines(stderr),
        "files": produced,
    }


def run(binary, out, lib_dir):
    fetch()
    binary = str(Path(binary).resolve())
    env = dict(os.environ)
    env.pop("LLAMA_TRACE", None)
    if lib_dir:
        env["LD_LIBRARY_PATH"] = str(Path(lib_dir).resolve())
    results = []
    tmp_root = Path(os.environ.get("TMPDIR", Path.home() / ".cache"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        work = Path(tmp)
        shutil.copy(FIXTURES / MODEL, work / MODEL)
        for name in IMATRIX_FILES:
            shutil.copy(IMATRIX_DIR / name, work / name)
        (work / "text.txt").write_text(TEXT)
        (work / "tt.txt").write_text(TENSOR_TYPES)
        for args in CASES:
            results.append(run_case(binary, env, work, args))
        trace_env = dict(env, LLAMA_TRACE="1")
        for args in TRACE_CASES:
            results.append(dict(run_case(binary, trace_env, work, args), trace=True))
        imatrix = str(Path(binary).with_name("llama-imatrix"))
        for args in IMATRIX_CASES:
            results.append(run_case(imatrix, env, work, args, imatrix=True))
    Path(out).write_text(json.dumps(results, indent=1) + "\n")


def diff(expected, actual):
    a = json.loads(Path(expected).read_text())
    b = json.loads(Path(actual).read_text())
    if len(a) != len(b):
        sys.exit(f"case count differs: {len(a)} vs {len(b)}")
    failures = 0
    for x, y in zip(a, b):
        for key in ("rc", "files", "stdout", "stderr"):
            same = sorted(x[key].splitlines()) == sorted(y[key].splitlines()) if key == "stderr" else x[key] == y[key]
            if not same:
                failures += 1
                print(f"DIFF {key} for {x['args']}{' (trace)' if x.get('trace') else ''}")
                if key in ("stdout", "stderr"):
                    import difflib
                    for line in difflib.unified_diff(x[key].splitlines(), y[key].splitlines(), lineterm="", n=1):
                        print("   ", line)
                else:
                    print(f"    expected {x[key]}\n    actual   {y[key]}")
    print(f"{len(a)} cases, {failures} differences")
    sys.exit(1 if failures else 0)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) >= 4 and sys.argv[1] == "run":
        lib_dir = sys.argv[5] if len(sys.argv) >= 6 and sys.argv[4] == "--lib-dir" else None
        run(sys.argv[2], sys.argv[3], lib_dir)
    elif len(sys.argv) == 4 and sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        with tempfile.TemporaryDirectory() as tmp:
            actual = Path(tmp) / "actual.json"
            run(sys.argv[2], actual, None)
            diff(Path(__file__).resolve().parent / "quantize_golden.json", actual)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
