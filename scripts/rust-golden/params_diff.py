#!/usr/bin/env python3
"""Record or compare how common_params_parse fills common_params, through params_driver.

  params_diff.py record <params_driver> [out.json]   (build the corpus from the option tables and record it)
  params_diff.py check  <params_driver>              (rerun the cases of params_golden.json and compare)

Each case runs the driver in its own temporary directory with a clean environment
(HOME, XDG_CONFIG_HOME and LLAMA_CACHE inside it), so a user config.ini or model
cache cannot leak in. A case records the exit code, stdout, the stderr lines as a
sorted multiset (common_log writes from its own thread), the files the case left
behind, and the fields of common_params that differ from a default-constructed one.

The corpus comes from the option tables of the binary that records it: every option
of every example, with valid, invalid and environment-variable values, plus the
help, completion and exit paths of each example. The goldens are recorded from the
C++ parser, before the Rust port replaces it.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "params_golden.json"
MODEL = "model.gguf"
FIXTURES = {
    "model.gguf": "",
    "prompt.txt": "hello from a file\n",
    "grammar.gbnf": 'root ::= "yes" | "no"\n',
    "schema.json": '{"type": "object", "properties": {"a": {"type": "integer"}}}\n',
    "template.jinja": "{% for m in messages %}{{ m.content }}{% endfor %}",
    "list.txt": "one\ntwo\n",
    "config/llama.cpp/.keep": "",
    "dir/.keep": "",
}
EXIT_ONLY = {"--version", "--list-devices", "--cache-list", "--completion-bash", "-h", "--help", "--usage"}

VALUES = {
    "N": ["7", "-1", "abc"],
    "SECONDS": ["7", "abc"],
    "INDEX": ["1", "abc"],
    "PORT": ["9090", "abc"],
    "<0|1>": ["1", "0", "abc"],
    "0|1": ["1", "0"],
    "<0...100>": ["30", "abc"],
    "M": ["1", "abc"],
    "F": ["0.5", "abc"],
    "P": ["0.5", "abc"],
    "ALPHA": ["0.5", "abc"],
    "FRACTION": ["0.25", "abc"],
    "WD": ["0.1", "abc"],
    "SIMILARITY": ["0.5", "abc"],
    "L": ["0.5", "abc"],
    "lo-hi": ["0-0", "3", "abc"],
    "FNAME": ["prompt.txt", "missing.txt"],
    "FILE": ["prompt.txt", "missing.txt"],
    "PATH": ["dir", "missing"],
    "DIR": ["dir", "missing"],
    "JINJA_TEMPLATE_FILE": ["template.jinja", "missing.jinja"],
    "JINJA_TEMPLATE": ["chatml", "{{ messages }}", "bogus-template"],
    "GRAMMAR": ['root ::= "a"'],
    "SCHEMA": ['{"type": "string"}', "{bad"],
    "JSON": ['{"a": 1}', "{bad"],
    "n0,n1,...": ["1,2,3", "1,x"],
    "N0,N1,N2,...": ["1,2,3", "1,x"],
    "P0,P1,...": ["0.5,0.25", "x"],
    "MiB0,MiB1,MiB2,...": ["512", "512,256", "x"],
    "TOKEN_ID(+/-)BIAS": ["15+1", "15-inf", "x"],
    "KEY=TYPE:VALUE,...": ["a.b=int:1", "a.b=str:x,c.d=bool:true", "a.b=float:0.5", "bad"],
    "FNAME:SCALE,...": ["prompt.txt:0.5", "prompt.txt", "prompt.txt:x"],
    "<tensor name pattern>=<buffer type>,...": ["blk=CPU", "blk=nodev", "bad"],
    "<dev1,dev2,..>": ["none", "CPU", "nodev"],
    "DEVICE": ["none", "CPU", "nodev"],
    "SAMPLERS": ["top_k;temperature", "top-k;bogus"],
    "SEQUENCE": ["kt", "kz"],
    "SEED": ["42", "-1", "abc"],
    "N|auto": ["auto", "3", "abc"],
    "TOOL1,TOOL2,...": ["all", "read_file", "bogus"],
    "HOST": ["0.0.0.0", "unix.sock"],
    "URL": ["https://example.com/m.gguf"],
    "MODEL_URL": ["https://example.com/m.gguf"],
    "<user>/<model>[:quant]": ["user/model:Q4_K_M", "user/model"],
    "[<repo>/]<model>[:quant]": ["ai/smollm2:Q4", "smollm2"],
    "[on|off|auto]": ["on", "off", "auto", "bogus"],
    "[on|off]": ["on", "off", "bogus"],
    "{gguf,dat}": ["gguf", "dat", "bogus"],
    "{pca, mean}": ["pca", "mean", "bogus"],
    "{none,linear,yarn}": ["none", "linear", "yarn", "bogus"],
    "{none,layer,row,tensor}": ["none", "layer", "row", "tensor", "bogus"],
    "{none,mean,cls,last,rank}": ["none", "mean", "cls", "last", "rank", "bogus"],
    "{causal,non-causal}": ["causal", "non-causal", "bogus"],
    "sgd|adamw": ["sgd", "adamw", "bogus"],
    "TYPE": ["q8_0", "f16", "bogus"],
    "FORMAT": ["none", "deepseek", "auto", "bogus"],
    "MODE": ["auto", "none", "bogus"],
    "LEVEL": ["4", "abc"],
    "REGEX": ["blk\\.0\\..*", "("],
}
VALUES_BY_ARG = {
    "--spec-type": ["ngram-simple", "draft-simple,ngram-mod", "bogus"],
    "--reasoning-budget": ["0", "-1", "abc"],
    "--load-mode": ["auto", "mmap", "bogus"],
    "--lazy-mode": ["auto", "off", "bogus"],
    "--numa": ["distribute", "isolate", "numactl", "bogus"],
    "--chat-template-kwargs": ['{"enable_thinking": false}', "{bad"],
    "--override-tensor-draft": ["blk=CPU", "bad"],
    "--prio": ["1", "-1", "abc"],
    "--prio-batch": ["1", "3", "7"],
    "--spec-draft-prio": ["1", "3", "7"],
    "--spec-draft-prio-batch": ["1", "3", "7"],
    "--json-schema-file": ["schema.json", "missing.json"],
    "--ui-config-file": ["schema.json", "missing.json"],
}
STRING_FALLBACK = ["abc", "1,2", ""]


def example_tables(driver):
    tables = {}
    for ex in EXAMPLES:
        tables[ex] = json.loads(run_simple([driver, "--options", ex]))
    return tables


def run_simple(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


EXAMPLES = [
    "batched", "debug", "common", "speculative", "completion", "cli", "embedding", "perplexity",
    "retrieval", "passkey", "imatrix", "bench", "server", "cvector-generator", "export-lora", "mtmd",
    "lookup", "parallel", "tts", "diffusion", "finetune", "fit-params", "results", "export-graph-ops",
    "download", "tokenize",
]


def home_example(opt, tables):
    key = opt["args"]
    in_server = any(o["args"] == key and o["in_example"] for o in tables["server"])
    if in_server:
        return "server"
    for ex in EXAMPLES:
        if ex in ("common", "server"):
            continue
        if any(o["args"] == key and o["in_example"] for o in tables[ex]):
            return ex
    return "common"


def values_for(opt):
    for arg in opt["args"]:
        if arg in VALUES_BY_ARG:
            return VALUES_BY_ARG[arg]
    hint = opt["value_hint"]
    if hint in VALUES:
        return VALUES[hint]
    if hint and hint.startswith("none,draft"):
        return VALUES_BY_ARG["--spec-type"]
    return STRING_FALLBACK


def model_args(ex):
    return [] if ex in ("server", "download", "export-graph-ops") else ["-m", MODEL]


def build_cases(tables):
    cases = []
    seen = set()

    def add(ex, args, env=None):
        case = {"example": ex, "args": args, "env": env or {}}
        key = json.dumps(case, sort_keys=True)
        if key not in seen:
            seen.add(key)
            cases.append(case)

    for ex in EXAMPLES:
        add(ex, [])
        add(ex, model_args(ex))
        add(ex, ["-h"])
        add(ex, ["--completion-bash"])
        add(ex, ["--bogus-flag"])
        add(ex, model_args(ex), {"XDG_CONFIG_HOME": "config", "@config.ini": "[*]\nctx-size = 1234\nbogus-key = 1\n"})

    options = {}
    for ex in EXAMPLES:
        for opt in tables[ex]:
            if opt["in_example"] and not opt["is_preset_only"]:
                options.setdefault(tuple(opt["args"]), opt)

    for opt in options.values():
        ex = home_example(opt, tables)
        base = model_args(ex)
        if "-m" in opt["args"] or "--model" in opt["args"]:
            base = []
        exit_only = any(a in EXIT_ONLY for a in opt["args"])
        for arg in opt["args"]:
            if opt["handler"] in ("void", "bool"):
                add(ex, base + [arg])
            else:
                for value in values_for(opt):
                    add(ex, base + [arg, value])
                    if arg == opt["args"][0]:
                        continue
                    break
        for neg in opt["args_neg"]:
            add(ex, base + [neg])
        if opt["handler"] not in ("void", "bool") and not exit_only:
            add(ex, base + [opt["args"][0]])
        if opt["env"] and not exit_only:
            if opt["handler"] in ("void", "bool"):
                for value in ("1", "false", "x"):
                    add(ex, base, {opt["env"]: value})
            elif opt["handler"] != "str_str":
                for value in values_for(opt)[:2]:
                    add(ex, base, {opt["env"]: value})
                first = values_for(opt)[0]
                add(ex, base + [opt["args"][0], first], {opt["env"]: first})
        if opt["handler"] == "str_str":
            add(ex, base + [opt["args"][0], "a", "b"])
            add(ex, base + [opt["args"][0], "1", "2"])

    add("server", ["--ctx-size", "1", "--ctx-size", "2"])
    add("server", ["--spec-type", "ngram-simple", "--spec-type", "ngram-mod"])
    add("server", ["--ctx_size", "64"])
    add("server", ["-c", "64", "--escape", "-p", "a\\nb"])
    add("cli", ["-m", MODEL, "-p", "a\\nb", "-r", "x\\ty", "--no-escape"])
    add("cli", ["-m", MODEL, "--prompt-cache-all", "-i"])
    add("server", ["--tools", "all"])
    add("server", ["--tools", "all", "--cors-origins", "*"])
    add("server", ["-ot", "blk=CPU", "-otd", "blk=CPU"])
    add("server", ["--kv-override", "a.b=int:1"])
    add("server", ["--chat-template", "bogus-template", "--no-jinja"])
    add("server", ["--chat-template-kwargs", '{"preserve_reasoning": false}'])
    for batch in (["-Cb", "7"], ["-Crb", "0-3"], ["--cpu-strict-batch", "1"], ["--poll-batch", "1"], ["--prio-batch", "1"]):
        add("server", ["-tb", "1023", *batch])
    for draft in (["--spec-draft-cpu-mask", "7"], ["--spec-draft-cpu-range", "0-3"], ["--spec-draft-cpu-strict", "1"],
                  ["--spec-draft-poll", "1"], ["--spec-draft-prio", "1"]):
        add("server", ["-td", "1023", *draft])
    for draft in (["--spec-draft-cpu-mask-batch", "7"], ["--spec-draft-cpu-range-batch", "0-3"],
                  ["--spec-draft-cpu-strict-batch", "1"], ["--spec-draft-poll-batch", "1"], ["--spec-draft-prio-batch", "1"]):
        add("speculative", ["-m", MODEL, "-tbd", "1023", *draft])
    return cases


def flatten(value, path, out):
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(v, f"{path}.{k}" if path else k, out)
    elif isinstance(value, list) and len(value) > 16 and all(not isinstance(e, (dict, list)) for e in value):
        for i, v in enumerate(value):
            out[f"{path}[{i}]"] = v
    else:
        out[path] = value
    return out


def normalize(text, host_threads):
    text = re.sub(r"^version: .*$", "version: $VERSION", text, flags=re.M)
    text = re.sub(r"^built with .*$", "built with $COMPILER", text, flags=re.M)
    return re.sub(r"(requested thread count: )(\d+)",
                  lambda m: m.group(1) + ("$HOST_THREADS" if int(m.group(2)) in host_threads else m.group(2)), text)


def run_case(driver, case, host_threads):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, text in FIXTURES.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "LLAMA_CACHE": str(tmp_path / "cache"),
               "XDG_CONFIG_HOME": str(tmp_path / "xdg"), "LANG": "C"}
        for k, v in case["env"].items():
            if k == "@config.ini":
                p = tmp_path / "config" / "llama.cpp" / "config.ini"
                p.write_text(v)
            elif k == "XDG_CONFIG_HOME":
                env[k] = str(tmp_path / v)
            else:
                env[k] = v
        before = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")}
        proc = subprocess.run([driver, "out.json", case["example"], *case["args"]], cwd=tmp, env=env,
                              capture_output=True, text=True, errors="replace", timeout=60)
        out_file = tmp_path / "out.json"
        result = {"exit": proc.returncode}
        if out_file.exists():
            data = json.loads(out_file.read_text())
            out_file.unlink()
            a = flatten(data["before"], "", {})
            b = flatten(data["after"], "", {})
            changed = {}
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    v = b.get(k)
                    if k.endswith("n_threads") and v in host_threads:
                        v = "$HOST_THREADS"
                    changed[k] = v
            result["ok"] = data["ok"]
            result["params"] = changed
        after = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")}
        result["new_files"] = sorted(after - before)
        result["stdout"] = normalize(proc.stdout.replace(tmp, "$TMP"), host_threads)
        result["stderr"] = sorted(normalize(proc.stderr.replace(tmp, "$TMP"), host_threads).splitlines())
        return result


def host_threads(driver):
    found = set()
    for args in ([], ["-t", "-1"]):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([driver, "out.json", "server", *args], cwd=tmp, capture_output=True, check=True)
            data = json.loads((Path(tmp) / "out.json").read_text())
            found.add(data["after"]["cpuparams"]["n_threads"])
    return found


def run_all(driver, cases):
    threads = host_threads(driver)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(lambda c: run_case(driver, c, threads), cases))


def record(driver, out):
    cases = build_cases(example_tables(driver))
    results = run_all(driver, cases)
    lines = ",\n".join(json.dumps({"case": c, "result": r}) for c, r in zip(cases, results))
    Path(out).write_text(f"[\n{lines}\n]\n")
    ok = sum(1 for r in results if r.get("ok"))
    print(f"{len(cases)} cases recorded, {ok} parsed successfully")


def check(driver):
    golden = json.loads(GOLDEN.read_text())
    results = run_all(driver, [g["case"] for g in golden])
    failures = 0
    for g, r in zip(golden, results):
        if g["result"] != r:
            failures += 1
            if failures <= 20:
                print(f"case: {json.dumps(g['case'])}")
                for key in sorted(set(g["result"]) | set(r)):
                    if g["result"].get(key) != r.get(key):
                        print(f"  {key}:\n    expected {json.dumps(g['result'].get(key))[:400]}\n    actual   {json.dumps(r.get(key))[:400]}")
    print(f"{len(golden)} cases, {failures} differences")
    return failures


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(str(Path(sys.argv[2]).resolve()), sys.argv[3] if len(sys.argv) > 3 else GOLDEN)
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        sys.exit(1 if check(str(Path(sys.argv[2]).resolve())) else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
