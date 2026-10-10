# Plan: migrate to Rust everything except inference (branch `rust-migration`)

## Context

Goal: incrementally move to Rust everything that is not inference, leaving ggml and the core of `src/` in C/C++. The HTTP layer moves to tokio. The `rust-migration` branch comes from `qwen3-cpu-kernel-experiment` (HEAD b347ad950). The untracked files (`docker-compose.yml`, `scripts/evals/phase2-affinity-sweep.sh`, `scripts/evals/qwen3-0.6b-phase2/`) are not touched.

Real dimension (lines of C/C++):

| Area | Lines | Target |
|---|---|---|
| `ggml/src` | 393k | C/C++ |
| `src/` | 100k | C/C++ |
| `tools/mtmd` | 27k | C/C++ (uses ggml directly; C API: `tools/mtmd/mtmd.h`, `mtmd-helper.h`) |
| `common/` | 41k | Rust |
| `tools/server` | 22k | Rust |
| rest of `tools/` | ~12k | Rust |

About 75k of ~600k lines are migrated. More than 85% remains in C/C++.

Exit state: the original C++ is kept in the tree and compiles until each Rust piece reaches parity. It is deleted only after. The migration is opt-in with `-DLLAMA_RUST=ON` (default OFF).

## Findings that fix the design

1. **The boundary is `llama.h`.** `common/` and `tools/` (without mtmd) only include one internal header: `src/llama-ext.h` (134 lines), from `common/speculative.cpp:13`, `common/fit.cpp:6`, `tools/fit-params/fit-params.cpp:2` and `tools/mtmd/mtmd-helper-gen.cpp:5`. Everything else goes through `include/llama.h` (259 `LLAMA_API`), `gguf.h`, `ggml-backend.h` and `mtmd*.h`.
2. **`src/llama-ext.h` is not FFI-safe.** `:84` defines `llama_memory_breakdown` as `std::map`, `:91` returns it, and `:22` exposes `quantize_state_impl`. The MTP/nextn/dflash functions (`:96-134`) are a moving API. A small C shim is needed (~8 functions).
3. **The HTTP layer already has a clean seam.** `tools/server/server-http.h:60-97` defines `server_http_context` with pimpl. The handlers are `handler_t = std::function<server_http_res_ptr(const server_http_req&)>`. There are 38 registrations `ctx_http.get/post/del` in `tools/server/server.cpp` and 26 handlers in `server_routes` (`tools/server/server-context.h:131-156`). The handlers in router mode (`tools/server/server-models.h:359-366`: `proxy_get`, `proxy_post`, `get_router_*`) use the same type.
4. **The scheduler is an actor with condvars.** `tools/server/server-queue.h:15-150` (`server_queue`), `:154-198` (`server_response`) and `:203-244` (`server_response_reader`). In Rust it is a dedicated `std::thread` for the decode loop (never a tokio worker, `llama_decode` blocks), with `mpsc` for input tasks and one `mpsc` per request for results. It moves from pull (`server_http_res::next`, `server-http.h:30`) to push, and cancellation comes from dropping the stream.
5. **The bulk is concentrated.** `tools/server/server-context.cpp`: `server_context_impl` at `:833-4151`, `server_slot` at `:239-735`, `server_routes::init_routes` at `:4650-5288` (mostly JSON), `handle_completions_impl` at `:4257-4523`.
6. **Two custom samplers in C++.** `common/llguidance.cpp:108` and `common/reasoning-budget.cpp:209` implement `llama_sampler_i`. In Rust they are an `extern "C"` vtable. llguidance is already a Rust crate and is currently consumed via its C API.
7. **Test oracles.** `tools/server/tests/unit/*.py` are 248 pytest blackbox tests over HTTP. They are run with `tools/server/tests/tests.sh` and `LLAMA_SERVER_BIN_PATH` (`tools/server/tests/utils.py:151`), so they work unchanged against a Rust binary. The C++ tests in `tests/` (`test-jinja.cpp`, `test-chat*.cpp`, `test-sampling.cpp`, `test-arg-parser.cpp`, `test-json-schema-to-grammar.cpp`, registered in `tests/CMakeLists.txt:157-299`) are not directly reusable: differential testing is used (same input to C++ and Rust).
8. **Project hardware.** The host is a Ryzen 5 3550H (4C/8T, SMT) and `todos/qwen3-0.6b-cpu-performance.md` sets the decision rule (improvement >=2%, regression <=1%) and pinning `--cpu-strict 1` on CPUs 0,2,4,6. The tokio layer cannot disturb that threadpool.

## Design rule: data structures

- Data structures stay in C/C++ by default. This covers `common_params` and its sub-structs, `common_params_sampling`, `common_params_model`, `common_chat_msg`, `common_chat_tool`, `common_chat_params`, `common_json`, the structs of `server-task.h` and `server_slot`.
- Whether a data structure moves to Rust is decided case by case, asking the user when the phase reaches it. Nothing moves without that answer.
- Rust provides logic behind a C layer: functions that take and return text (JSON, grammar, prompt) or opaque handles. If the layer needs fields, the C++ adapter copies them into flat C structs.
- Rust data types that already exist are accepted for now (see Pending decisions).

## Phase 0 - Scaffolding (no behavior change)

Legend: `[x]` done and committed, `[ ]` pending.

- [x] New workspace `rust/` (`Cargo.toml`, `rust-toolchain.toml` with rustc 1.98.1 installed on the host). Initial crates:
  - [x] `rust/llama-sys`: bindgen from `include/llama.h`, `ggml/include/ggml.h`, `gguf.h`, `ggml-backend.h`, `tools/mtmd/mtmd.h`, `mtmd-helper.h`.
  - [x] `rust/llama`: minimal safe wrapper (model, context, batch, vocab).
- [x] New C lib `common/rust-shim/` (`llama_ext_c.h/.cpp`): wraps `src/llama-ext.h` with C types (memory breakdown to flat array, quantize helpers).
- [x] CMake: `LLAMA_RUST` option alongside those in `CMakeLists.txt:129-146`. CMake continues building `libllama`/ggml (kernel flags matter) and `cargo` links against those libs (`LLAMA_LIB_DIR`). No Corrosion initially.
- [x] Document the build flow in `docs/build.md` or a `rust/README.md` (done in `rust/README.md`).

## Phase 1 - Small tools as landing strip

- [x] `tools/tokenize/tokenize.cpp:95-222` (uses `arg.h`/`common.h` only for parsing: replaced with `clap`).
- [x] `tools/gguf-split/gguf-split.cpp:203-591` (`split_strategy`, `gguf_split`, `gguf_merge`; only `gguf.h`/`ggml.h`).
- [ ] `quantize` (`tools/quantize/quantize.cpp:395-668`) remains for Phase 3: depends on `common/imatrix-loader.cpp` and `llama-ext.h`.
- [x] Validation: differential test against the C++ binary (same input GGUFs, compare output byte for byte).

## Phase 2 - HTTP with tokio behind the existing interface

Call stack (where the change lands):

```
tools/server/main.cpp                               <- entry point of llama-server
  tools/server/server.cpp:88   llama_server(argc, argv)
    tools/server/server.cpp:119 llama_server(params, ...)
      server.cpp:200   ctx_http.init(params)           -> server-http.cpp:111
      server.cpp:251+  ctx_http.get/post/del x38       -> server-http.cpp:658 / 679 / 730
      ctx_http.start()                                  -> server-http.cpp:476
        httplib handler
          process_handler_response                      server-http.cpp:618
            handler_t(req) -> server_http_res           <- C++ stays the same; next() is pull
```

Why here and not higher or lower: `server_http_context` already isolates all httplib (39 uses in `server-http.cpp`; `server-models.cpp` has another 14 that remain until Phase 5), and the handlers are a stable type that also serves the router. C++ keeps `main`, args parsing (`common_params`), the scheduler, sampling and chat. Only the transport is replaced.

Changes:
- [x] `server-http.cpp:19-26` (`server_http_context::Impl`): Rust variant, now the only server transport: `LLAMA_RUST_HTTP` and the cpp-httplib server path are gone, and the glue lives in `tools/server/server-http.cpp`. New crate `rust/llama-http` (tokio + hyper/axum) compiled as `staticlib` with a C ABI: `init(config)`, `register(method, path, cb)`, `start`, `stop`, `join`. The `cb` is wrapped `handler_t`: receives method/path/params/headers/body/files and returns status/headers and body or chunks.
- Port to Rust what lives in `server-http.cpp:158-474` (`init_listener`) and `:797-923` (`register_gcp_compat`):
  - [x] hosts, port, unix sockets, `SO_REUSEPORT`
  - [x] API keys and CORS
  - [x] Embedded static UI (assets, gzip, ETag/304)
  - [x] `--path` (static directory: files, `index.html`, 301, `If-None-Match`; no range or compression)
  - [x] read timeout (`header_read_timeout`)
  - [x] write timeout (`--timeout` applies to a write that makes no progress; stalled readers get the connection closed and the stream cancelled)
  - [x] stale unix socket (leftover `.sock` file is removed only when nothing is listening on it)
  - [x] SSL (see TLS below)
  - [x] `register_gcp_compat`: remains in C++ as shared code and works unchanged (not ported to Rust)
- [x] Streaming: each in-flight request uses `spawn_blocking` and calls `next()` in a loop (`server-http.h:30`). It is equivalent to thread-per-connection in httplib today. Bounded blocking pool. `should_stop` (`server-http.h:56`) maps to an `AtomicBool` that activates when dropping the hyper body stream.
- Tokio runtime with few workers and configurable affinity to not compete with CPUs 0,2,4,6 of the ggml threadpool:
  - [x] configurable workers with `--http-workers N|auto`
  - [ ] CPU affinity for workers (only if measurement justifies it)
- [x] TLS: `rustls` behind the same flag as `LLAMA_OPENSSL` (`CMakeLists.txt:144`).

## Phase 3 - `common/` from leaves inward (bottom-up)

Order (from the phase 3 plan file), each piece with differential test C++ vs Rust (jinja/chat are the highest fidelity risk). `arg.cpp` moved to Phase 4:

- [x] 3.0 groundwork: umbrella crate `rust/llama-rs` (one staticlib, `llama_rs_version` round trip from `libllama-common`), `add_subdirectory(rust)` moved before `common`, single copy of Rust `std` checked with `nm`. Golden-vector script `scripts/rust-golden/` lands with 3.1.
- [x] `json-schema-to-grammar` (`common/json-schema-to-grammar.cpp`, 1028 lines): `rust/llama-schema`, always used by `json_schema_to_grammar(json)`: `LLAMA_RUST_SCHEMA` and the C++ branch of that function are gone, and `scripts/rust-golden/schema_driver.cpp` calls the C++ converter directly as the oracle (`driver rust` calls the Rust path; `schema_diff.py run "./driver rust" ...`). Re-checked after the switch on a 45k corpus: 0 grammar diffs; 2 warnings differ because the C++ converter cuts a multi-byte UTF-8 character after `\` in an unsupported pattern escape, which the Rust message keeps whole. The pure schema-to-grammar function is ported; `build_grammar(cb)` and the `common_chat_schema` AST stay in C++ until 3.5/3.6. A Rust `trie.rs` exists in `llama-schema`, while `common/trie.cpp` is still used by C++ (reasoning-budget, peg-parser); see Pending decisions. Verified with the three C++ tests with the flag ON, 81 golden cases, a 45k-schema differential against the C++ build and the server pytest (`test_chat_completion`, `test_compat_anthropic`, `test_basic`: 90 pass). Built and passing with `BUILD_SHARED_LIBS=OFF` (`test-json-schema-to-grammar`, `test-json-schema`, `test-grammar-integration`). Static builds skip the `llama-gguf-split` and `llama-tokenize` binaries, because `llama-sys` always links libllama as a shared library.
- [x] 3.2 `download` + `hf-cache`. Done in Rust (`rust/llama-download`) and wired through C++ shims with unchanged headers: `common/download.cpp` (about 1090 to 264 lines; keeps `ProgressBar`, `is_output_a_tty`, `is_http_status_ok` and `common_download_run_tasks`) and `common/hf-cache.cpp` (513 to 28 lines). `llama-download` is a fixed dependency of `llama-rs`, so the `LLAMA_RUST_DOWNLOAD` flag is gone; `LLAMA_RUST` now defaults to ON, because `common/` fails at configure without it. Checks: `tests/golden.rs` replays 92 vectors recorded from `common/download.cpp`; `tests/cache.rs` replays a cache scenario against the C++ oracle; `tests/remote.rs` covers a plain URL (200, etag, 304, resume, offline); `tests/docker.rs` resolves `ai/smollm2:135M-Q2_K`. Against the old C++ build: Docker `ai/smollm2:135M-Q4_0` gives the same sha256 and etag; a plain URL gives 200 then 304 with the same bytes and etag; 404 gives 404 in both; an unknown host fails in both. C++ tests on `build-rust-static` (`BUILD_SHARED_LIBS=OFF`): `test-arg-parser`, `test-model-resolution`, `test-download-model` and `test-generate-models` pass. TLS: a `llama-server` built with `LLAMA_OPENSSL=ON` and `LLAMA_RUST_HTTP=ON` serves HTTPS (ring) and downloads its model from Hugging Face in the same process, both on rustls with `ring`, and `/health` answers 200 over TLS. Logs: the download path emits the C++ messages again, from Rust through `llama_dl_set_log_sink`; checked against a real run for a missing HF file, URL retries (500, then retrying after 2 and 4 seconds, then download failed after 3 attempts) and a Docker model.
- [x] 3.3 `imatrix-loader` + `quantize`. `rust/llama-quantize-core` holds the pure logic (arguments, ftype table, legacy imatrix reader, imatrix normalization and filters, KV overrides) and `rust/llama-quantize` the FFI side (GGUF imatrix through `gguf.h`, `llama_model_quantize`, C ABI in `llama_quantize.h`); no `llama-ext.h` was needed. `tools/quantize/quantize.cpp` (668 lines) is gone: `tools/quantize/main.cpp` and `app/llama.cpp` call `llama_rs_quantize` directly, and `common/imatrix-loader.cpp` (173 lines) is gone: `tools/imatrix/imatrix.cpp` reads the `llama_imatrix_view` from `llama_imatrix_load` directly. `common/build-info.h` is now `extern "C"` so Rust prints the same build lines. Checks: `scripts/rust-golden/quantize_diff.py` (71 cases recorded from the C++ tools: 63 `llama-quantize` runs and 8 `llama-imatrix --in-file` conversions, merges and load errors; ctest `test-quantize-golden`) gives 0 differences on the shared and the static build; full logs including libllama output and `llama-imatrix --in-file` statistics match the C++ binaries; `llama quantize` from the unified app works. The static build needed `llama` in the link interface of the imported `llama-rs` target, because `libllama_rs.a` now calls libllama.
- [-] 3.4 jinja (`common/jinja`, 6.3k): stays in C++ (user decision). A Rust port would be a rewrite with no gain: no crate keeps the C++ behavior (input-marked string parts against special-token injection, usage stats for `caps_get`, the llama.cpp quirks and error texts), so `minijinja` would change rendering for the templates in `models/templates`.
- [-] 3.5 `peg-parser.cpp`, `chat-peg-parser.cpp`, `build_grammar`: skipped, stays in C++ (user decision).
- [-] 3.6 chat cluster (`chat.cpp`, `chat-diff-analyzer.cpp`, `chat-auto-parser*`, `common/parsers/`): skipped, stays in C++ (user decision).
- [x] 3.7 samplers: `reasoning-budget` and `llguidance` in Rust; `sampling.cpp` and `common_params_sampling` stay in C++ (user decision).
  - [x] `reasoning-budget`: `rust/llama-sampling` (vtable, Aho-Corasick, UTF-8 check, trace sink); `common/reasoning-budget.cpp` (310 lines) is gone: `common/sampling.cpp` calls `llama_rs_reasoning_budget_*` directly, the state enum lives in `llama_sampling.h` and `common/log.cpp` installs the trace sink. Oracle: `test-reasoning-budget`, now on the C API through a flattening helper; server pytest non-slow 376 passed, 4 skipped.
  - [x] `llguidance`: `llama-sampling/src/llg.rs` behind the cargo feature `llama-rs/llguidance` (set by `LLAMA_LLGUIDANCE`), over the `llguidance` 1.0.1 crate; the `ExternalProject` clone and its separate static lib are gone, `common/llguidance.cpp` (260 to 22 lines) forwards. Oracle: `test-grammar-llguidance` passes.
  - [-] `sampling.cpp` (`common_sampler`): stays in C++ (user decision). It is the per-token path and mostly calls libllama, so a port brings no concrete gain.

The ~100 ggml/gguf calls in `common/` (`common.cpp`: 30, `arg.cpp`: 26, `imatrix-loader.cpp`: 22, `fit.cpp`: 20) go through `llama-sys`.

## Phase 4 - Scheduler and `main` in Rust

- [ ] `arg.cpp`, `preset.cpp` and `common_params` (moved from Phase 3), boundary in decision 4. Steps, each with its regression:
  - [x] 0 oracle: `scripts/rust-golden/params_driver.cpp` runs `common_params_parse` per example and dumps every field of `common_params` (dump generated from the clang AST by `scripts/gen-common-params.py` into `params_dump.inc`; `--check` fails when a struct changes) and the option table (`--options <example>`). `params_diff.py record` builds 1884 cases from the option tables (every option of every example: valid, invalid, env, negative and repeated forms; help, completion, config.ini and exit paths) and records exit code, stdout, stderr lines, new files and the changed fields; ctest `test-params-golden` replays them (0 differences on two reruns). Not covered: the log options only change the logger state.
  - [x] 1 `rust/llama-args` `Params` (hand-maintained) and the boundary: CBOR with field names instead of a flat `#[repr(C)]` mirror (336 fields with strings, vectors, sets and maps would need hand-written ownership code on both sides; CBOR also carries non-UTF-8 strings and `inf`/NaN floats, and a missing or extra field fails). `common/params-serde.cpp` holds a small CBOR codec (no nlohmann outside `common/json.cpp`) plus the generated `params-serde.inc`; `scripts/gen-common-params.py` generates it and the oracle dump from the clang AST. `params_driver` round-trips every golden case through Rust (`llama_args_params_roundtrip`): 1884 cases, 0 differences; dropping a Rust field makes the cases fail.
  - [ ] 2 option table, help, completion and gen-docs from Rust.
  - [ ] 3 parse engine in Rust (env, argv, negatives, errors) calling the C++ handlers by id.
  - [ ] 4 handlers to Rust by group (sampling, speculative, model, server, context and rope, devices and rpc, logs, tools).
  - [ ] 5 `preset.cpp` and `common_params_to_map` to Rust behind an opaque handle for `server-models.cpp`; `arg.cpp` becomes a shim; `test-arg-parser` table checks move to Rust tests.
- [ ] Port the logic of `server_context_impl` (`server-context.cpp:833-4151`) as a `std::thread` with task `mpsc` and one `mpsc` per request. `server_slot` (`server-context.cpp:239-735`) defaults to a C++ struct; ask when Phase 4 starts. Remove `server-queue.{h,cpp}` and the `next()` pull from `server-http.h`.
- [ ] `main` moves to Rust; C++ becomes a library (`llama-common` reduced to the shim). `app/llama.cpp:129` (unified binary) and `tools/server/main.cpp` are adapted.
- [ ] Routes (`server-context.cpp:4650-5288`) to axum. The `server-task.h` structs (`:50-644`) default to C++ (see Pending decisions).
- [ ] Async workers that delegate to the scheduler: handlers no longer block (`server_response_reader::next`, `tools/server/server-queue.h:203-244`) and each request becomes an async task awaiting its channel. The blocking thread per request that `serve_blocking` creates today (`rust/llama-http/src/service.rs`, `spawn_blocking`) and the cap `n_threads_http + 1024` (`tools/server/server-http.cpp`, `config.n_blocking`) disappear. Already done: the number of async workers is configurable with `--http-workers N|auto` (`common/arg.cpp`, `common/common.h`, `rust_http_workers()` in `tools/server/server-http.cpp`).
- [ ] Token coalescing of the stream, at the application level (study and measure before activating by default):
  - Insertion point: the async task that is currently `StreamBody::poll_frame` (`rust/llama-http/src/service.rs`), with a `select!` between the result channel and a timer.
  - Accumulates complete SSE events (never splits an event) up to a maximum of bytes or until the configurable interval passes; default 150 ms (user proposal, pending measurement). The first token always comes out immediately (TTFT).
  - Flush the buffer on completion, on errors, on pings (`--sse-ping-interval`) and when receiving `[DONE]`.
  - New CLI option for the interval (0 = disabled) and another for the maximum bytes. As the write pattern changes from C++, measure with `scripts/evals/phase2-affinity-sweep.sh` (`strace -c`, context switches, TTFT, jitter between tokens) and with `tools/server/tests/unit/test_stream.py` before setting the default value.

## Phase 5 - What remains

- [ ] `server-models.cpp` (router, 2612 lines, child process + proxy, ideal for tokio)
- [ ] `server-mcp.cpp`/`server-tools.cpp` (MCP, crate `rmcp`). Check the protocol types against the design rule before choosing `rmcp`.
- [ ] `common/speculative.cpp` (2995 lines, last due to its dependence on `llama-ext.h`)
- [ ] `common/fit.cpp`
- [ ] remaining tools (`llama-bench`, `perplexity`, `imatrix`, `cli`, `completion`)

## Critical files

Modify: `CMakeLists.txt:129-146` (`LLAMA_RUST` option), `tools/server/server-http.cpp:19-26`, `tools/server/CMakeLists.txt` (Rust variant of `llama-server-impl`), `tools/CMakeLists.txt` (Rust tools).
New: `rust/Cargo.toml`, `rust/llama-sys/`, `rust/llama/`, `rust/llama-http/`, `rust/tools/{tokenize,gguf-split}/`, `common/rust-shim/llama_ext_c.{h,cpp}`, `rust/README.md`.
Read-only (behavior reference): `tools/server/server-queue.h`, `tools/server/server-context.cpp`, `tools/server/server-task.h`, `common/*`.

## Verification

- [x] Base build: a default `cmake -B build` builds the Rust workspace, and `cargo build/test/clippy` run in `rust/`. `LLAMA_RUST=OFF` stops at configure since 3.2.
- Phase 2: `LLAMA_SERVER_BIN_PATH=build-rust/bin/llama-server tools/server/tests/tests.sh`. Register C++ binary failures as baseline first. Focus: `test_stream.py`, `test_security.py` (API key/CORS), `test_proxy.py`, `test_router.py`, `test_compat_gcp.py`, `test_sleep.py`, `test_basic.py`.
  - [x] non-`slow` tests: 376 passed, 4 skipped, identical to C++ baseline
  - [ ] `slow` tests (199, require downloading models)
    - Partial: Llama-3.2-1B Q4 slow tests (10) run on Rust and C++: same 4 pass, same 6 fail (the 1B model answers prose instead of code, so not transport-related). Larger models skipped: `llama-server -hf` downloads of 2-8 GB got the shell killed by the OS twice (low RAM/swap). Also Qwen3-0.6B C++ vs Rust, byte-identical: chat with tools (JSON and SSE stream), OpenAI-client stream with `stop`, `/completion`, cancel mid-stream releases the slot.
- [x] Disconnection: cut a client mid-stream and verify the slot is released (`should_stop`).
- Performance (criterion from `todos/qwen3-0.6b-cpu-performance.md`: regression <=1%):
  - [x] tg and TTFT end-to-end C++ vs Rust (`-t 4 -C 0x55 --cpu-strict 1`, 3 rounds): no regression, Rust uses 10 threads vs 15 and ~14% more voluntary context switches
  - [ ] repeat with more rounds and with `perf stat` (CPU migrations)
    - Done once on the shared dev box (6 rounds, too noisy to conclude on tg/pp128: tg C++ 42.0 vs Rust 39.0 with per-round spread 21-46; pp128 88 vs 93). Only clear signal: Rust has ~3.7x more CPU migrations (median 505 vs 138), likely unpinned tokio threads. Repeat on a quiet node (`perf.sh` at repo root toggles `perf_event_paranoid`).
  - [ ] pp128 with the Rust layer active
- Differential test C++ vs Rust on the same GGUFs and templates (`models/templates`, `tests/test-chat.cpp` as vector source).
  - [x] Phase 1 (tokenize and gguf-split, byte-for-byte output)
  - [ ] Phase 3 (`models/templates`, `tests/test-chat.cpp` as vector source)

## Next step

3.2 close-out: done. The server pytest on the Rust HTTP build passes the non-slow suite (376 passed, 4 skipped, 199 slow deselected). Boundary step closed: `common_params_model` and `common_download_callback` stay in C++ (decision 7). Timeouts closed by moving the client to hyper (decision 8). Rust HTTP and the Rust schema converter are now the only paths, and their C++ twins are deleted (the `build_grammar` converter stays in C++). 3.3 `imatrix-loader` + `quantize` is done. Phase 3 is closed: jinja, PEG and chat (3.4-3.6) stay in C++, 3.7 moved `reasoning-budget` and `llguidance` to `rust/llama-sampling`, and `sampling.cpp` stays in C++. Next: Phase 4, the `arg.cpp` port (decision 4): steps 0 (C++ oracle) and 1 (`rust/llama-args` `Params` and the CBOR boundary) are done; step 2 (option table, help, completion and gen-docs from Rust) is next. Decision 3 (`server-task.h` structs, `server_slot`) is still open.

## Pending decisions (asked when the phase reaches them)

1. **Jinja, PEG and chat (3.4-3.6) (decided).** All three stay in C++; Phase 3 shrinks to download, quantize and samplers.
2. **Existing Rust data types.** `rust/llama-schema/src/schema.rs` (JSON schema AST), `rust/llama-schema/src/trie.rs` and `rust/llama-http/src/config.rs`. Accepted for now; revisit only if they change.
3. **Phase 4 structs.** `server-task.h` structs and `server_slot`: default C++, ask when Phase 4 starts.
4. **arg.cpp and `common_params` (Phase 4) (decided).** The user chose Rust for both: argument parsing (`common/arg.cpp`, 4785 lines, 356 options) and the `common_params` type. Boundary (approved): an own table-driven parser in Rust that ports `common_params_parse_ex`, not clap (env fallback applied before argv, `args_neg`, per-example visibility, help with caller defaults, preset keys, `stoi` semantics and exact error texts would all fight clap). `rust/llama-args` owns `Params`; the C++ code keeps `common_params` (with `common_params_sampling`) as DTOs filled from Rust through CBOR (step 1), so `common_params_parse` keeps its signature and its 42 caller files do not change. Devices and buffer types cross as names and C++ resolves them; callbacks never cross. The parser starts from the caller's params, because 16 callers set defaults before parsing and the help shows them.
5. **3.2 download (revised).** The first design used `hf-hub` 1.0.0 for Hugging Face transport and cache. The user chose to remove it: `src/hf.rs`, `tests/hf.rs` and the dependency are gone, and the lock adds 102 package names over HEAD instead of 219. The bytes of every file, Hugging Face included, go through the URL path in `rust/llama-download/src/remote.rs` (progress, cancellation, etag, resume) over the hyper client in `rust/llama-download/src/client.rs`. The Hugging Face listing reproduces the C++ calls: `api/models/<repo>/refs`, `api/models/<repo>/tree/<commit>?recursive=true` and `resolve/<commit>/<path>`. Its coverage is `test-download-model` and `test-model-resolution` in ctest. Rust owns the selection logic (`find_best_*`, split GGUF, `get_all_parts`), the cache-dir resolution and the Docker registry. The C++ `download.h` and `hf-cache.h` are unchanged and `common/download.cpp` and `common/hf-cache.cpp` forward to Rust, so consumers (`arg.cpp`, `server-models.cpp`, tests) do not change. `hf_cache::hf_file` and `common_download_opts` stay in C++. `http.h` keeps `parse_url`, `format_host` and `get_free_port` for `server-http.cpp` and `server-models.cpp`. Known gaps against C++: `LOG_TRC` lines and the error lines printed when a cancelled download is torn down are not reproduced. Timeouts match C++ since decision 8.
6. **Still open:** `examples/` and `pocs/`, `tools/mtmd`, and `tools/rpc`.
7. **Boundary types (decided).** `common_params_model` and `common_download_callback` stay in C++ as data, per the design rule. Their consumers are the CLI and server: the flags in `common/arg.cpp`, `get_name()` in the server, `llama-bench`, the tests, and the progress callbacks in `common/download.cpp` and `tools/server/server-models.cpp`. Only 31 lines name the types, and about 128 touch their fields. Moving them would add FFI without moving any logic, because the logic already lives in `llama-download`. Rust receives the fields as C strings. Revisit with Phase 4 (`arg.cpp`).
8. **HTTP client and timeouts (decided).** reqwest 0.13 could not express the httplib timeouts: its `read_timeout` is one timer from `send()` to the headers, connect included. The user chose hyper directly. `rust/llama-download/src/client.rs` uses hyper 1 (http1 client connection, one connection per request like httplib), rustls with `ring` and `rustls-platform-verifier` (system trust store). Timeouts follow httplib: 300 s for DNS, TCP and the TLS handshake, then 5 s per socket read or write (`common_remote_get_content` passes its `timeout` as that per-read value, as `set_read_timeout` did). Redirects follow httplib: up to 20, only http and https, and a change of scheme, host or port drops `Authorization`, `Proxy-Authorization` and cookies. Proxies from environment variables are not used, as in httplib. reqwest and aws-lc-rs are gone from the lock (782 lines fewer). Tests in `tests/remote.rs`: a response that takes 6 s but sends data every 2 s succeeds (it failed with reqwest), a silent server fails after about 5 s, and a cross-origin redirect does not forward the bearer token.

## Risks and default decisions

- **rustls provider:** `llama-download` and `llama-http` both use `ring` through explicit `builder_with_provider` (`rust/llama-download/src/client.rs`, `rust/llama-http/src/tls.rs`), so only one provider is compiled. Verified by a run: a `llama-server` built with `LLAMA_OPENSSL=ON` and `LLAMA_RUST_HTTP=ON` serves HTTPS and downloads its model from Hugging Face in the same process.
- **Upstream divergence:** `tools/server` and `common/` are active areas in ggml-org. Keep C++ in parallel until parity and treat upstream as a source of changes to port.
- **`llama.h` and `llama-ext.h` change:** the bindings are committed (`rust/llama-sys/src/bindings.rs`) and `llama-sys/build.rs` fails when a header they came from changes (hash of the inputs recorded at generation); regenerate with `cargo build -p llama-sys --features bindgen`. Normal builds no longer compile bindgen or clang-sys (clean Rust build 37 s instead of about 43 s).
- **Thread cost in Phase 2:** each stream occupies a blocking thread until Phase 4. Accepted as an intermediate step.
- Defaults chosen: workspace in `rust/`, axum/hyper, rustls with `ring` always enabled for the server (`llama-http/tls`), independent of OpenSSL, `LLAMA_RUST` default ON since 3.2 (`common/CMakeLists.txt:153` stops at configure when it is OFF), CMake as the C++ driver and cargo for Rust.
- **Build speed:** CMake builds Rust with the cargo profile `cmake` (release plus incremental) in non-Debug configs; `-DLLAMA_RUST_CARGO_PROFILE=release` gives the fully optimized build for packaging. `CMAKE_LINK_DEPENDS_NO_SHARED` defaults to ON, so a change in `libllama-common` relinks only that library. Measured on 8 cores: touching a Rust file and rebuilding went from 11.0 s (96 relinks) to 2.7 s, touching `common/download.cpp` from 9.3 s to 4.9 s. `llama-download` dropped `regex` (split names are parsed by hand, `tests/gguf.rs` checks parity against the old regexes).
- Limits of analysis: did not measure coupling at the call-site level (gmem does not index references) and the outline of `llama.h` came out partial due to `LLAMA_API` macros.
