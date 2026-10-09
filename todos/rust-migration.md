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
- [x] `server-http.cpp:19-26` (`server_http_context::Impl`): Rust variant behind `LLAMA_RUST_HTTP`. New crate `rust/llama-http` (tokio + hyper/axum) compiled as `staticlib` with a C ABI: `init(config)`, `register(method, path, cb)`, `start`, `stop`, `join`. The `cb` is wrapped `handler_t`: receives method/path/params/headers/body/files and returns status/headers and body or chunks.
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

Order, each piece with differential test C++ vs Rust (jinja/chat are the highest fidelity risk):

- [x] 3.0 groundwork: umbrella crate `rust/llama-rs` (one staticlib, `llama_rs_version` round trip from `libllama-common`), `add_subdirectory(rust)` moved before `common`, single copy of Rust `std` checked with `nm`. Golden-vector script `scripts/rust-golden/` lands with 3.1.
- [x] `json-schema-to-grammar` (`common/json-schema-to-grammar.cpp`, 1028 lines): `rust/llama-schema` behind `LLAMA_RUST_SCHEMA`. The pure schema-to-grammar function is ported; `build_grammar(cb)`, the `common_chat_schema` AST and `trie` stay in C++ until 3.5/3.6. Verified with the three C++ tests with the flag ON, 81 golden cases, a 45k-schema differential against the C++ build and the server pytest (`test_chat_completion`, `test_compat_anthropic`, `test_basic`: 90 pass). Not built with `BUILD_SHARED_LIBS=OFF`.
- [ ] `download` (`common/download.cpp`, `reqwest`)
- [ ] `arg.cpp` (`clap`, 4770 lines; maintain exact flag compatibility)
- [ ] jinja (`common/jinja`, 6.3k)
- [ ] `peg-parser.cpp`, `chat-peg-parser.cpp`, `chat.cpp`
- [ ] `sampling.cpp` + the two custom samplers (`extern "C"` vtable)
- [ ] `imatrix-loader.cpp` and `quantize`

The ~100 ggml/gguf calls in `common/` (`common.cpp`: 30, `arg.cpp`: 26, `imatrix-loader.cpp`: 22, `fit.cpp`: 20) go through `llama-sys`.

## Phase 4 - Scheduler and `main` in Rust

- [ ] Port `server_context_impl`/`server_slot` (`server-context.cpp:239-4151`) as a `std::thread` with task `mpsc` and one `mpsc` per request. Remove `server-queue.{h,cpp}` and the `next()` pull from `server-http.h`.
- [ ] `main` moves to Rust; C++ becomes a library (`llama-common` reduced to the shim). `app/llama.cpp:129` (unified binary) and `tools/server/main.cpp` are adapted.
- [ ] Routes (`server-context.cpp:4650-5288`) to axum + serde; `server-task.h` (structs `:50-644`) to serde types.
- [ ] Async workers that delegate to the scheduler: handlers no longer block (`server_response_reader::next`, `tools/server/server-queue.h:203-244`) and each request becomes an async task awaiting its channel. The blocking thread per request that `serve_blocking` creates today (`rust/llama-http/src/service.rs`, `spawn_blocking`) and the cap `n_threads_http + 1024` (`tools/server/server-http-rust.inl`, `config.n_blocking`) disappear. Already done: the number of async workers is configurable with `--http-workers N|auto` (`common/arg.cpp`, `common/common.h`, `rust_http_workers()` in `server-http-rust.inl`).
- [ ] Token coalescing of the stream, at the application level (study and measure before activating by default):
  - Insertion point: the async task that is currently `StreamBody::poll_frame` (`rust/llama-http/src/service.rs`), with a `select!` between the result channel and a timer.
  - Accumulates complete SSE events (never splits an event) up to a maximum of bytes or until the configurable interval passes; default 150 ms (user proposal, pending measurement). The first token always comes out immediately (TTFT).
  - Flush the buffer on completion, on errors, on pings (`--sse-ping-interval`) and when receiving `[DONE]`.
  - New CLI option for the interval (0 = disabled) and another for the maximum bytes. As the write pattern changes from C++, measure with `scripts/evals/phase2-affinity-sweep.sh` (`strace -c`, context switches, TTFT, jitter between tokens) and with `tools/server/tests/unit/test_stream.py` before setting the default value.

## Phase 5 - What remains

- [ ] `server-models.cpp` (router, 2612 lines, child process + proxy, ideal for tokio)
- [ ] `server-mcp.cpp`/`server-tools.cpp` (MCP, crate `rmcp`)
- [ ] `common/speculative.cpp` (2995 lines, last due to its dependence on `llama-ext.h`)
- [ ] `common/fit.cpp`
- [ ] remaining tools (`llama-bench`, `perplexity`, `imatrix`, `cli`, `completion`)

## Critical files

Modify: `CMakeLists.txt:129-146` (`LLAMA_RUST` option), `tools/server/server-http.cpp:19-26`, `tools/server/CMakeLists.txt` (Rust variant of `llama-server-impl`), `tools/CMakeLists.txt` (Rust tools).
New: `rust/Cargo.toml`, `rust/llama-sys/`, `rust/llama/`, `rust/llama-http/`, `rust/tools/{tokenize,gguf-split}/`, `common/rust-shim/llama_ext_c.{h,cpp}`, `rust/README.md`.
Read-only (behavior reference): `tools/server/server-queue.h`, `tools/server/server-context.cpp`, `tools/server/server-task.h`, `common/*`.

## Verification

- [x] Base build: `cmake -B build-rust -DLLAMA_RUST=ON -DLLAMA_RUST_HTTP=ON` and `cargo build/test/clippy` in `rust/`. C++ build without Rust remains green (`LLAMA_RUST=OFF`).
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

## Risks and default decisions

- **Upstream divergence:** `tools/server` and `common/` are active areas in ggml-org. Keep C++ in parallel until parity and treat upstream as a source of changes to port.
- **`llama.h` and `llama-ext.h` change:** pin the revision and regenerate bindings in CI.
- **Thread cost in Phase 2:** each stream occupies a blocking thread until Phase 4. Accepted as an intermediate step.
- Defaults chosen: workspace in `rust/`, axum/hyper, rustls behind the same flag as OpenSSL, `LLAMA_RUST` default OFF, CMake as the C++ driver and cargo for Rust.
- Limits of analysis: did not measure coupling at the call-site level (gmem does not index references) and the outline of `llama.h` came out partial due to `LLAMA_API` macros.
