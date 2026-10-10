# Rust migration: code organization

How to lay out code that moves from C/C++ to `rust/`. The style is hexagonal at the
I/O boundaries, without full DDD: no aggregates, repositories, or domain events. Most
"domain" logic here is selection rules and orchestration ported for byte-level parity
with C++. Keep layers only where they buy testability or keep C++ out of Rust logic.

## Layers inside a crate

| Layer | Contains | Must not contain |
| --- | --- | --- |
| Pure logic | parsing, selection rules, plans, state machines, formatting | filesystem, network, env vars, threads, clocks, stdout, globals |
| Orchestration | runs a plan: calls adapters in order, collects results | decisions that a pure function could make |
| Adapters | HTTP client, disk cache, GGUF via `llama-sys`, terminal output | business rules |
| `ffi.rs` | `extern "C"` functions, `#[repr(C)]` types, C string and buffer conversion, `catch_unwind` | anything else |

Reference split already in the tree: `llama-quantize-core` (pure: args, ftype table,
imatrix rules) vs `llama-quantize` (GGUF, libllama, C ABI). Pure helpers in
`llama-download`: `select.rs`, `gguf.rs`, `repo.rs`, replayed by `tests/golden.rs`.

## Rules

1. **Planner / executor split.** When logic decides *what* to do and then does I/O, write
   a pure function that returns the decision (a list of tasks, the params to change),
   plus an executor that performs it. Unit-test the planner with plain data. Example:
   `llama-args/src/models.rs`, where `plan_tasks` works on a small `Sources` struct
   (not the whole `Params`) with the cache path and the GGUF hook injected, `finish`
   takes `finalize` as a function, and `apply` only orders Docker, plan, downloads
   and finish.
2. **Ports are traits only at a real seam.** Add a trait (for example a Hugging Face
   catalog or a model cache) when a test would otherwise need a loopback server or a
   real cache, or when a second implementation exists. Otherwise pass a function or keep
   a plain struct. Existing ports: `remote::Callback` (progress), the log sinks
   (`llama_dl_set_log_sink`, `llama_rs_sampling_set_log_sink`), the
   `spec_types_from_gguf` hook of the models handler.
3. **Inject configuration.** Environment and home directory are read by one function
   that takes them as arguments (`cache::cache_dir_from(var, home)`), wrapped by a thin
   default (`cache_dir()`). Pure code receives paths, not env lookups.
4. **C++ stays behind `ffi.rs`.** No other module sees raw pointers, `CStr`, or C
   callbacks. Convert C callbacks to a Rust trait object at the boundary
   (`ffi::Adapter`). A C++ function Rust needs (for example
   `common_speculative_types_from_gguf`) comes in as a hook; do not port it just to call it.
5. **Typed values inside, integers only on the wire.** `Params` mirrors `common_params`
   for CBOR (`ByteBuf`, `i32` enums). New logic converts to typed enums
   (`SpecType`, `Example`) at the edge instead of spreading constants like
   `SPEC_DRAFT_MTP = 3`. Booleans the C++ side can compute from its own enums (for
   example `use_mmproj`) may cross as booleans.
6. **One FFI error convention.** New C functions return success plus an out buffer
   with the message and an `int32_t` kind (`LLAMA_ARGS_ERROR_INVALID_ARGUMENT`,
   `LLAMA_ARGS_ERROR_RUNTIME`), so C++ rethrows `std::invalid_argument` or
   `std::runtime_error`. Scalars and string lists are typed (`llama_dl_resolve_path`,
   `llama_dl_list_cached_models`); JSON envelopes remain only for structured results
   that existing callers already parse (`llama_dl_hf_plan`). Never add a new JSON
   envelope for a scalar.
7. **No new global state.** Existing globals (log `OnceLock` sinks, the stdout progress
   `Board`, `SYMLINKS_DISABLED`) are boundary concerns. New state goes in a value the
   caller owns. When output must be shared (terminal lines), make the shared part a
   type with an injectable writer (`progress::Board<W>`) and keep one static instance
   at the edge.
8. **Parity text is a contract, keep it at the edge.** Log and error strings that
   reproduce C++ output (`common_download_file_single_online: ...`) must stay
   byte-identical while goldens compare them. Keep them in the module that emits them;
   do not let pure logic depend on them.
9. **Crates follow bounded contexts:** model acquisition (`llama-download`), config and
   CLI (`llama-args`), quantization, sampling, HTTP transport (`llama-http`), schema to
   grammar (`llama-schema`). A crate depends on another only for its public API, never
   its `ffi` module, except to reuse a boundary type such as `LlamaDlCallback`.
10. **No forwarders.** C++ callers call the Rust C API directly. A C++ wrapper is allowed
    only as the glue of a consumer that is itself scheduled to move (for example the
    CBOR glue of `common_models_handler_*` in `common/arg.cpp`).

## Checklist for a new or ported module

- Which part is pure? Can it be tested with plain values, without network or disk?
- Does any function both decide and do I/O? Split it.
- Is every env, clock, or global access in an adapter or a `*_from` function?
- Does the C ABI follow rule 6 and stay inside `ffi.rs`?
- Are enums typed past the boundary?
- Does a trait have a seam that justifies it (rule 2)?
- Is the C++ oracle still the regression test (goldens, ctest, server pytest)?

## Do not

- Introduce aggregates, repositories, domain events, or service layers for their own sake.
- Restructure working code only to match these rules; apply them to code you are
  already changing, or as a separate refactor commit with its own tests.
- Split `common_params` into contexts before `arg.cpp` is fully in Rust (Phase 4 steps
  3 to 5); until then it is the boundary DTO.
