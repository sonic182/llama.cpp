#pragma once

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum llama_rs_reasoning_budget_state {
    LLAMA_RS_REASONING_BUDGET_IDLE,         // waiting for start sequence
    LLAMA_RS_REASONING_BUDGET_COUNTING,     // counting down tokens
    LLAMA_RS_REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    LLAMA_RS_REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    LLAMA_RS_REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be NULL)
//   start_*        - token sequences, any of which activates counting
//   end_*          - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//
struct llama_sampler * llama_rs_reasoning_budget_init(
        const struct llama_vocab * vocab,
        const llama_token * start_tokens, const size_t * start_lens, size_t n_start,
        const llama_token * end_tokens,   const size_t * end_lens,   size_t n_end,
        const llama_token * forced_tokens, size_t n_forced,
        int32_t budget,
        int32_t initial_state);

int32_t llama_rs_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or empty if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_token * llama_rs_reasoning_budget_get_end_match(const struct llama_sampler * smpl, size_t * n_tokens);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool llama_rs_reasoning_budget_force(struct llama_sampler * smpl);

typedef void (*llama_rs_sampling_log_sink)(int level, const char * func, const char * message);
void llama_rs_sampling_set_log_sink(llama_rs_sampling_log_sink sink);

struct llama_sampler * llama_rs_sampler_init_llg(const struct llama_vocab * vocab, const char * grammar_kind, const char * grammar_data);

#ifdef __cplusplus
}
#endif
