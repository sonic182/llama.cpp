#pragma once

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_sampler * llama_rs_reasoning_budget_init(
        const struct llama_vocab * vocab,
        const llama_token * start_tokens, const size_t * start_lens, size_t n_start,
        const llama_token * end_tokens,   const size_t * end_lens,   size_t n_end,
        const llama_token * forced_tokens, size_t n_forced,
        int32_t budget,
        int32_t initial_state);

int32_t llama_rs_reasoning_budget_get_state(const struct llama_sampler * smpl);

const llama_token * llama_rs_reasoning_budget_get_end_match(const struct llama_sampler * smpl, size_t * n_tokens);

bool llama_rs_reasoning_budget_force(struct llama_sampler * smpl);

typedef void (*llama_rs_sampling_log_sink)(int level, const char * func, const char * message);
void llama_rs_sampling_set_log_sink(llama_rs_sampling_log_sink sink);

struct llama_sampler * llama_rs_sampler_init_llg(const struct llama_vocab * vocab, const char * grammar_kind, const char * grammar_data);

#ifdef __cplusplus
}
#endif
