#include "reasoning-budget.h"
#include "log.h"

#include "llama_sampling.h"

#include <vector>

static void common_sampling_log(int level, const char * func, const char * message) {
    if (level == 0) {
        LOG_TRC("cmn  %12.*s: %s", 12, func, message);
    } else {
        LOG_ERR("%s", message);
    }
}

static const struct common_sampling_log_installer {
    common_sampling_log_installer() { llama_rs_sampling_set_log_sink(common_sampling_log); }
} common_sampling_log_installer_instance;

struct flat_seqs {
    llama_tokens        tokens;
    std::vector<size_t> lens;

    flat_seqs(const std::vector<llama_tokens> & seqs) {
        for (const auto & seq : seqs) {
            tokens.insert(tokens.end(), seq.begin(), seq.end());
            lens.push_back(seq.size());
        }
    }
};

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    const flat_seqs start(start_seqs);
    const flat_seqs end(end_seqs);
    return llama_rs_reasoning_budget_init(vocab,
        start.tokens.data(), start.lens.data(), start.lens.size(),
        end.tokens.data(),   end.lens.data(),   end.lens.size(),
        forced_tokens.data(), forced_tokens.size(),
        budget, initial_state);
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    return (common_reasoning_budget_state) llama_rs_reasoning_budget_get_state(smpl);
}

llama_tokens common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    size_t n = 0;
    const llama_token * tokens = llama_rs_reasoning_budget_get_end_match(smpl, &n);
    return tokens ? llama_tokens(tokens, tokens + n) : llama_tokens();
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    return llama_rs_reasoning_budget_force(smpl);
}
