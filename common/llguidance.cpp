#include "sampling.h"
#include "log.h"

#ifdef LLAMA_USE_LLGUIDANCE

#    include "llama_sampling.h"

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab, const char * grammar_kind,
                                       const char * grammar_data) {
    return llama_rs_sampler_init_llg(vocab, grammar_kind, grammar_data);
}

#else

llama_sampler * llama_sampler_init_llg(const llama_vocab *, const char *, const char *) {
    LOG_WRN("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
    return nullptr;
}

#endif  // LLAMA_USE_LLGUIDANCE
