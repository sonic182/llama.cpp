#include "llama_ext_c.h"

#include "../../src/llama-ext.h"

int32_t llama_ext_c_model_n_expert(const struct llama_model * model) {
    return llama_model_n_expert(model);
}

int32_t llama_ext_c_model_n_devices(const struct llama_model * model) {
    return llama_model_n_devices(model);
}

ggml_backend_dev_t llama_ext_c_model_get_device(const struct llama_model * model, int i) {
    return llama_model_get_device(model, i);
}

uint32_t llama_ext_c_model_get_tok_embd(const struct llama_model * model, float * out) {
    return llama_model_get_tok_embd(model, out);
}

size_t llama_ext_c_memory_breakdown(const struct llama_context * ctx, struct llama_ext_c_memory_entry * out, size_t cap) {
    const llama_memory_breakdown mb = llama_get_memory_breakdown(ctx);
    size_t n = 0;
    for (const auto & [buft, data] : mb) {
        if (n < cap) {
            out[n] = { buft, data.model, data.context, data.compute };
        }
        n++;
    }
    return n;
}
