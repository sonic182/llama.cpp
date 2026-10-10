#pragma once

#include "llama.h"

#if defined(_WIN32)
#    define LLAMA_EXT_C_API
#else
#    define LLAMA_EXT_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_ext_c_memory_entry {
    ggml_backend_buffer_type_t buft;
    size_t model;
    size_t context;
    size_t compute;
};

LLAMA_EXT_C_API int32_t llama_ext_c_model_n_expert (const struct llama_model * model);
LLAMA_EXT_C_API int32_t llama_ext_c_model_n_devices(const struct llama_model * model);

LLAMA_EXT_C_API ggml_backend_dev_t llama_ext_c_model_get_device(const struct llama_model * model, int i);

LLAMA_EXT_C_API uint32_t llama_ext_c_model_get_tok_embd(const struct llama_model * model, float * out);

LLAMA_EXT_C_API size_t llama_ext_c_memory_breakdown(const struct llama_context * ctx, struct llama_ext_c_memory_entry * out, size_t cap);

#ifdef __cplusplus
}
#endif
