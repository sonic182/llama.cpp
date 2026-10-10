#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool llama_args_params_roundtrip(const uint8_t * data, size_t len, uint8_t ** out, size_t * out_len);

void llama_args_free_buffer(uint8_t * data, size_t len);

struct llama_dl_callback;

typedef struct llama_args_models_handler llama_args_models_handler;

enum llama_args_error_kind {
    LLAMA_ARGS_ERROR_INVALID_ARGUMENT,
    LLAMA_ARGS_ERROR_RUNTIME,
};

typedef size_t (*llama_args_spec_types_from_gguf)(const char * path, int32_t * types, size_t cap);

llama_args_models_handler * llama_args_models_handler_init(const uint8_t * params, size_t len, bool use_mmproj,
                                                           uint8_t ** error, size_t * error_len, int32_t * error_kind);

bool llama_args_models_handler_is_preset_repo(const llama_args_models_handler * handler);

bool llama_args_models_handler_apply(llama_args_models_handler * handler, const uint8_t * params, size_t len,
                                     const struct llama_dl_callback * callback,
                                     llama_args_spec_types_from_gguf spec_types_from_gguf,
                                     uint8_t ** out, size_t * out_len, int32_t * error_kind);

void llama_args_models_handler_free(llama_args_models_handler * handler);

#ifdef __cplusplus
}
#endif
