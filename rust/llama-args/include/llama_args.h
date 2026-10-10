#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool llama_args_params_roundtrip(const uint8_t * data, size_t len, uint8_t ** out, size_t * out_len);

void llama_args_free_buffer(uint8_t * data, size_t len);

#ifdef __cplusplus
}
#endif
