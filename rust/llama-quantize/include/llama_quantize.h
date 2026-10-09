#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int llama_rs_quantize(int argc, const char * const * argv);

typedef struct llama_imatrix llama_imatrix;

typedef struct llama_imatrix_entry {
    const char * name;
    const float * sums;
    size_t n_sums;
    const int64_t * counts;
    size_t n_counts;
} llama_imatrix_entry;

typedef struct llama_imatrix_view {
    const llama_imatrix_entry * entries;
    size_t n_entries;
    const char * const * datasets;
    size_t n_datasets;
    int32_t chunk_count;
    int32_t chunk_size;
    bool is_legacy;
    bool has_metadata;
} llama_imatrix_view;

llama_imatrix * llama_imatrix_load(const char * fname, void (*log_err)(const char * message));
llama_imatrix_view llama_imatrix_get(const llama_imatrix * imatrix);
void llama_imatrix_free(llama_imatrix * imatrix);

#ifdef __cplusplus
}
#endif
