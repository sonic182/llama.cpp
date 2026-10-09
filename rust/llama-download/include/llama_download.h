#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Structured results are JSON envelopes: {"ok": true, "value": ...} or
// {"ok": false, "kind": "invalid_argument" | "runtime", "error": "..."}.
// Free returned strings with llama_dl_free_string.

typedef struct llama_dl_progress {
    const char * url;
    uint64_t downloaded;
    uint64_t total;
    bool cached;
} llama_dl_progress;

typedef struct llama_dl_callback {
    void * ctx;
    void (*on_start)(void * ctx, const llama_dl_progress * p);
    void (*on_update)(void * ctx, const llama_dl_progress * p);
    void (*on_done)(void * ctx, const llama_dl_progress * p, bool ok);
    bool (*is_cancelled)(void * ctx);
} llama_dl_callback;

typedef void (*llama_dl_log_sink)(int level, const char * message);
void llama_dl_set_log_sink(llama_dl_log_sink sink);
void llama_dl_free_string(char * text);
void llama_dl_free_buffer(uint8_t * ptr, size_t len);

char * llama_dl_remote_get(const char * url, const char * const * header_names, const char * const * header_values,
                           size_t n_headers, uint64_t timeout_seconds, size_t max_size,
                           int64_t * out_status, uint8_t ** out_body, size_t * out_len);

int32_t llama_dl_file_single(const char * url, const char * path, const char * const * header_names,
                             const char * const * header_values, size_t n_headers, const char * bearer_token,
                             bool offline, bool skip_etag, const llama_dl_callback * callback);

char * llama_dl_split_repo_tag(const char * spec);
char * llama_dl_all_parts(const char * url);
char * llama_dl_list_cached_models(void);
char * llama_dl_resolve_path(const char * spec, const char * file);
char * llama_dl_remove(const char * spec);
char * llama_dl_docker_resolve(const char * docker);
char * llama_dl_hf_repo_files(const char * repo, const char * token);
char * llama_dl_hf_cached_files(const char * repo);
char * llama_dl_hf_plan(const char * spec, const char * hf_file, const char * token, uint32_t flags);
char * llama_dl_hf_finalize(const char * local_path, const char * final_path);
char * llama_dl_hf_remove_repo(const char * repo);
char * llama_dl_hf_cache_path(void);

#ifdef __cplusplus
}
#endif
