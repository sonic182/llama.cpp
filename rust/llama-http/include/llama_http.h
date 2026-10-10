#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_http_server llama_http_server;

typedef struct {
    const char * ptr;
    size_t       len;
} llama_http_str;

typedef struct {
    llama_http_str key;
    llama_http_str value;
} llama_http_kv;

typedef struct {
    llama_http_str key;
    llama_http_str filename;
    llama_http_str content_type;
    llama_http_str data;
} llama_http_file;

typedef struct {
    const llama_http_str * hosts;
    size_t                 n_hosts;
    int                    port;
    llama_http_str         api_prefix;
    const llama_http_str * api_keys;
    size_t                 n_api_keys;
    const llama_http_str * public_paths;
    size_t                 n_public_paths;
    const llama_http_str * frontend_paths;
    size_t                 n_frontend_paths;
    llama_http_str         cors_origins;
    llama_http_str         cors_methods;
    llama_http_str         cors_headers;
    int                    cors_credentials;
    int                    timeout_read_sec;
    int                    timeout_write_sec;
    int                    reuse_port;
    int                    n_workers;
    int                    n_blocking;
    const uint8_t *        ready;
    llama_http_str         static_dir;
    llama_http_str         ssl_cert_file;
    llama_http_str         ssl_key_file;
} llama_http_config;

typedef struct {
    llama_http_str         path;
    llama_http_str         query_string;
    llama_http_str         body;
    const llama_http_kv *  params;
    size_t                 n_params;
    const llama_http_kv *  headers;
    size_t                 n_headers;
    int                    is_multipart;
    const llama_http_kv *  fields;
    size_t                 n_fields;
    const llama_http_file * files;
    size_t                 n_files;
} llama_http_request;

typedef struct {
    int                   status;
    llama_http_str        content_type;
    const llama_http_kv * headers;
    size_t                n_headers;
    llama_http_str        data;
    int                   is_stream;
    void *                handle;
} llama_http_response;

typedef struct {
    void * user;
    int    (*dispatch)(void * user, size_t route, const llama_http_request * request, const uint8_t * cancelled, llama_http_response * response);
    int    (*next)(void * handle, llama_http_str * chunk);
    void   (*release)(void * handle);
    void   (*log)(void * user, int level, llama_http_str message);
} llama_http_callbacks;

enum {
    LLAMA_HTTP_GET    = 0,
    LLAMA_HTTP_POST   = 1,
    LLAMA_HTTP_DELETE = 2,
};

enum {
    LLAMA_HTTP_LOG_ERR = 0,
    LLAMA_HTTP_LOG_WRN = 1,
    LLAMA_HTTP_LOG_INF = 2,
    LLAMA_HTTP_LOG_DBG = 3,
};

llama_http_server * llama_http_server_new(const llama_http_config * config, const llama_http_callbacks * callbacks);
int                 llama_http_server_route(const llama_http_server * server, int method, llama_http_str path, size_t id);
int                 llama_http_server_start(const llama_http_server * server, int * port);
void                llama_http_server_stop(const llama_http_server * server);
void                llama_http_server_join(const llama_http_server * server);
void                llama_http_server_free(llama_http_server * server);

#ifdef __cplusplus
}
#endif
