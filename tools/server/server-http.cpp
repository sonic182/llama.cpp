#include "common.h"
#include "http.h"
#include "server-http.h"
#include "server-common.h"
#include "ui.h"
#include "llama_http.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// For Google Cloud Platform deployment compatibility
struct gcp_params {
    bool enabled;
    std::string path_health;
    std::string path_predict;
    int port;

    // Ref: https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
    gcp_params() {
        enabled = getenv("AIP_MODE", "") == "PREDICTION";
        path_health = getenv("AIP_HEALTH_ROUTE", "", true); // default: using the route defined in server.cpp
        path_predict = getenv("AIP_PREDICT_ROUTE", "/predict", true);
        port = std::stoi(getenv("AIP_HTTP_PORT", "8080"));
    }

    static std::string getenv(const char * name, const std::string & default_value, bool ensure_leading_slash = false) {
        const auto * value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') {
            return default_value;
        }
        std::string val = value;
        if (ensure_leading_slash && !val.empty() && val[0] != '/') {
            val.insert(val.begin(), '/');
        }
        return val;
    }
};

//
// HTTP implementation using the Rust llama-http crate (hyper + tokio)
//

static_assert(sizeof(std::atomic<bool>) == 1 && std::atomic<bool>::is_always_lock_free, "is_ready is shared with Rust as a byte");

static constexpr int rust_http_max_auto_workers = 4;
static constexpr int rust_http_max_dynamic_threads = 1024;

static int rust_http_workers(int requested) {
    if (requested > 0) {
        return requested;
    }
    const int cores = static_cast<int>(std::thread::hardware_concurrency());
    return std::clamp(cores / 4, 1, rust_http_max_auto_workers);
}

static llama_http_str rust_str(const std::string & s) {
    return { s.data(), s.size() };
}

static std::string rust_to_string(const llama_http_str & s) {
    return s.len == 0 ? std::string() : std::string(s.ptr, s.len);
}

class server_http_context::Impl {
public:
    llama_http_server * server = nullptr;
    std::vector<std::string> hosts;
    int n_threads_http = 0;
    std::vector<server_http_context::handler_t> routes;

    ~Impl() {
        if (server) {
            llama_http_server_free(server);
        }
    }

    void add_route(int method, const std::string & path, const server_http_context::handler_t & handler) {
        routes.push_back(handler);
        if (llama_http_server_route(server, method, rust_str(path), routes.size() - 1) != 0) {
            routes.pop_back();
        }
    }
};

struct rust_exchange {
    std::function<bool()> should_stop;
    std::unique_ptr<server_http_req> request;
    server_http_res_ptr response;
    std::vector<std::pair<std::string, std::string>> header_storage;
    std::vector<llama_http_kv> header_kvs;
    std::string chunk;
};

static void rust_set_error(rust_exchange & ex, const std::string & message) {
    ex.response = std::make_unique<server_http_res>();
    ex.response->status = 500;
    ex.response->content_type = "text/plain";
    ex.response->data = message;
}

static void rust_build_request(rust_exchange & ex, const llama_http_request & rq) {
    std::map<std::string, std::string> params;
    for (size_t i = 0; i < rq.n_params; i++) {
        params[rust_to_string(rq.params[i].key)] = rust_to_string(rq.params[i].value);
    }
    std::map<std::string, std::string> headers;
    for (size_t i = 0; i < rq.n_headers; i++) {
        headers[rust_to_string(rq.headers[i].key)] = rust_to_string(rq.headers[i].value);
    }

    std::string body = rust_to_string(rq.body);
    std::map<std::string, uploaded_file> files;
    if (rq.is_multipart) {
        json form_json = json::object();
        for (size_t i = 0; i < rq.n_fields; i++) {
            const std::string key     = rust_to_string(rq.fields[i].key);
            const std::string content = rust_to_string(rq.fields[i].value);
            if (form_json.contains(key)) {
                if (!form_json[key].is_array()) {
                    json existing_value = form_json[key];
                    form_json[key] = json::array({existing_value});
                }
                form_json[key].push_back(content);
            } else {
                form_json[key] = content;
            }
        }
        body = form_json.dump();

        for (size_t i = 0; i < rq.n_files; i++) {
            const llama_http_file & file = rq.files[i];
            files[rust_to_string(file.key)] = uploaded_file{
                raw_buffer(file.data.ptr, file.data.ptr + file.data.len),
                rust_to_string(file.filename),
                rust_to_string(file.content_type),
            };
        }
    }

    ex.request = std::make_unique<server_http_req>(server_http_req{
        std::move(params),
        std::move(headers),
        rust_to_string(rq.path),
        rust_to_string(rq.query_string),
        std::move(body),
        std::move(files),
        ex.should_stop,
    });
}

static int rust_dispatch(void * user, size_t route, const llama_http_request * rq, const uint8_t * cancelled, llama_http_response * out) {
    auto * impl = static_cast<server_http_context::Impl *>(user);
    auto ex = std::make_unique<rust_exchange>();
    ex->should_stop = [cancelled]() { return __atomic_load_n(cancelled, __ATOMIC_ACQUIRE) != 0; };

    try {
        rust_build_request(*ex, *rq);
        ex->response = impl->routes.at(route)(*ex->request);
        if (!ex->response) {
            rust_set_error(*ex, "empty response from handler");
        }
    } catch (const std::exception & e) {
        SRV_ERR("got exception: %s\n", e.what());
        rust_set_error(*ex, e.what());
    } catch (...) {
        SRV_ERR("%s", "got exception: Unknown Exception\n");
        rust_set_error(*ex, "Unknown Exception");
    }

    const server_http_res & res = *ex->response;
    for (const auto & [key, value] : res.headers) {
        ex->header_storage.emplace_back(key, value);
    }
    for (const auto & [key, value] : ex->header_storage) {
        ex->header_kvs.push_back({ rust_str(key), rust_str(value) });
    }

    out->status       = res.status;
    out->content_type = rust_str(res.content_type);
    out->headers      = ex->header_kvs.data();
    out->n_headers    = ex->header_kvs.size();
    out->data         = rust_str(res.data);
    out->is_stream    = res.is_stream() ? 1 : 0;
    out->handle       = ex.release();
    return 1;
}

static int rust_next(void * handle, llama_http_str * chunk) {
    auto * ex = static_cast<rust_exchange *>(handle);
    ex->chunk.clear();
    bool has_next = false;
    try {
        has_next = ex->response->next(ex->chunk);
    } catch (const std::exception & e) {
        SRV_ERR("got exception while streaming: %s\n", e.what());
    } catch (...) {
        SRV_ERR("%s", "got exception while streaming: Unknown Exception\n");
    }
    *chunk = rust_str(ex->chunk);
    return has_next ? 1 : 0;
}

static void rust_release(void * handle) {
    std::unique_ptr<rust_exchange> ex(static_cast<rust_exchange *>(handle));
    try {
        ex->response->on_complete();
    } catch (...) {
    }
}

static void rust_log(void *, int level, llama_http_str message) {
    std::string text = rust_to_string(message);
    if (!text.empty() && text.back() == '\n') {
        text.pop_back();
    }
    switch (level) {
        case LLAMA_HTTP_LOG_ERR: SRV_ERR("%s\n", text.c_str()); break;
        case LLAMA_HTTP_LOG_WRN: SRV_WRN("%s\n", text.c_str()); break;
        case LLAMA_HTTP_LOG_INF: SRV_INF("%s\n", text.c_str()); break;
        default:                 SRV_DBG("%s\n", text.c_str()); break;
    }
}

server_http_context::server_http_context()
    : pimpl(std::make_unique<Impl>())
{}

server_http_context::~server_http_context() {
    try {
        stop();
        join();
    } catch (const std::exception & e) {
        SRV_ERR("failed to stop HTTP server: %s\n", e.what());
    } catch (...) {
        SRV_ERR("%s", "failed to stop HTTP server\n");
    }
}

#if defined(LLAMA_UI_HAS_ASSETS)
static constexpr auto rust_cache_immutable  = "public, max-age=31536000, immutable";
static constexpr auto rust_cache_revalidate = "no-cache";

static server_http_res_ptr rust_serve_asset(const server_http_req & req, const std::string & name, bool isolation, const char * cache_control, bool etag) {
    auto res = std::make_unique<server_http_res>();
    const auto header = [&req](const std::string & key) {
        for (const auto & [k, v] : req.headers) {
            if (k.size() == key.size() && std::equal(k.begin(), k.end(), key.begin(), [](char a, char b) { return std::tolower(static_cast<unsigned char>(a)) == b; })) {
                return v;
            }
        }
        return std::string();
    };

    if (llama_ui_use_gzip()) {
        if (header("accept-encoding").find("gzip") == std::string::npos) {
            res->status       = 415;
            res->content_type = "text/plain";
            res->data         = "Error: gzip is not supported by this browser";
            return res;
        }
        res->headers["Content-Encoding"] = "gzip";
    }

    const llama_ui_asset * a = llama_ui_find_asset(name);
    if (!a) {
        res->status = 404;
        res->data   = safe_json_to_str(json {
            {"error", {
                {"message", "File Not Found"},
                {"type", "not_found_error"},
                {"code", 404}
            }}
        });
        return res;
    }

    if (etag) {
        res->headers["ETag"] = a->etag;
        if (const std::string inm = header("if-none-match");
            !inm.empty() && (inm == a->etag || inm == std::string("W/") + a->etag)) {
            res->status       = 304;
            res->content_type = "";
            return res;
        }
    }
    if (isolation) {
        res->headers["Cross-Origin-Embedder-Policy"] = "require-corp";
        res->headers["Cross-Origin-Opener-Policy"]   = "same-origin";
    }
    res->headers["Cache-Control"] = cache_control;
    res->content_type = a->type;
    res->data.assign(reinterpret_cast<const char *>(a->data), a->size);
    return res;
}

static void rust_register_ui(server_http_context::Impl & impl) {
    const auto index = [](const server_http_req & req) {
        return rust_serve_asset(req, "index.html", true, rust_cache_revalidate, true);
    };
    impl.add_route(LLAMA_HTTP_GET, "/",           index);
    impl.add_route(LLAMA_HTTP_GET, "/index.html", index);

    static const std::unordered_set<std::string> no_cache_names = {
        "sw.js",
        "manifest.webmanifest",
        "_app/version.json",
        "build.json"
    };

    for (const auto & a : llama_ui_get_assets()) {
        if (a.name == "index.html") continue;
        if (no_cache_names.count(a.name)) {
            impl.add_route(LLAMA_HTTP_GET, "/" + a.name, [name = a.name](const server_http_req & req) {
                return rust_serve_asset(req, name, false, rust_cache_revalidate, false);
            });
        } else {
            impl.add_route(LLAMA_HTTP_GET, "/" + a.name, [name = a.name](const server_http_req & req) {
                return rust_serve_asset(req, name, false, rust_cache_immutable, true);
            });
        }
    }
}
#endif

bool server_http_context::init(const common_params & params) {
    const gcp_params gcp;

    path_prefix = params.api_prefix;
    port = params.port;

    if (gcp.enabled) {
        SRV_TRC("Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n", gcp.path_health.c_str(), gcp.path_predict.c_str(), gcp.port);

        if (port != gcp.port) {
            SRV_WRN("Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", port, gcp.port);
        }

        port = gcp.port;
    }

    is_ssl = !params.ssl_file_key.empty() && !params.ssl_file_cert.empty();
    SRV_TRC("running %s SSL\n", is_ssl ? "with" : "without");

    pimpl->hosts = params.hostnames;
    size_t n_tcp_hosts = 0;
    for (const auto & host : pimpl->hosts) {
        if (!string_ends_with(host, ".sock")) {
            n_tcp_hosts++;
        }
    }
    if (port == 0 && n_tcp_hosts > 1) {
        SRV_ERR("%s", "--port 0 is not supported with multiple TCP addresses\n");
        return false;
    }

    pimpl->n_threads_http = params.n_threads_http;
    if (pimpl->n_threads_http < 1) {
        pimpl->n_threads_http = std::max(params.n_parallel + 4, static_cast<int32_t>(std::thread::hardware_concurrency() - 1));
    }
    SRV_TRC("using %d async workers and up to %d handler threads for HTTP server\n", rust_http_workers(params.http_workers), pimpl->n_threads_http);

    if (params.api_keys.size() == 1) {
        const auto key = params.api_keys[0];
        const std::string substr = key.substr(std::max(static_cast<int>(key.length() - 4), 0));
        SRV_TRC("api_keys: ****%s\n", substr.c_str());
    } else if (params.api_keys.size() > 1) {
        SRV_TRC("api_keys: %zu keys loaded\n", params.api_keys.size());
    }

    std::vector<std::string> frontend_paths = { "/" };
    for (const llama_ui_asset & a : llama_ui_get_assets()) {
        frontend_paths.push_back("/" + a.name);
    }
    std::vector<std::string> public_paths = { "/health", "/v1/health" };
    public_paths.insert(public_paths.end(), frontend_paths.begin(), frontend_paths.end());

    const auto to_strs = [](const std::vector<std::string> & in) {
        std::vector<llama_http_str> out;
        out.reserve(in.size());
        for (const auto & s : in) {
            out.push_back(rust_str(s));
        }
        return out;
    };
    const auto hosts           = to_strs(pimpl->hosts);
    const auto api_keys        = to_strs(params.api_keys);
    const auto public_strs     = to_strs(public_paths);
    const auto frontend_strs   = to_strs(frontend_paths);

    llama_http_config config = {};
    config.hosts             = hosts.data();
    config.n_hosts           = hosts.size();
    config.port              = port;
    config.api_prefix        = rust_str(params.api_prefix);
    config.api_keys          = api_keys.data();
    config.n_api_keys        = api_keys.size();
    config.public_paths      = public_strs.data();
    config.n_public_paths    = public_strs.size();
    config.frontend_paths    = frontend_strs.data();
    config.n_frontend_paths  = frontend_strs.size();
    config.cors_origins      = rust_str(params.cors_origins);
    config.cors_methods      = rust_str(params.cors_methods);
    config.cors_headers      = rust_str(params.cors_headers);
    config.cors_credentials  = params.cors_credentials ? 1 : 0;
    config.timeout_read_sec  = params.timeout_read;
    config.timeout_write_sec = params.timeout_write;
    config.reuse_port        = params.reuse_port ? 1 : 0;
    config.n_workers         = rust_http_workers(params.http_workers);
    config.n_blocking        = pimpl->n_threads_http + rust_http_max_dynamic_threads;
    config.ready             = reinterpret_cast<const uint8_t *>(&is_ready);
    config.ssl_cert_file     = rust_str(params.ssl_file_cert);
    config.ssl_key_file      = rust_str(params.ssl_file_key);
    if (params.ui) {
        config.static_dir = rust_str(params.public_path);
    }

    llama_http_callbacks callbacks = {};
    callbacks.user     = pimpl.get();
    callbacks.dispatch = rust_dispatch;
    callbacks.next     = rust_next;
    callbacks.release  = rust_release;
    callbacks.log      = rust_log;

    pimpl->server = llama_http_server_new(&config, &callbacks);
    if (!pimpl->server) {
        return false;
    }

    if (!params.ui) {
        SRV_INF("%s", "The UI is disabled\n");
        SRV_INF("%s", "Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable\n");
    } else {
#if defined(LLAMA_UI_HAS_ASSETS)
        rust_register_ui(*pimpl);
#endif
    }
    return true;
}

bool server_http_context::start() {
    listening_addresses.clear();
    if (!pimpl->server) {
        return false;
    }

    int bound_port = 0;
    if (llama_http_server_start(pimpl->server, &bound_port) != 0) {
        SRV_ERR("couldn't bind HTTP server socket, port: %d\n", port);
        return false;
    }
    if (port == 0) {
        port = bound_port;
    }

    for (const auto & host : pimpl->hosts) {
        const bool is_sock = string_ends_with(host, ".sock");
        listening_addresses.push_back(is_sock ? string_format("unix://%s", host.c_str())
                                              : string_format("%s://%s:%d", is_ssl ? "https" : "http", common_http_format_host(host).c_str(), port));
    }
    return true;
}

void server_http_context::stop() const {
    if (pimpl->server) {
        llama_http_server_stop(pimpl->server);
    }
}

void server_http_context::join() {
    if (pimpl->server) {
        llama_http_server_join(pimpl->server);
    }
}

void server_http_context::get(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->add_route(LLAMA_HTTP_GET, path, handler);
}

void server_http_context::post(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->add_route(LLAMA_HTTP_POST, path, handler);
}

void server_http_context::del(const std::string & path, const server_http_context::handler_t & handler) const {
    handlers.emplace(path, handler);
    pimpl->add_route(LLAMA_HTTP_DELETE, path, handler);
}


//
// Vertex AI Prediction protocol (AIP_PREDICT_ROUTE)
// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
//

// Derives the camelCase @requestFormat alias for a registered path.
// e.g. "/v1/chat/completions" -> "chatCompletions", "/apply-template" -> "applyTemplate"
static std::string path_to_gcp_format(const std::string & path) {
    std::string s = path;
    if (s.size() > 3 && s[0] == '/' && s[1] == 'v' && s[2] == '1') {
        s = s.substr(3);
    }
    if (!s.empty() && s[0] == '/') {
        s = s.substr(1);
    }
    std::string result;
    bool cap = false;
    for (unsigned char c : s) {
        if (c == ':') break; // stop before path parameters
        if (c == '/' || c == '-' || c == '_') {
            cap = true;
        } else {
            result += static_cast<char>(cap ? std::toupper(c) : c);
            cap = false;
        }
    }
    return result;
}

static json parse_gcp_predict_response(const server_http_res_ptr & res) {
    if (res == nullptr) {
        throw std::runtime_error("empty response from internal handler");
    }
    if (res->is_stream()) {
        throw std::invalid_argument("predict route does not support streaming responses");
    }
    if (res->data.empty()) {
        return nullptr;
    }
    try {
        return json::parse(res->data);
    } catch (...) {
        return res->data;
    }
}

void server_http_context::register_gcp_compat() const {
    const gcp_params gcp;

    if (!gcp.enabled) {
        // do nothing
        return;
    }

    if (handlers.count(gcp.path_predict)) {
        SRV_ERR("AIP_PREDICT_ROUTE=%s conflicts with an existing llama-server route\n", gcp.path_predict.c_str());
        exit(1);
    }

    // camelCase alias -> canonical path (first registration wins on collision)
    // e.g. "chatCompletions" -> "/v1/chat/completions"
    std::unordered_map<std::string, std::string> alias_to_path;
    for (const auto & [path, _] : handlers) {
        alias_to_path.emplace(path_to_gcp_format(path), path);
    }

    if (!gcp.path_health.empty()) {
        const auto health_handler = handlers.find("/health");
        GGML_ASSERT(health_handler != handlers.end());
        get(gcp.path_health, health_handler->second);
    }

    post(gcp.path_predict, [this, alias_to_path = std::move(alias_to_path)](const server_http_req & req) -> server_http_res_ptr {
        static const auto build_error = [](const std::string & message, error_type type) -> json {
            return json {{"error", format_error_response(message, type)}};
        };

        json data;
        try {
            data = json::parse(req.body);
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.is_object()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must be a JSON object", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }
        if (!data.contains("instances") || !data.at("instances").is_array()) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("request body must include an array field named instances", ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        const json & instances = data.at("instances");
        static const size_t MAX_INSTANCES = 128;
        if (instances.size() > MAX_INSTANCES) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->data = safe_json_to_str({{"error", format_error_response("instances array exceeds maximum size of " + std::to_string(MAX_INSTANCES), ERROR_TYPE_INVALID_REQUEST)}});
            return res;
        }

        std::vector<std::future<json>> futures;
        futures.reserve(instances.size());

        for (const auto & instance : instances) {
            futures.push_back(std::async(std::launch::async, [this, &req, &alias_to_path, instance]() -> json {
                if (!instance.is_object()) {
                    return build_error("each instance must be a JSON object", ERROR_TYPE_INVALID_REQUEST);
                }
                if (!instance.contains("@requestFormat") || !instance.at("@requestFormat").is_string()) {
                    return build_error("each instance must include a string @requestFormat", ERROR_TYPE_INVALID_REQUEST);
                }

                try {
                    json payload = instance;
                    const std::string format = payload.at("@requestFormat").get<std::string>();
                    payload.erase("@requestFormat");

                    if (payload.contains("stream")) {
                        SRV_WRN("%s", "ignoring client-provided stream field in instance, streaming is not supported in predict route\n");
                        payload["stream"] = false;
                    }

                    // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
                    std::string dispatch_path;
                    auto it_alias = alias_to_path.find(format);
                    if (it_alias != alias_to_path.end()) {
                        dispatch_path = it_alias->second;
                    } else if (handlers.count(format)) {
                        dispatch_path = format;
                    } else {
                        return build_error("no handler registered for @requestFormat: " + format, ERROR_TYPE_INVALID_REQUEST);
                    }

                    const server_http_req internal_req {
                        req.params,
                        req.headers,
                        path_prefix + dispatch_path,
                        req.query_string,
                        payload.dump(),
                        {},
                        req.should_stop,
                    };

                    server_http_res_ptr internal_res = handlers.at(dispatch_path)(internal_req);
                    return parse_gcp_predict_response(internal_res);
                } catch (const std::invalid_argument & e) {
                    return build_error(e.what(), ERROR_TYPE_INVALID_REQUEST);
                } catch (const std::exception & e) {
                    return build_error(e.what(), ERROR_TYPE_SERVER);
                } catch (...) {
                    return build_error("unknown error", ERROR_TYPE_SERVER);
                }
            }));
        }

        json predictions = json::array();
        for (auto & future : futures) {
            predictions.push_back(future.get());
        }

        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"predictions", predictions}});
        return res;
    });
}
