#include "arg.h"
#include "common.h"
#include "log.h"
#include "params-serde.h"

#include "llama_args.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

using json = nlohmann::ordered_json;

static const char * const EXAMPLES[] = {
    "batched", "debug", "common", "speculative", "completion", "cli", "embedding", "perplexity",
    "retrieval", "passkey", "imatrix", "bench", "server", "cvector-generator", "export-lora", "mtmd",
    "lookup", "parallel", "tts", "diffusion", "finetune", "fit-params", "results", "export-graph-ops",
    "download", "tokenize",
};
static_assert(sizeof(EXAMPLES) / sizeof(EXAMPLES[0]) == LLAMA_EXAMPLE_COUNT, "example names");

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static json dump(T v) {
    return v;
}

template <class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
static json dump(T v) {
    return static_cast<int64_t>(v);
}

static json dump(const std::string & v) {
    return v;
}

static json dump(const char * v) {
    return v ? json(v) : json(nullptr);
}

static json dump(ggml_backend_dev_t v) {
    return v ? json(ggml_backend_dev_name(v)) : json(nullptr);
}

static json dump(ggml_backend_buffer_type_t v) {
    return v ? json(ggml_backend_buft_name(v)) : json(nullptr);
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
static json dump(T * v) {
    return v != nullptr;
}

static json dump(const llama_model_kv_override & v) {
    json j = {{"key", v.key}, {"tag", static_cast<int64_t>(v.tag)}};
    switch (v.tag) {
        case LLAMA_KV_OVERRIDE_TYPE_INT:   j["value"] = v.val_i64;  break;
        case LLAMA_KV_OVERRIDE_TYPE_FLOAT: j["value"] = v.val_f64;  break;
        case LLAMA_KV_OVERRIDE_TYPE_BOOL:  j["value"] = v.val_bool; break;
        case LLAMA_KV_OVERRIDE_TYPE_STR:   j["value"] = v.val_str;  break;
    }
    return j;
}

static json dump(const llama_model_tensor_buft_override & v) {
    return {{"pattern", dump(v.pattern)}, {"buft", dump(v.buft)}};
}

static json dump(const std::vector<llama_model_tensor_buft_override> & v) {
    json items = json::array();
    for (const auto & e : v) {
        if (e.pattern || e.buft) {
            items.push_back(dump(e));
        }
    }
    return {{"items", items}, {"size", v.size()}};
}

static json dump(const llama_logit_bias & v) {
    return {{"token", v.token}, {"bias", v.bias}};
}

template <class T, size_t N>
static json dump(const T (&v)[N]) {
    json j = json::array();
    for (const auto & e : v) {
        j.push_back(dump(e));
    }
    return j;
}

template <class T>
static json dump(const std::vector<T> & v) {
    json j = json::array();
    for (const auto & e : v) {
        j.push_back(dump(e));
    }
    return j;
}

template <class T>
static json dump(const std::set<T> & v) {
    json j = json::array();
    for (const auto & e : v) {
        j.push_back(dump(e));
    }
    return j;
}

static json dump(const std::map<std::string, std::string> & v) {
    json j = json::object();
    for (const auto & [k, e] : v) {
        j[k] = e;
    }
    return j;
}

#include "params_dump.inc"

static int example_from_name(const char * name) {
    for (int i = 0; i < LLAMA_EXAMPLE_COUNT; i++) {
        if (strcmp(EXAMPLES[i], name) == 0) {
            return i;
        }
    }
    fprintf(stderr, "params_driver: unknown example %s\n", name);
    exit(2);
}

static json examples(const std::set<llama_example> & set) {
    json j = json::array();
    for (auto ex : set) {
        j.push_back(EXAMPLES[ex]);
    }
    return j;
}

static json options(llama_example ex) {
    common_params params{};
    auto ctx = common_params_parser_init(params, ex);
    common_params_add_preset_options(ctx.options);
    json j = json::array();
    for (auto & opt : ctx.options) {
        const char * kind = opt.handler_void ? "void" : opt.handler_bool ? "bool" : opt.handler_int ? "int"
                          : opt.handler_string ? "string" : opt.handler_str_str ? "str_str" : "none";
        j.push_back({
            {"args", opt.args},
            {"args_neg", opt.args_neg},
            {"value_hint", dump(opt.value_hint)},
            {"value_hint_2", dump(opt.value_hint_2)},
            {"env", dump(opt.env)},
            {"help", opt.help},
            {"examples", examples(opt.examples)},
            {"excludes", examples(opt.excludes)},
            {"in_example", opt.in_example(ex)},
            {"is_sampling", opt.is_sampling},
            {"is_spec", opt.is_spec},
            {"is_preset_only", opt.is_preset_only},
            {"handler", kind},
            {"usage", opt.to_string()},
        });
    }
    return j;
}

static std::string first_difference(const json & a, const json & b, const std::string & path) {
    if (a.is_object() && b.is_object()) {
        for (auto it = a.begin(); it != a.end(); ++it) {
            if (!b.contains(it.key())) {
                return path + "." + it.key();
            }
            auto diff = first_difference(it.value(), b.at(it.key()), path + "." + it.key());
            if (!diff.empty()) {
                return diff;
            }
        }
        return a.size() == b.size() ? "" : path;
    }
    if (a.is_array() && b.is_array() && a.size() == b.size()) {
        for (size_t i = 0; i < a.size(); i++) {
            auto diff = first_difference(a[i], b[i], path + "[" + std::to_string(i) + "]");
            if (!diff.empty()) {
                return diff;
            }
        }
        return "";
    }
    return a.dump() == b.dump() ? "" : path;
}

static void check_roundtrip(const common_params & params) {
    auto data = common_params_to_cbor(params);
    uint8_t * out = nullptr;
    size_t out_len = 0;
    bool ok = llama_args_params_roundtrip(data.data(), data.size(), &out, &out_len);
    std::vector<uint8_t> back_data(out, out + out_len);
    llama_args_free_buffer(out, out_len);
    if (!ok) {
        fprintf(stderr, "params_driver: Params rejected the C++ params: %s\n", std::string(back_data.begin(), back_data.end()).c_str());
        exit(3);
    }
    common_params back{};
    common_params_from_cbor(back_data.data(), back_data.size(), back);
    auto diff = first_difference(dump(params), dump(back), "params");
    if (!diff.empty()) {
        fprintf(stderr, "params_driver: round trip through Params changed %s\n", diff.c_str());
        exit(3);
    }
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--options") == 0) {
        printf("%s\n", options((llama_example) example_from_name(argv[2])).dump(1).c_str());
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: params_driver <out.json> <example> [args...]\n"
                        "       params_driver --options <example>\n");
        return 2;
    }
    const char * out_path = argv[1];
    llama_example ex = (llama_example) example_from_name(argv[2]);

    std::vector<char *> args = {argv[0]};
    for (int i = 3; i < argc; i++) {
        args.push_back(argv[i]);
    }

    common_params params{};
    json before = dump(params);
    bool ok = common_params_parse((int) args.size(), args.data(), params, ex);
    common_log_flush(common_log_main());
    check_roundtrip(params);

    std::ofstream out(out_path);
    out << json({{"ok", ok}, {"before", before}, {"after", dump(params)}}).dump() << "\n";
    return ok ? 0 : 1;
}
