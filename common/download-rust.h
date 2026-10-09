#pragma once

#include "download.h"
#include "json.h"
#include "llama_download.h"

#include <stdexcept>
#include <string>
#include <vector>

inline common_json llama_dl_unwrap(char * raw) {
    const std::string text = raw;
    llama_dl_free_string(raw);
    common_json envelope = common_json::parse(text);
    if (!envelope.value("ok", false)) {
        const std::string message = envelope.value("error", std::string());
        if (envelope.value("kind", std::string()) == "invalid_argument") {
            throw std::invalid_argument(message);
        }
        throw std::runtime_error(message);
    }
    return envelope.at("value");
}

struct llama_dl_headers {
    std::vector<const char *> names;
    std::vector<const char *> values;

    explicit llama_dl_headers(const common_header_list & headers) {
        for (const auto & h : headers) {
            names.push_back(h.first.c_str());
            values.push_back(h.second.c_str());
        }
    }
};

inline hf_cache::hf_file llama_dl_to_hf_file(const common_json & j) {
    hf_cache::hf_file file;
    if (j.is_null()) {
        return file;
    }
    file.path       = j.value("path", std::string());
    file.url        = j.value("url", std::string());
    file.local_path = j.value("local_path", std::string());
    file.final_path = j.value("final_path", std::string());
    file.oid        = j.value("oid", std::string());
    file.repo_id    = j.value("repo_id", std::string());
    return file;
}

inline hf_cache::hf_files llama_dl_to_hf_files(const common_json & list) {
    hf_cache::hf_files files;
    for (size_t i = 0; i < list.size(); ++i) {
        files.push_back(llama_dl_to_hf_file(list.at(i)));
    }
    return files;
}
