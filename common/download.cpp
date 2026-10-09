#include "common.h"
#include "download.h"
#include "download-rust.h"
#include "hf-cache.h"
#include "json.h"
#include "log.h"

#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
}

static void llama_dl_log(int level, const char * message) {
    switch (level) {
        case 0:  LOG_DBG("%s\n", message); break;
        case 1:  LOG_INF("%s\n", message); break;
        case 2:  LOG_WRN("%s\n", message); break;
        default: LOG_ERR("%s\n", message); break;
    }
}

static const struct llama_dl_log_installer {
    llama_dl_log_installer() { llama_dl_set_log_sink(llama_dl_log); }
} llama_dl_log_installer_instance;


class ProgressBar : public common_download_callback {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
#if defined(_WIN32)
        return _isatty(_fileno(stdout));
#else
        return isatty(1);
#endif
    }

public:
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = p.url;

        if (auto pos = filename.rfind('/'); pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        if (auto pos = filename.find('?'); pos != std::string::npos) {
            filename = filename.substr(0, pos);
        }
        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

static common_download_progress to_progress(const llama_dl_progress & p) {
    common_download_progress out;
    out.url        = p.url ? p.url : "";
    out.downloaded = p.downloaded;
    out.total      = p.total;
    out.cached     = p.cached;
    return out;
}

static llama_dl_callback to_raw_callback(common_download_callback * callback) {
    return {
        callback,
        [](void * ctx, const llama_dl_progress * p) {
            static_cast<common_download_callback *>(ctx)->on_start(to_progress(*p));
        },
        [](void * ctx, const llama_dl_progress * p) {
            static_cast<common_download_callback *>(ctx)->on_update(to_progress(*p));
        },
        [](void * ctx, const llama_dl_progress * p, bool ok) {
            static_cast<common_download_callback *>(ctx)->on_done(to_progress(*p), ok);
        },
        [](void * ctx) {
            return static_cast<common_download_callback *>(ctx)->is_cancelled();
        },
    };
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    const common_json value = llama_dl_unwrap(llama_dl_split_repo_tag(hf_repo_with_tag.c_str()));
    return { value.at("repo").get<std::string>(), value.at("tag").get<std::string>() };
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts,
                                bool skip_etag) {
    ProgressBar tty_cb;
    common_download_callback * callback = opts.callback;
    if (!opts.offline && !callback) {
        callback = &tty_cb;
    }

    const llama_dl_headers headers(opts.headers);
    llama_dl_callback raw = callback ? to_raw_callback(callback) : llama_dl_callback{};
    return llama_dl_file_single(url.c_str(), path.c_str(),
                                headers.names.data(), headers.values.data(), headers.names.size(),
                                opts.bearer_token.empty() ? nullptr : opts.bearer_token.c_str(),
                                opts.offline, skip_etag, callback ? &raw : nullptr);
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    const llama_dl_headers headers(params.headers);
    int64_t status = 0;
    uint8_t * body = nullptr;
    size_t len = 0;

    char * error = llama_dl_remote_get(url.c_str(), headers.names.data(), headers.values.data(), headers.names.size(),
                                       params.timeout > 0 ? uint64_t(params.timeout) : 0,
                                       params.max_size > 0 ? size_t(params.max_size) : 0,
                                       &status, &body, &len);
    if (error) {
        const std::string message = error;
        llama_dl_free_string(error);
        throw std::runtime_error(message);
    }

    std::vector<char> data(body, body + len);
    llama_dl_free_buffer(body, len);
    return { long(status), std::move(data) };
}

std::vector<std::string> common_download_get_all_parts(const std::string & url) {
    const common_json parts = llama_dl_unwrap(llama_dl_all_parts(url.c_str()));
    std::vector<std::string> result;
    for (size_t i = 0; i < parts.size(); ++i) {
        result.push_back(parts.at(i).get<std::string>());
    }
    return result;
}

std::vector<common_cached_model_info> common_list_cached_models() {
    const common_json models = llama_dl_unwrap(llama_dl_list_cached_models());
    std::vector<common_cached_model_info> result;
    for (size_t i = 0; i < models.size(); ++i) {
        result.push_back({ models.at(i).at("repo").get<std::string>(), models.at(i).at("tag").get<std::string>() });
    }
    return result;
}

std::string common_download_resolve_path(const std::string & hf_repo_with_tag, const std::string & hf_file) {
    const common_json path = llama_dl_unwrap(llama_dl_resolve_path(hf_repo_with_tag.c_str(), hf_file.c_str()));
    return path.is_null() ? std::string() : path.get<std::string>();
}

bool common_download_remove(const std::string & hf_repo_with_tag) {
    return llama_dl_unwrap(llama_dl_remove(hf_repo_with_tag.c_str())).get<bool>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    return llama_dl_unwrap(llama_dl_docker_resolve(docker.c_str())).get<std::string>();
}

common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts) {
    const uint32_t flags =
        (opts.download_mmproj ? 1u : 0u) |
        (opts.download_mtp    ? 2u : 0u) |
        (opts.download_eagle3 ? 4u : 0u) |
        (opts.download_dflash ? 8u : 0u) |
        (opts.download_dspark ? 16u : 0u) |
        (opts.offline         ? 32u : 0u);

    const common_json value = llama_dl_unwrap(llama_dl_hf_plan(model.hf_repo.c_str(), model.hf_file.c_str(),
                                                               opts.bearer_token.c_str(), flags));
    common_download_hf_plan plan;
    plan.preset      = llama_dl_to_hf_file(value.at("preset"));
    plan.primary     = llama_dl_to_hf_file(value.at("primary"));
    plan.model_files = llama_dl_to_hf_files(value.at("model_files"));
    plan.mmproj      = llama_dl_to_hf_file(value.at("mmproj"));
    plan.mtp         = llama_dl_to_hf_file(value.at("mtp"));
    plan.eagle3      = llama_dl_to_hf_file(value.at("eagle3"));
    plan.dflash      = llama_dl_to_hf_file(value.at("dflash"));
    plan.dspark      = llama_dl_to_hf_file(value.at("dspark"));
    return plan;
}

void common_download_run_tasks(const std::vector<common_download_task> & tasks) {
    std::vector<std::future<int>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task]() {
                return common_download_file_single(task.url, task.local_path, task.opts, task.is_hf);
            }
        ));
    }

    for (size_t i = 0; i < futures.size(); ++i) {
        std::string url = tasks[i].url;
        int status = futures[i].get();
        bool is_ok = is_http_status_ok(status);
        if (!is_ok) {
            throw std::runtime_error(string_format("Download '%s' failed with status code: %d", url.c_str(), status));
        }
    }
}
