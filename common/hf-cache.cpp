#include "hf-cache.h"
#include "download-rust.h"

#include <string>

namespace hf_cache {

hf_files get_repo_files(const std::string & repo_id, const std::string & token) {
    return llama_dl_to_hf_files(llama_dl_unwrap(llama_dl_hf_repo_files(repo_id.c_str(), token.c_str())));
}

hf_files get_cached_files(const std::string & repo_id) {
    return llama_dl_to_hf_files(llama_dl_unwrap(llama_dl_hf_cached_files(repo_id.c_str())));
}

std::string finalize_file(const hf_file & file) {
    return llama_dl_unwrap(llama_dl_hf_finalize(file.local_path.c_str(), file.final_path.c_str())).get<std::string>();
}

bool remove_cached_repo(const std::string & repo_id) {
    return llama_dl_unwrap(llama_dl_hf_remove_repo(repo_id.c_str())).get<bool>();
}

std::string get_cache_path() {
    return llama_dl_unwrap(llama_dl_hf_cache_path()).get<std::string>();
}

} // namespace hf_cache
