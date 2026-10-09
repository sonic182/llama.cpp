#include "imatrix-loader.h"
#include "log.h"

#include "llama_quantize.h"

static void log_imatrix_error(const char * message) {
    LOG_ERR("%s", message);
}

bool common_imatrix_load(const std::string & fname, common_imatrix & imatrix) {
    llama_imatrix * loaded = llama_imatrix_load(fname.c_str(), log_imatrix_error);
    if (!loaded) {
        return false;
    }

    const llama_imatrix_view view = llama_imatrix_get(loaded);
    for (size_t i = 0; i < view.n_entries; ++i) {
        const llama_imatrix_entry & src = view.entries[i];
        common_imatrix_entry & dst = imatrix.entries[src.name];
        dst.sums.assign(src.sums, src.sums + src.n_sums);
        dst.counts.assign(src.counts, src.counts + src.n_counts);
    }
    imatrix.datasets.insert(imatrix.datasets.end(), view.datasets, view.datasets + view.n_datasets);
    imatrix.chunk_count  = view.chunk_count;
    imatrix.chunk_size   = view.chunk_size;
    imatrix.is_legacy    = view.is_legacy;
    imatrix.has_metadata = view.has_metadata;

    llama_imatrix_free(loaded);
    return true;
}
