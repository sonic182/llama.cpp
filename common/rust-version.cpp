#include "rust-version.h"

#include "llama_rs.h"

const char * common_rust_version() {
    return llama_rs_version();
}
