add_subdirectory(${PROJECT_SOURCE_DIR}/common/rust-shim ${CMAKE_BINARY_DIR}/common/rust-shim)

find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(LLAMA_RUST_CARGO_FEATURES --features llama-http/tls)
if (LLAMA_LLGUIDANCE)
    list(APPEND LLAMA_RUST_CARGO_FEATURES --features llama-rs/llguidance)
endif()

# llama-sys always links libllama as a shared library, so the binaries cannot link against static libs
set(LLAMA_RUST_CARGO_EXCLUDES "")
if (NOT BUILD_SHARED_LIBS)
    list(APPEND LLAMA_RUST_CARGO_EXCLUDES --exclude llama-gguf-split --exclude llama-tokenize)
endif()

add_custom_target(llama-rust ALL
    COMMAND ${CMAKE_COMMAND} -E env
        LLAMA_LIB_DIR=$<TARGET_FILE_DIR:llama-ext-c>
        CARGO_TARGET_DIR=${CMAKE_BINARY_DIR}/rust
        ${CARGO_EXECUTABLE} build --workspace --profile $<IF:$<CONFIG:Debug>,dev,${LLAMA_RUST_CARGO_PROFILE}>
        ${LLAMA_RUST_CARGO_FEATURES}
        ${LLAMA_RUST_CARGO_EXCLUDES}
        --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    COMMAND_EXPAND_LISTS
    USES_TERMINAL
    VERBATIM
)

add_dependencies(llama-rust llama llama-ext-c)

add_library(llama-rs STATIC IMPORTED GLOBAL)
set_target_properties(llama-rs PROPERTIES
    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/rust/${LLAMA_RUST_CARGO_PROFILE}/libllama_rs.a
    IMPORTED_LOCATION_DEBUG ${CMAKE_BINARY_DIR}/rust/debug/libllama_rs.a
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/llama-rs/include;${CMAKE_CURRENT_SOURCE_DIR}/llama-schema/include;${CMAKE_CURRENT_SOURCE_DIR}/llama-download/include;${CMAKE_CURRENT_SOURCE_DIR}/llama-quantize/include;${CMAKE_CURRENT_SOURCE_DIR}/llama-sampling/include"
    INTERFACE_LINK_LIBRARIES "llama;${CMAKE_DL_LIBS};m;${CMAKE_THREAD_LIBS_INIT}"
)
add_dependencies(llama-rs llama-rust)

add_library(llama-http INTERFACE)
target_include_directories(llama-http INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/llama-http/include)
add_dependencies(llama-http llama-rs)

add_library(llama-quantize-rs INTERFACE)
target_include_directories(llama-quantize-rs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/llama-quantize/include)
add_dependencies(llama-quantize-rs llama-rs)
