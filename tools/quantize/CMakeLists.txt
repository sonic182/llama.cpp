set(TARGET llama-quantize)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama-quantize-rs)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if (LLAMA_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        add_test(NAME test-quantize-golden
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/rust-golden/quantize_diff.py check $<TARGET_FILE:${TARGET}>)
        set_tests_properties(test-quantize-golden PROPERTIES
            LABELS main
            ENVIRONMENT "LLAMA_QUANT_FIXTURES=${CMAKE_BINARY_DIR}/quantize-fixtures")
    endif()
endif()
