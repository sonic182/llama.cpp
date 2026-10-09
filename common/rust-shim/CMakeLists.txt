set(TARGET llama-ext-c)

add_library(${TARGET}
    llama_ext_c.cpp
    llama_ext_c.h
)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PUBLIC llama)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
