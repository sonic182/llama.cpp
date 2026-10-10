#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llama_schema_buf {
    char * data;
    size_t len;
} llama_schema_buf;

typedef struct llama_schema_result {
    llama_schema_buf grammar;
    llama_schema_buf error;
    llama_schema_buf warnings;
} llama_schema_result;

int  llama_schema_to_grammar(const char * schema, size_t len, llama_schema_result * out);
void llama_schema_result_free(llama_schema_result * result);

#ifdef __cplusplus
}
#endif
