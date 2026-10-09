#include "llama_quantize.h"

// satisfies -Wmissing-declarations
int llama_quantize(int argc, char ** argv);

int llama_quantize(int argc, char ** argv) {
    return llama_rs_quantize(argc, argv);
}
