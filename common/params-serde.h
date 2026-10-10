#pragma once

#include "common.h"

#include <cstddef>
#include <cstdint>
#include <vector>

std::vector<uint8_t> common_params_to_cbor(const common_params & params);

void common_params_from_cbor(const uint8_t * data, size_t size, common_params & params);
