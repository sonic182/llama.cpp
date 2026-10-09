#include "json-schema-to-grammar.h"
#include "json.h"

#include <cstdio>
#include <iostream>
#include <string>

static std::string hex(const std::string & s) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    for (unsigned char c : s) {
        out += digits[c >> 4];
        out += digits[c & 15];
    }
    return out;
}

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const std::string grammar = json_schema_to_grammar(common_json::parse(line), true);
            std::printf("ok %s\n", hex(grammar).c_str());
        } catch (const std::exception & e) {
            std::printf("err %s\n", hex(e.what()).c_str());
        }
    }
    return 0;
}
