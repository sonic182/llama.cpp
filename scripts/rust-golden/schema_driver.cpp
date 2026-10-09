#include "json-schema-to-grammar.h"
#include "json.h"

#include <cstdio>
#include <iostream>
#include <stdexcept>
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

int main(int argc, char ** argv) {
    const bool rust = argc > 1 && std::string(argv[1]) == "rust";
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const common_json schema = common_json::parse(line);
            std::string grammar;
            if (rust) {
                grammar = json_schema_to_grammar(schema, true);
            } else {
                try {
                    grammar = json_schema_to_grammar(common_chat_schema_from_json(schema));
                } catch (const std::runtime_error & e) {
                    throw std::invalid_argument(std::string("JSON schema conversion failed:\n") + e.what());
                }
            }
            std::printf("ok %s\n", hex(grammar).c_str());
        } catch (const std::exception & e) {
            std::printf("err %s\n", hex(e.what()).c_str());
        }
    }
    return 0;
}
