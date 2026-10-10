#include "params-serde.h"

#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

struct cbor {
    enum kind_t { UINT, NINT, BYTES, TEXT, ARRAY, MAP, BOOL, NUL, FLOAT };

    kind_t kind = NUL;
    uint64_t u = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<cbor> items;
    std::vector<std::pair<std::string, cbor>> entries;

    static cbor map() {
        cbor c;
        c.kind = MAP;
        return c;
    }

    static cbor array() {
        cbor c;
        c.kind = ARRAY;
        return c;
    }

    static cbor null() {
        return cbor();
    }

    static cbor boolean(bool v) {
        cbor c;
        c.kind = BOOL;
        c.b = v;
        return c;
    }

    static cbor integer(int64_t v) {
        cbor c;
        if (v < 0) {
            c.kind = NINT;
            c.u = (uint64_t) (-(v + 1));
        } else {
            c.kind = UINT;
            c.u = (uint64_t) v;
        }
        return c;
    }

    static cbor uinteger(uint64_t v) {
        cbor c;
        c.kind = UINT;
        c.u = v;
        return c;
    }

    static cbor real(double v) {
        cbor c;
        c.kind = FLOAT;
        c.f = v;
        return c;
    }

    static cbor bytes(std::string v) {
        cbor c;
        c.kind = BYTES;
        c.s = std::move(v);
        return c;
    }

    void add(const char * key, cbor value) {
        entries.emplace_back(key, std::move(value));
    }

    void push(cbor value) {
        items.push_back(std::move(value));
    }

    const cbor & at(const char * key) const {
        for (const auto & [k, v] : entries) {
            if (k == key) {
                return v;
            }
        }
        throw std::runtime_error(std::string("params cbor: missing field ") + key);
    }

    void expect(kind_t k, const char * what) const {
        if (kind != k) {
            throw std::runtime_error(std::string("params cbor: expected ") + what);
        }
    }

    void expect_map(const char * name, size_t n) const {
        expect(MAP, name);
        if (entries.size() != n) {
            throw std::runtime_error(string_format("params cbor: %s has %zu fields, expected %zu", name, entries.size(), n));
        }
    }

    const std::vector<cbor> & array_of(size_t n, const char * what) const {
        expect(ARRAY, what);
        if (n != SIZE_MAX && items.size() != n) {
            throw std::runtime_error(std::string("params cbor: wrong length for ") + what);
        }
        return items;
    }
};

static void put_head(std::vector<uint8_t> & out, uint8_t major, uint64_t n) {
    major <<= 5;
    if (n < 24) {
        out.push_back(major | (uint8_t) n);
        return;
    }
    int bytes = n <= 0xff ? 1 : n <= 0xffff ? 2 : n <= 0xffffffffULL ? 4 : 8;
    out.push_back(major | (bytes == 1 ? 24 : bytes == 2 ? 25 : bytes == 4 ? 26 : 27));
    for (int i = bytes - 1; i >= 0; i--) {
        out.push_back((uint8_t) (n >> (8 * i)));
    }
}

static void put(std::vector<uint8_t> & out, const cbor & c) {
    switch (c.kind) {
        case cbor::UINT:  put_head(out, 0, c.u); break;
        case cbor::NINT:  put_head(out, 1, c.u); break;
        case cbor::BYTES:
        case cbor::TEXT:
            put_head(out, c.kind == cbor::BYTES ? 2 : 3, c.s.size());
            out.insert(out.end(), c.s.begin(), c.s.end());
            break;
        case cbor::ARRAY:
            put_head(out, 4, c.items.size());
            for (const auto & item : c.items) {
                put(out, item);
            }
            break;
        case cbor::MAP:
            put_head(out, 5, c.entries.size());
            for (const auto & [k, v] : c.entries) {
                put_head(out, 3, k.size());
                out.insert(out.end(), k.begin(), k.end());
                put(out, v);
            }
            break;
        case cbor::BOOL:  out.push_back(c.b ? 0xf5 : 0xf4); break;
        case cbor::NUL:   out.push_back(0xf6); break;
        case cbor::FLOAT: {
            uint64_t bits;
            memcpy(&bits, &c.f, sizeof(bits));
            out.push_back(0xfb);
            for (int i = 7; i >= 0; i--) {
                out.push_back((uint8_t) (bits >> (8 * i)));
            }
            break;
        }
    }
}

struct cbor_reader {
    const uint8_t * p;
    const uint8_t * end;

    uint8_t byte() {
        if (p >= end) {
            throw std::runtime_error("params cbor: truncated data");
        }
        return *p++;
    }

    uint64_t uint_n(int n) {
        uint64_t v = 0;
        for (int i = 0; i < n; i++) {
            v = (v << 8) | byte();
        }
        return v;
    }

    uint64_t arg(uint8_t info) {
        if (info < 24) {
            return info;
        }
        switch (info) {
            case 24: return uint_n(1);
            case 25: return uint_n(2);
            case 26: return uint_n(4);
            case 27: return uint_n(8);
        }
        throw std::runtime_error("params cbor: unsupported length encoding");
    }

    std::string str(uint64_t n) {
        if ((uint64_t) (end - p) < n) {
            throw std::runtime_error("params cbor: truncated data");
        }
        std::string s((const char *) p, (size_t) n);
        p += n;
        return s;
    }

    static double half(uint16_t h) {
        int exp = (h >> 10) & 0x1f;
        int mant = h & 0x3ff;
        double v = exp == 0 ? std::ldexp(mant, -24)
                 : exp == 31 ? (mant == 0 ? INFINITY : NAN)
                 : std::ldexp(mant + 1024, exp - 25);
        return (h & 0x8000) ? -v : v;
    }

    cbor value(int depth = 0) {
        if (depth > 64) {
            throw std::runtime_error("params cbor: nesting too deep");
        }
        uint8_t head = byte();
        uint8_t major = head >> 5;
        uint8_t info = head & 0x1f;
        cbor c;
        switch (major) {
            case 0: c.kind = cbor::UINT; c.u = arg(info); return c;
            case 1: c.kind = cbor::NINT; c.u = arg(info); return c;
            case 2: c.kind = cbor::BYTES; c.s = str(arg(info)); return c;
            case 3: c.kind = cbor::TEXT; c.s = str(arg(info)); return c;
            case 4: {
                c.kind = cbor::ARRAY;
                uint64_t n = arg(info);
                for (uint64_t i = 0; i < n; i++) {
                    c.items.push_back(value(depth + 1));
                }
                return c;
            }
            case 5: {
                c.kind = cbor::MAP;
                uint64_t n = arg(info);
                for (uint64_t i = 0; i < n; i++) {
                    cbor key = value(depth + 1);
                    key.expect(cbor::TEXT, "text map key");
                    c.entries.emplace_back(key.s, value(depth + 1));
                }
                return c;
            }
            case 7:
                switch (info) {
                    case 20: return cbor::boolean(false);
                    case 21: return cbor::boolean(true);
                    case 22: return cbor::null();
                    case 25: return cbor::real(half((uint16_t) uint_n(2)));
                    case 26: {
                        uint32_t bits = (uint32_t) uint_n(4);
                        float f;
                        memcpy(&f, &bits, sizeof(f));
                        return cbor::real(f);
                    }
                    case 27: {
                        uint64_t bits = uint_n(8);
                        double d;
                        memcpy(&d, &bits, sizeof(d));
                        return cbor::real(d);
                    }
                }
                break;
        }
        throw std::runtime_error("params cbor: unsupported item");
    }
};

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static cbor enc(T v) {
    if constexpr (std::is_same_v<T, bool>) {
        return cbor::boolean(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        return cbor::real(v);
    } else if constexpr (std::is_signed_v<T>) {
        return cbor::integer(v);
    } else {
        return cbor::uinteger(v);
    }
}

template <class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
static cbor enc(T v) {
    return cbor::integer((int64_t) v);
}

static cbor enc(const std::string & v) {
    return cbor::bytes(v);
}

static cbor enc(const char * v) {
    return v ? cbor::bytes(v) : cbor::null();
}

static cbor enc(ggml_backend_dev_t v) {
    return v ? cbor::bytes(ggml_backend_dev_name(v)) : cbor::null();
}

static cbor enc(ggml_backend_buffer_type_t v) {
    return v ? cbor::bytes(ggml_backend_buft_name(v)) : cbor::null();
}

static cbor enc(const llama_model_kv_override & v) {
    cbor c = cbor::map();
    c.add("key", cbor::bytes(v.key));
    c.add("tag", enc(v.tag));
    switch (v.tag) {
        case LLAMA_KV_OVERRIDE_TYPE_INT:   c.add("value", cbor::integer(v.val_i64)); break;
        case LLAMA_KV_OVERRIDE_TYPE_FLOAT: c.add("value", cbor::real(v.val_f64));    break;
        case LLAMA_KV_OVERRIDE_TYPE_BOOL:  c.add("value", cbor::boolean(v.val_bool)); break;
        case LLAMA_KV_OVERRIDE_TYPE_STR:   c.add("value", cbor::bytes(v.val_str));   break;
    }
    return c;
}

static cbor enc(const llama_model_tensor_buft_override & v) {
    cbor c = cbor::map();
    c.add("pattern", enc(v.pattern));
    c.add("buft", enc(v.buft));
    return c;
}

static cbor enc(const llama_logit_bias & v) {
    cbor c = cbor::map();
    c.add("token", enc(v.token));
    c.add("bias", enc(v.bias));
    return c;
}

template <class T, size_t N>
static cbor enc(const T (&v)[N]) {
    cbor c = cbor::array();
    for (const auto & e : v) {
        c.push(enc(e));
    }
    return c;
}

template <class T>
static cbor enc(const std::vector<T> & v) {
    cbor c = cbor::array();
    for (const auto & e : v) {
        c.push(enc(e));
    }
    return c;
}

template <class T>
static cbor enc(const std::set<T> & v) {
    cbor c = cbor::array();
    for (const auto & e : v) {
        c.push(enc(e));
    }
    return c;
}

static cbor enc(const std::map<std::string, std::string> & v) {
    cbor c = cbor::array();
    for (const auto & [k, e] : v) {
        cbor pair = cbor::array();
        pair.push(cbor::bytes(k));
        pair.push(cbor::bytes(e));
        c.push(std::move(pair));
    }
    return c;
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
static void dec(const cbor & c, T & v) {
    if constexpr (std::is_same_v<T, bool>) {
        c.expect(cbor::BOOL, "bool");
        v = c.b;
    } else if constexpr (std::is_floating_point_v<T>) {
        c.expect(cbor::FLOAT, "float");
        v = (T) c.f;
    } else {
        if (c.kind == cbor::UINT) {
            if (c.u > (uint64_t) std::numeric_limits<T>::max()) {
                throw std::runtime_error("params cbor: integer out of range");
            }
            v = (T) c.u;
        } else {
            c.expect(cbor::NINT, "integer");
            if (!std::is_signed_v<T> || c.u > (uint64_t) std::numeric_limits<int64_t>::max() ||
                -1 - (int64_t) c.u < (int64_t) std::numeric_limits<T>::min()) {
                throw std::runtime_error("params cbor: integer out of range");
            }
            v = (T) (-1 - (int64_t) c.u);
        }
    }
}

template <class T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
static void dec(const cbor & c, T & v) {
    std::underlying_type_t<T> raw;
    dec(c, raw);
    v = (T) raw;
}

static void dec(const cbor & c, std::string & v) {
    c.expect(cbor::BYTES, "byte string");
    v = c.s;
}

static const char * intern(const std::string & s) {
    static std::mutex mutex;
    static std::list<std::string> strings;
    std::lock_guard<std::mutex> lock(mutex);
    strings.push_back(s);
    return strings.back().c_str();
}

static void dec(const cbor & c, const char *& v) {
    if (c.kind == cbor::NUL) {
        v = nullptr;
        return;
    }
    c.expect(cbor::BYTES, "byte string");
    v = intern(c.s);
}

static void dec(const cbor & c, ggml_backend_dev_t & v) {
    if (c.kind == cbor::NUL) {
        v = nullptr;
        return;
    }
    c.expect(cbor::BYTES, "device name");
    ggml_backend_load_all();
    v = ggml_backend_dev_by_name(c.s.c_str());
    if (!v) {
        throw std::runtime_error("params cbor: unknown device " + c.s);
    }
}

static void dec(const cbor & c, ggml_backend_buffer_type_t & v) {
    if (c.kind == cbor::NUL) {
        v = nullptr;
        return;
    }
    c.expect(cbor::BYTES, "buffer type name");
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * buft = ggml_backend_dev_buffer_type(ggml_backend_dev_get(i));
        if (buft && c.s == ggml_backend_buft_name(buft)) {
            v = buft;
            return;
        }
    }
    throw std::runtime_error("params cbor: unknown buffer type " + c.s);
}

template <size_t N>
static void copy_cstr(const cbor & c, char (&dst)[N]) {
    c.expect(cbor::BYTES, "byte string");
    if (c.s.size() >= N) {
        throw std::runtime_error("params cbor: string too long");
    }
    memset(dst, 0, N);
    memcpy(dst, c.s.data(), c.s.size());
}

static void dec(const cbor & c, llama_model_kv_override & v) {
    c.expect_map("llama_model_kv_override", 3);
    v = {};
    copy_cstr(c.at("key"), v.key);
    dec(c.at("tag"), v.tag);
    const cbor & value = c.at("value");
    switch (v.tag) {
        case LLAMA_KV_OVERRIDE_TYPE_INT:   dec(value, v.val_i64);  break;
        case LLAMA_KV_OVERRIDE_TYPE_FLOAT: dec(value, v.val_f64);  break;
        case LLAMA_KV_OVERRIDE_TYPE_BOOL:  dec(value, v.val_bool); break;
        case LLAMA_KV_OVERRIDE_TYPE_STR:   copy_cstr(value, v.val_str); break;
        default: throw std::runtime_error("params cbor: unknown kv override tag");
    }
}

static void dec(const cbor & c, llama_model_tensor_buft_override & v) {
    c.expect_map("llama_model_tensor_buft_override", 2);
    dec(c.at("pattern"), v.pattern);
    dec(c.at("buft"), v.buft);
}

static void dec(const cbor & c, llama_logit_bias & v) {
    c.expect_map("llama_logit_bias", 2);
    dec(c.at("token"), v.token);
    dec(c.at("bias"), v.bias);
}

template <class T, size_t N>
static void dec(const cbor & c, T (&v)[N]) {
    const auto & items = c.array_of(N, "fixed array");
    for (size_t i = 0; i < N; i++) {
        dec(items[i], v[i]);
    }
}

template <class T>
static void dec(const cbor & c, std::vector<T> & v) {
    const auto & items = c.array_of(SIZE_MAX, "array");
    v.clear();
    v.reserve(items.size());
    for (const auto & item : items) {
        T e{};
        dec(item, e);
        v.push_back(std::move(e));
    }
}

template <class T>
static void dec(const cbor & c, std::set<T> & v) {
    const auto & items = c.array_of(SIZE_MAX, "set");
    v.clear();
    for (const auto & item : items) {
        T e{};
        dec(item, e);
        v.insert(std::move(e));
    }
}

static void dec(const cbor & c, std::map<std::string, std::string> & v) {
    const auto & items = c.array_of(SIZE_MAX, "map");
    v.clear();
    for (const auto & item : items) {
        const auto & pair = item.array_of(2, "map entry");
        std::string k;
        std::string e;
        dec(pair[0], k);
        dec(pair[1], e);
        v[k] = e;
    }
}

#include "params-serde.inc"

std::vector<uint8_t> common_params_to_cbor(const common_params & params) {
    std::vector<uint8_t> out;
    put(out, enc(params));
    return out;
}

void common_params_from_cbor(const uint8_t * data, size_t size, common_params & params) {
    cbor_reader reader{data, data + size};
    cbor c = reader.value();
    if (reader.p != reader.end) {
        throw std::runtime_error("params cbor: trailing data");
    }
    dec(c, params);
}
