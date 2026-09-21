#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static constexpr size_t PTQ_BLOCK = 28, Q2_BLOCK = 18, QK = 128;

static int ptq1_trit(const uint8_t * b, size_t pos) {
    static constexpr uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    uint8_t byte; int digit;
    if (pos < 80) { digit = int(pos / 16); byte = b[pos % 16]; }
    else if (pos < 120) { const size_t q = pos - 80; digit = int(q / 8); byte = b[16 + q % 8]; }
    else { const size_t q = pos - 120; digit = int(q / 2); byte = b[24 + q % 2]; }
    const uint8_t wrapped = static_cast<uint8_t>(byte * pow3[digit]);
    return (int(wrapped) * 3 >> 8) - 1;
}

static std::array<int8_t, QK> ptq1_oracle(const uint8_t * b) {
    static constexpr size_t stages[2] = { 16, 8 };
    static constexpr uint8_t pow3[5] = { 1, 3, 9, 27, 81 };
    std::array<int8_t, QK> out {};
    size_t pos = 0;
    for (size_t stage = 0, j = 0; stage < 2; ++stage) {
        const size_t c = stages[stage];
        for (; j + c <= 24; j += c) {
            for (size_t digit = 0; digit < 5; ++digit) {
                for (size_t m = 0; m < c; ++m) {
                    const uint8_t wrapped = static_cast<uint8_t>(b[j + m] * pow3[digit]);
                    out[pos++] = int8_t((int(wrapped) * 3 >> 8) - 1);
                }
            }
        }
    }
    for (size_t digit = 0; digit < 4; ++digit) {
        for (size_t h = 0; h < 2; ++h) {
            const uint8_t wrapped = static_cast<uint8_t>(b[24 + h] * pow3[digit]);
            out[pos++] = int8_t((int(wrapped) * 3 >> 8) - 1);
        }
    }
    if (pos != QK) throw std::runtime_error("PTQ1 oracle position count mismatch");
    return out;
}
static void repack_block(const uint8_t * in, uint8_t * out) {
    std::array<int8_t, QK> trits {};
    for (size_t pos = 0; pos < QK; ++pos) trits[pos] = int8_t(ptq1_trit(in, pos));
    std::memcpy(out, in + 26, 2);
    std::memcpy(out + Q2_BLOCK, in + 26, 2);
    std::fill(out + 2, out + Q2_BLOCK, uint8_t(0));
    std::fill(out + Q2_BLOCK + 2, out + 2 * Q2_BLOCK, uint8_t(0));
    for (size_t pos = 0; pos < QK; ++pos) {
        const uint8_t q = uint8_t(trits[pos] + 1);
        const size_t block = pos / 64, j = pos % 64;
        out[block * Q2_BLOCK + 2 + j / 4] |= uint8_t(q << ((j % 4) * 2));
    }
}

static void verify_block(const uint8_t * in, const uint8_t * out) {
    if (std::memcmp(out, in + 26, 2) != 0 || std::memcmp(out + Q2_BLOCK, in + 26, 2) != 0) {
        throw std::runtime_error("scale mismatch");
    }
    const auto expected = ptq1_oracle(in);
    for (size_t pos = 0; pos < QK; ++pos) {
        const size_t block = pos / 64, j = pos % 64;
        const uint8_t q = (out[block * Q2_BLOCK + 2 + j / 4] >> ((j % 4) * 2)) & 3;
        if (int(q) - 1 != expected[pos]) throw std::runtime_error("trit mismatch");
    }
}

static void copy_bytes(std::ifstream & in, std::ofstream & out, uint64_t offset, size_t size) {
    const size_t chunk = 1u << 20;
    std::vector<char> buf(std::min(chunk, size));
    in.seekg(static_cast<std::streamoff>(offset));
    if (!in) throw std::runtime_error("input seek failed");
    while (size) {
        const size_t n = std::min(size, buf.size());
        in.read(buf.data(), static_cast<std::streamsize>(n));
        if (in.gcount() != static_cast<std::streamsize>(n)) throw std::runtime_error("input read failed");
        out.write(buf.data(), static_cast<std::streamsize>(n));
        if (!out) throw std::runtime_error("output write failed");
        size -= n;
    }
}

static void verify_ptq_payload(std::ifstream & in, std::ifstream & out, size_t input_size) {
    constexpr size_t max_blocks = 4096;
    std::vector<uint8_t> src(max_blocks * PTQ_BLOCK), dst(max_blocks * 2 * Q2_BLOCK);
    size_t remaining = input_size;
    while (remaining) {
        const size_t blocks = std::min(max_blocks, remaining / PTQ_BLOCK);
        const size_t src_bytes = blocks * PTQ_BLOCK, dst_bytes = blocks * 2 * Q2_BLOCK;
        in.read(reinterpret_cast<char *>(src.data()), static_cast<std::streamsize>(src_bytes));
        out.read(reinterpret_cast<char *>(dst.data()), static_cast<std::streamsize>(dst_bytes));
        if (in.gcount() != static_cast<std::streamsize>(src_bytes) || out.gcount() != static_cast<std::streamsize>(dst_bytes)) throw std::runtime_error("validation payload read failed");
        for (size_t b = 0; b < blocks; ++b) verify_block(src.data() + b * PTQ_BLOCK, dst.data() + b * 2 * Q2_BLOCK);
        remaining -= src_bytes;
    }
}
static void validate_output(const fs::path & input, const fs::path & output,
                            const gguf_context * in_ctx, const gguf_context * out_ctx) {
    if (gguf_get_alignment(in_ctx) != gguf_get_alignment(out_ctx)) throw std::runtime_error("alignment changed");
    const size_t meta_size = gguf_get_meta_size(out_ctx);
    std::vector<uint8_t> expected_meta(meta_size), actual_meta(meta_size);
    gguf_get_meta_data(out_ctx, expected_meta.data());
    std::ifstream meta_file(output, std::ios::binary);
    meta_file.read(reinterpret_cast<char *>(actual_meta.data()), static_cast<std::streamsize>(meta_size));
    if (meta_file.gcount() != static_cast<std::streamsize>(meta_size) || actual_meta != expected_meta) {
        throw std::runtime_error("output metadata mismatch");
    }
    std::ifstream in(input, std::ios::binary), out(output, std::ios::binary);
    if (!in || !out) throw std::runtime_error("validation open failed");
    uint64_t expected_end = meta_size;
    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const auto input_type = gguf_get_tensor_type(in_ctx, i);
        const size_t input_size = gguf_get_tensor_size(in_ctx, i);
        const size_t output_size = gguf_get_tensor_size(out_ctx, i);
        const uint64_t input_offset = uint64_t(gguf_get_data_offset(in_ctx)) + gguf_get_tensor_offset(in_ctx, i);
        const uint64_t output_offset = uint64_t(meta_size) + gguf_get_tensor_offset(out_ctx, i);
        expected_end = std::max(expected_end, output_offset + GGML_PAD(output_size, gguf_get_alignment(out_ctx)));
        in.seekg(static_cast<std::streamoff>(input_offset));
        out.seekg(static_cast<std::streamoff>(output_offset));
        if (input_type != GGML_TYPE_PTQ1_0) {
            std::vector<char> a(1u << 20), b(1u << 20);
            size_t rem = input_size;
            while (rem) {
                const size_t n = std::min(rem, a.size());
                in.read(a.data(), static_cast<std::streamsize>(n)); out.read(b.data(), static_cast<std::streamsize>(n));
                if (in.gcount() != static_cast<std::streamsize>(n) || out.gcount() != static_cast<std::streamsize>(n) ||
                    std::memcmp(a.data(), b.data(), n) != 0) throw std::runtime_error("non-PTQ payload mismatch");
                rem -= n;
            }
        } else {
            if (input_size % PTQ_BLOCK != 0 || output_size != input_size / PTQ_BLOCK * 2 * Q2_BLOCK) {
                throw std::runtime_error("validation size mismatch");
            }
            verify_ptq_payload(in, out, input_size);
        }
    }
    if (fs::file_size(output) != expected_end) throw std::runtime_error("output file size mismatch");
}
int main(int argc, char ** argv) {
    const bool verify_only = argc == 4 && std::string(argv[1]) == "--verify-only";
    if ((!verify_only && argc != 3) || (verify_only && argc != 4)) {
        std::fprintf(stderr, "usage: %s input.gguf output.gguf | --verify-only input.gguf candidate.gguf\n", argv[0]); return 2;
    }
    const fs::path input = verify_only ? argv[2] : argv[1];
    const fs::path output = verify_only ? argv[3] : argv[2];
    const fs::path temp = output.string() + ".tmp";
    if (!fs::exists(input) || (verify_only ? !fs::exists(output) : (fs::exists(output) || fs::exists(temp)))) return 2;

    ggml_context * meta = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context * in_ctx = gguf_init_from_file(input.string().c_str(), params);
    if (!in_ctx || !meta) return 1;
    gguf_context * out_ctx = gguf_init_empty();
    if (!out_ctx) { gguf_free(in_ctx); ggml_free(meta); return 1; }
    try {
        gguf_set_kv(out_ctx, in_ctx);
        gguf_set_val_u32(out_ctx, "general.file_type", LLAMA_FTYPE_MOSTLY_Q2_0);
        const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
        size_t converted = 0, copied = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(in_ctx, i);
            ggml_tensor * tensor = ggml_get_tensor(meta, name);
            if (!tensor) throw std::runtime_error("missing tensor metadata");
            gguf_add_tensor(out_ctx, tensor);
            if (gguf_get_tensor_type(in_ctx, i) == GGML_TYPE_PTQ1_0) {
                gguf_set_tensor_type(out_ctx, name, GGML_TYPE_Q2_0); ++converted;
            } else ++copied;
        }
        if (verify_only) {
            validate_output(input, output, in_ctx, out_ctx);
            gguf_free(out_ctx); gguf_free(in_ctx); ggml_free(meta);
            std::printf("verify_only_ok converted_ptq1=%zu copied=%zu tensors=%lld\n", converted, copied, (long long) n_tensors);
            return 0;
        }
        if (!gguf_write_to_file(out_ctx, temp.string().c_str(), true)) throw std::runtime_error("metadata write failed");
        std::ofstream out(temp, std::ios::binary | std::ios::app);
        std::ifstream in(input, std::ios::binary);
        if (!out || !in) throw std::runtime_error("payload open failed");
        const size_t output_meta_size = gguf_get_meta_size(out_ctx);
        out.seekp(static_cast<std::streamoff>(output_meta_size));
        if (!out) throw std::runtime_error("output data seek failed");
        for (int64_t i = 0; i < n_tensors; ++i) {
            const uint64_t expected_offset = uint64_t(output_meta_size) + gguf_get_tensor_offset(out_ctx, i);
            if (static_cast<uint64_t>(out.tellp()) != expected_offset) throw std::runtime_error("output tensor offset mismatch at tensor " + std::to_string(i) + " got " + std::to_string(static_cast<uint64_t>(out.tellp())) + " expected " + std::to_string(expected_offset));
            const auto input_type = gguf_get_tensor_type(in_ctx, i);
            const auto output_type = gguf_get_tensor_type(out_ctx, i);
            const size_t input_size = gguf_get_tensor_size(in_ctx, i);
            const size_t output_size = gguf_get_tensor_size(out_ctx, i);
            const uint64_t offset = uint64_t(gguf_get_data_offset(in_ctx)) + gguf_get_tensor_offset(in_ctx, i);
            if (input_type != GGML_TYPE_PTQ1_0) {
                copy_bytes(in, out, offset, input_size);
            } else {
                if (output_type != GGML_TYPE_Q2_0 || input_size % PTQ_BLOCK != 0 ||
                    output_size != input_size / PTQ_BLOCK * 2 * Q2_BLOCK) throw std::runtime_error("size mismatch");
                constexpr size_t max_blocks = 4096;
                std::vector<uint8_t> src(max_blocks * PTQ_BLOCK), dst(max_blocks * 2 * Q2_BLOCK);
                size_t remaining = input_size; uint64_t current = offset;
                while (remaining) {
                    const size_t blocks = std::min(max_blocks, remaining / PTQ_BLOCK), bytes = blocks * PTQ_BLOCK;
                    in.seekg(static_cast<std::streamoff>(current));
                    in.read(reinterpret_cast<char *>(src.data()), static_cast<std::streamsize>(bytes));
                    if (in.gcount() != static_cast<std::streamsize>(bytes)) throw std::runtime_error("PTQ1 read failed");
                    for (size_t b = 0; b < blocks; ++b) {
                        repack_block(src.data() + b * PTQ_BLOCK, dst.data() + b * 2 * Q2_BLOCK);
                        verify_block(src.data() + b * PTQ_BLOCK, dst.data() + b * 2 * Q2_BLOCK);
                    }
                    out.write(reinterpret_cast<const char *>(dst.data()), static_cast<std::streamsize>(blocks * 2 * Q2_BLOCK));
                    if (!out) throw std::runtime_error("PTQ1 write failed");
                    current += bytes; remaining -= bytes;
                }
            }
            const size_t pad = GGML_PAD(output_size, gguf_get_alignment(out_ctx)) - output_size;
            for (size_t p = 0; p < pad; ++p) out.put('\0');
        }
        out.flush();
        if (!out) throw std::runtime_error("output flush failed");
        out.close(); in.close();
        if (out.fail() || in.fail()) throw std::runtime_error("output close failed");
        validate_output(input, temp, in_ctx, out_ctx);
        fs::rename(temp, output);
        gguf_free(out_ctx); gguf_free(in_ctx); ggml_free(meta);
        std::printf("converted_ptq1=%zu copied=%zu tensors=%lld\n", converted, copied, (long long) n_tensors);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "conversion failed: %s\n", e.what());
        gguf_free(out_ctx); gguf_free(in_ctx); ggml_free(meta);
        if (!verify_only) { std::error_code ec; fs::remove(temp, ec); } return 1;
    }
}
