/**
 * Encoding-anomaly + data-munging GPU kernel (Phase 6, v1.4.0).
 *
 * One thread per (passphrase × encoding) pair. For each enabled
 * encoding bit, mutate the passphrase, derive priv via SHA-256 (or
 * SHA-256d if double_hash is set), and run the existing puzzle-target
 * mask compare.
 *
 * Mirrors src/gpu/v2/encoding_munge_cpu.hpp exactly. KAT-tested CPU
 * reference is the spec.
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"
#include "puzzle_check.cuh"   // static __device__ helper + extern __constant__

#include <cuda_runtime.h>

namespace collider {
namespace gpu {
namespace v2 {

namespace {

// Encoding bit values match the AddressType bit layout for plumbing
// simplicity; they live in their own enum-like in the CPU header.
enum DevEncoding : uint8_t {
    DE_UTF8            = 0,
    DE_UTF16_LE        = 1,
    DE_UTF16_BE        = 2,
    DE_UTF32_LE        = 3,
    DE_UTF32_BE        = 4,
    DE_LATIN1          = 5,
    DE_STRIP_NON_ASCII = 6,
    DE_UPPER_ASCII     = 7,
    DE_LOWER_ASCII     = 8,
    DE_NULL_TERMINATED = 9,    // append \0 (Phase 6 spec line 1)
    DE_ENCODING_COUNT  = 10
};

// Decode one UTF-8 codepoint from `in` starting at *pos. Advances pos.
// Returns -1 on malformed input.
__device__ __forceinline__ int32_t utf8_decode_one(
    const uint8_t* in, uint32_t in_len, uint32_t& pos)
{
    if (pos >= in_len) return -1;
    uint8_t c0 = in[pos];
    if (c0 < 0x80) { ++pos; return c0; }
    if ((c0 & 0xE0) == 0xC0) {
        if (pos + 1 >= in_len) return -1;
        uint8_t c1 = in[pos + 1];
        if ((c1 & 0xC0) != 0x80) return -1;
        pos += 2;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0) {
        if (pos + 2 >= in_len) return -1;
        uint8_t c1 = in[pos + 1], c2 = in[pos + 2];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return -1;
        pos += 3;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0) {
        if (pos + 3 >= in_len) return -1;
        uint8_t c1 = in[pos + 1], c2 = in[pos + 2], c3 = in[pos + 3];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return -1;
        pos += 4;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12)
             | ((c2 & 0x3F) << 6)  |  (c3 & 0x3F);
    }
    return -1;
}

// Apply encoding `enc` to the input bytes; write result into `out`.
// Returns the number of bytes written, or -1 if `enc` cannot represent
// the input (e.g. non-Latin1 codepoint into LATIN1).
__device__ static int32_t munge(
    DevEncoding enc, const uint8_t* in, uint32_t in_len,
    uint8_t* out, uint32_t out_cap)
{
    auto write = [&](uint8_t b, uint32_t& pos) -> bool {
        if (pos >= out_cap) return false;
        out[pos++] = b;
        return true;
    };
    uint32_t pos = 0;
    switch (enc) {
        case DE_UTF8: {
            for (uint32_t i = 0; i < in_len; ++i) if (!write(in[i], pos)) return -1;
            return (int32_t)pos;
        }
        case DE_NULL_TERMINATED: {
            for (uint32_t i = 0; i < in_len; ++i) if (!write(in[i], pos)) return -1;
            if (!write(0, pos)) return -1;
            return (int32_t)pos;
        }
        case DE_UTF16_LE: case DE_UTF16_BE:
        case DE_UTF32_LE: case DE_UTF32_BE: case DE_LATIN1: {
            uint32_t ip = 0;
            while (ip < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, ip);
                if (cp < 0) return -1;
                if (enc == DE_UTF16_LE || enc == DE_UTF16_BE) {
                    if (cp <= 0xFFFF) {
                        uint16_t v = (uint16_t)cp;
                        uint8_t lo = (uint8_t)(v & 0xFF), hi = (uint8_t)((v >> 8) & 0xFF);
                        if (enc == DE_UTF16_LE) { if (!write(lo, pos) || !write(hi, pos)) return -1; }
                        else                    { if (!write(hi, pos) || !write(lo, pos)) return -1; }
                    } else {
                        uint32_t v = (uint32_t)cp - 0x10000;
                        const uint16_t pair[2] = {
                            (uint16_t)(0xD800 | (v >> 10)),
                            (uint16_t)(0xDC00 | (v & 0x3FF)),
                        };
                        for (int k = 0; k < 2; ++k) {
                            uint16_t s = pair[k];
                            uint8_t lo = (uint8_t)(s & 0xFF), hi = (uint8_t)((s >> 8) & 0xFF);
                            if (enc == DE_UTF16_LE) { if (!write(lo, pos) || !write(hi, pos)) return -1; }
                            else                    { if (!write(hi, pos) || !write(lo, pos)) return -1; }
                        }
                    }
                } else if (enc == DE_UTF32_LE) {
                    uint32_t v = (uint32_t)cp;
                    if (!write((uint8_t)(v      ), pos)) return -1;
                    if (!write((uint8_t)(v >>  8), pos)) return -1;
                    if (!write((uint8_t)(v >> 16), pos)) return -1;
                    if (!write((uint8_t)(v >> 24), pos)) return -1;
                } else if (enc == DE_UTF32_BE) {
                    uint32_t v = (uint32_t)cp;
                    if (!write((uint8_t)(v >> 24), pos)) return -1;
                    if (!write((uint8_t)(v >> 16), pos)) return -1;
                    if (!write((uint8_t)(v >>  8), pos)) return -1;
                    if (!write((uint8_t)(v      ), pos)) return -1;
                } else { // LATIN1
                    if (cp > 0xFF) return -1;
                    if (!write((uint8_t)cp, pos)) return -1;
                }
            }
            return (int32_t)pos;
        }
        case DE_STRIP_NON_ASCII: {
            for (uint32_t i = 0; i < in_len; ++i) {
                if (in[i] < 0x80) {
                    if (!write(in[i], pos)) return -1;
                }
            }
            return (int32_t)pos;
        }
        case DE_UPPER_ASCII: {
            for (uint32_t i = 0; i < in_len; ++i) {
                uint8_t b = in[i];
                if (b >= 'a' && b <= 'z') b -= 0x20;
                if (!write(b, pos)) return -1;
            }
            return (int32_t)pos;
        }
        case DE_LOWER_ASCII: {
            for (uint32_t i = 0; i < in_len; ++i) {
                uint8_t b = in[i];
                if (b >= 'A' && b <= 'Z') b += 0x20;
                if (!write(b, pos)) return -1;
            }
            return (int32_t)pos;
        }
        default: return -1;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Per-(passphrase, encoding) kernel.
//
// Thread mapping: index = pp_idx * num_encodings_set + enc_local.
// num_encodings_set == popcount(encoding_mask) so we don't waste threads
// on disabled bits.
// ---------------------------------------------------------------------------

// Per-encoding worst-case expansion factor:
//   UTF-8 / Latin1 / strip / case-fold       : 1x
//   UTF-16-LE / UTF-16-BE                    : 4x (surrogate pair from a single
//                                                  4-byte UTF-8 codepoint)
//   UTF-32-LE / UTF-32-BE                    : 4x (one codepoint == 4 bytes)
//   NULL_TERMINATED                          : 1x + 1 byte
//
// Cap input at 64 bytes pre-mutation; max munged size = 64*4 = 256.
constexpr uint32_t kEncodingMaxInputLen = 64;
constexpr uint32_t kEncodingMaxMungedLen = 256;

template <bool DOUBLE_HASH>
__global__ void v2_kernel_encoding_munge(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    size_t count,
    uint32_t encoding_mask,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t pp_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pp_idx >= count) return;

    const uint8_t* pw = passphrases + offsets[pp_idx];
    uint32_t pw_len = lengths[pp_idx];
    if (pw_len > kEncodingMaxInputLen) pw_len = kEncodingMaxInputLen;

    uint8_t munged[kEncodingMaxMungedLen];
    uint8_t hash[32];

    #pragma unroll
    for (uint8_t bit = 0; bit < DE_ENCODING_COUNT; ++bit) {
        if (!(encoding_mask & (1u << bit))) continue;
        int32_t mlen = munge((DevEncoding)bit, pw, pw_len, munged, sizeof(munged));
        if (mlen < 0) continue;  // input not representable in this encoding
        device::sha256(munged, (uint32_t)mlen, hash);
        if (DOUBLE_HASH) device::sha256(hash, 32, hash);

        v2_check_priv_against_puzzles(
            (uint32_t)pp_idx, (uint64_t)bit,
            bit, hash, matches, match_count);
    }
}

cudaError_t v2_encoding_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint32_t encoding_mask,
    bool double_hash,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (count == 0 || encoding_mask == 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    int blocks = (int)((count + BLOCK - 1) / BLOCK);
    if (double_hash) {
        v2_kernel_encoding_munge<true><<<blocks, BLOCK, 0, stream>>>(
            d_passphrases, d_offsets, d_lengths, count, encoding_mask,
            d_matches, d_match_count);
    } else {
        v2_kernel_encoding_munge<false><<<blocks, BLOCK, 0, stream>>>(
            d_passphrases, d_offsets, d_lengths, count, encoding_mask,
            d_matches, d_match_count);
    }
    return cudaGetLastError();
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
