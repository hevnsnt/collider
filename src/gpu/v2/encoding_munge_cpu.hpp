/**
 * Encoding anomaly mutations -- CPU reference (Phase 6, v1.4.0).
 *
 * Many real-world brain-wallet generators encode the user's passphrase
 * as UTF-16-LE / UTF-32-LE before SHA-256, due to language defaults
 * (Java's String.getBytes() default encoding, .NET's Encoding.Unicode,
 * etc.). Some web-tool implementations use Latin-1 fallbacks or strip
 * non-ASCII. The honest crack of those wallets requires running the
 * passphrase through every plausible encoding before deriving priv.
 *
 * This header exposes the byte-transform half. The caller chains:
 *   pw -> munge(pw) -> SHA-256(munged) -> priv
 *
 * The eventual GPU kernel will inline these transforms and run the
 * existing v2 multi-scheme + multi-address path on each variant.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {
namespace encmunge {

enum class Encoding : uint8_t {
    UTF8                = 0,    // identity (most JavaScript / Linux)
    UTF16_LE            = 1,    // .NET, Windows API default; null after each char
    UTF16_BE            = 2,    // Java getBytes() with no charset arg, big-endian platforms
    UTF32_LE            = 3,    // rare but seen in some web-tool outputs
    UTF32_BE            = 4,
    LATIN1              = 5,    // 8-bit truncate of code points; dies on >0xFF
    STRIP_NON_ASCII     = 6,    // drop bytes >= 0x80 entirely
    UPPER_ASCII         = 7,    // toupper on a..z
    LOWER_ASCII         = 8,    // tolower on A..Z
    ENCODING_COUNT      = 9,
};

// ---------------------------------------------------------------------------
// Helpers: simple UTF-8 -> codepoint walker.
// Returns codepoint and advances pos; returns -1 on malformed bytes (caller
// should treat as passphrase rejected for that encoding).
// ---------------------------------------------------------------------------

inline int32_t utf8_decode_one(const uint8_t* in, size_t len, size_t& pos) {
    if (pos >= len) return -1;
    uint8_t c0 = in[pos];
    if (c0 < 0x80) { ++pos; return c0; }
    if ((c0 & 0xE0) == 0xC0) {
        if (pos + 1 >= len) return -1;
        uint8_t c1 = in[pos + 1];
        if ((c1 & 0xC0) != 0x80) return -1;
        pos += 2;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0) {
        if (pos + 2 >= len) return -1;
        uint8_t c1 = in[pos + 1], c2 = in[pos + 2];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return -1;
        pos += 3;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0) {
        if (pos + 3 >= len) return -1;
        uint8_t c1 = in[pos + 1], c2 = in[pos + 2], c3 = in[pos + 3];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return -1;
        pos += 4;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12)
             | ((c2 & 0x3F) << 6)  |  (c3 & 0x3F);
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Munge: produces the byte sequence under the chosen encoding.
//
// Returns false if the input is unrepresentable in `enc` (e.g. emoji
// in LATIN1; the corresponding GPU kernel will short-circuit such
// inputs without running SHA-256).
// ---------------------------------------------------------------------------

inline bool munge(Encoding enc, const uint8_t* in, size_t in_len,
                  std::vector<uint8_t>& out)
{
    out.clear();
    switch (enc) {
        case Encoding::UTF8:
            out.assign(in, in + in_len);
            return true;

        case Encoding::UTF16_LE: {
            size_t p = 0;
            while (p < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, p);
                if (cp < 0) return false;
                if (cp <= 0xFFFF) {
                    out.push_back(static_cast<uint8_t>(cp & 0xFF));
                    out.push_back(static_cast<uint8_t>((cp >> 8) & 0xFF));
                } else {
                    // Surrogate pair
                    uint32_t v = static_cast<uint32_t>(cp) - 0x10000;
                    uint32_t hi = 0xD800 | (v >> 10);
                    uint32_t lo = 0xDC00 | (v & 0x3FF);
                    out.push_back(static_cast<uint8_t>(hi & 0xFF));
                    out.push_back(static_cast<uint8_t>((hi >> 8) & 0xFF));
                    out.push_back(static_cast<uint8_t>(lo & 0xFF));
                    out.push_back(static_cast<uint8_t>((lo >> 8) & 0xFF));
                }
            }
            return true;
        }

        case Encoding::UTF16_BE: {
            size_t p = 0;
            while (p < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, p);
                if (cp < 0) return false;
                if (cp <= 0xFFFF) {
                    out.push_back(static_cast<uint8_t>((cp >> 8) & 0xFF));
                    out.push_back(static_cast<uint8_t>(cp & 0xFF));
                } else {
                    uint32_t v = static_cast<uint32_t>(cp) - 0x10000;
                    uint32_t hi = 0xD800 | (v >> 10);
                    uint32_t lo = 0xDC00 | (v & 0x3FF);
                    out.push_back(static_cast<uint8_t>((hi >> 8) & 0xFF));
                    out.push_back(static_cast<uint8_t>(hi & 0xFF));
                    out.push_back(static_cast<uint8_t>((lo >> 8) & 0xFF));
                    out.push_back(static_cast<uint8_t>(lo & 0xFF));
                }
            }
            return true;
        }

        case Encoding::UTF32_LE: {
            size_t p = 0;
            while (p < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, p);
                if (cp < 0) return false;
                uint32_t v = static_cast<uint32_t>(cp);
                out.push_back(static_cast<uint8_t>(v & 0xFF));
                out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
                out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
                out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
            }
            return true;
        }

        case Encoding::UTF32_BE: {
            size_t p = 0;
            while (p < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, p);
                if (cp < 0) return false;
                uint32_t v = static_cast<uint32_t>(cp);
                out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
                out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
                out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
                out.push_back(static_cast<uint8_t>(v & 0xFF));
            }
            return true;
        }

        case Encoding::LATIN1: {
            size_t p = 0;
            while (p < in_len) {
                int32_t cp = utf8_decode_one(in, in_len, p);
                if (cp < 0 || cp > 0xFF) return false;
                out.push_back(static_cast<uint8_t>(cp));
            }
            return true;
        }

        case Encoding::STRIP_NON_ASCII: {
            for (size_t i = 0; i < in_len; ++i) {
                if (in[i] < 0x80) out.push_back(in[i]);
            }
            return true;
        }

        case Encoding::UPPER_ASCII: {
            out.reserve(in_len);
            for (size_t i = 0; i < in_len; ++i) {
                uint8_t b = in[i];
                if (b >= 'a' && b <= 'z') b -= 0x20;
                out.push_back(b);
            }
            return true;
        }

        case Encoding::LOWER_ASCII: {
            out.reserve(in_len);
            for (size_t i = 0; i < in_len; ++i) {
                uint8_t b = in[i];
                if (b >= 'A' && b <= 'Z') b += 0x20;
                out.push_back(b);
            }
            return true;
        }

        case Encoding::ENCODING_COUNT:
            return false;
    }
    return false;
}

}  // namespace encmunge
}  // namespace v2
}  // namespace gpu
}  // namespace collider
