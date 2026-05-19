/**
 * secure_buffer.hpp -- RAII buffer that zeroes its storage on destruction.
 *
 * Why this exists: private-key bytes, HMAC keys, and pool passwords that
 * sit on the heap after use survive into core dumps, swap files, and
 * (on Windows) the page file. OPENSSL_cleanse is a compiler-barrier
 * wipe that the compiler cannot dead-store-eliminate. Wrapping the
 * lifetime of sensitive bytes in a SecureBuffer makes the zero-on-free
 * contract a property of the type, not of "remember to memset" in the
 * call site.
 *
 * Two concrete types:
 *   - SecureBuffer<T>   : owns a heap-allocated buffer of N T's. Wipes
 *                         the full byte range on destruction. Move-only.
 *   - SecureString      : same idea for std::string-like bytes (8-bit
 *                         characters). Useful for credentials whose
 *                         length is not known at compile time.
 *
 * Both types deliberately disable copy. A copied secret is a leaked
 * secret. Use std::move when ownership must transfer.
 */

#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/crypto.h>
#endif

namespace collider {

// Volatile-store wipe. The volatile pointer prevents the compiler from
// eliding the writes as dead stores. On a 64-bit MSVC release build this
// emits the same code (or a no-op DCE plus an inline rep stosb) as
// OPENSSL_cleanse modulo the explicit memory barrier OpenSSL adds; in
// practice neither MSVC nor clang-cl have ever observed to dead-store-
// eliminate this loop.
//
// Why not OPENSSL_cleanse? It is strictly stronger against LTO (it ships
// with an explicit memory barrier symbol the linker cannot fold), and an
// earlier iteration of this file did call it. But on Windows the OpenSSL
// DLL is unloaded by the loader at process shutdown BEFORE the static
// SecureBuffer destructors fire (the loader walks dependencies bottom-up;
// our DLLs depend on OpenSSL so OpenSSL goes first). Any SecureBuffer
// owned by a static or thread_local then segfaults dereferencing an
// unloaded function pointer. The volatile-loop has the same effective
// guarantee in our release build matrix and is safe to invoke from any
// destructor ordering.
//
// If you switch this to OPENSSL_cleanse: also wrap the call in a guard
// that detects DLL teardown (e.g. via std::atexit ordering) or accept
// that any static SecureBuffer becomes a process-exit AV on Windows.
inline void secure_wipe(void* p, size_t n) noexcept {
    if (!p || n == 0) return;
    volatile unsigned char* vp = reinterpret_cast<volatile unsigned char*>(p);
    while (n--) {
        *vp++ = 0;
    }
}

/**
 * SecureBuffer<T> -- fixed-size RAII byte buffer that wipes on destruction.
 *
 * Trivially-copyable T only (the type wipes raw bytes, so non-trivial
 * destructors on T would be skipped). The intended use is uint8_t and
 * compatible POD types holding key material.
 */
template <typename T>
class SecureBuffer {
    static_assert(sizeof(T) >= 1, "SecureBuffer<T> requires sized T");

    T* data_ = nullptr;
    size_t size_ = 0;

public:
    SecureBuffer() = default;

    explicit SecureBuffer(size_t n) {
        if (n > 0) {
            data_ = new T[n]();   // value-init (zero-fill) on construction
            size_ = n;
        }
    }

    ~SecureBuffer() {
        wipe();
        delete[] data_;
    }

    // Move-only. Copying a buffer of key material is exactly the leak
    // this type is meant to prevent; force the caller to be explicit.
    SecureBuffer(const SecureBuffer&) = delete;
    SecureBuffer& operator=(const SecureBuffer&) = delete;

    SecureBuffer(SecureBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    SecureBuffer& operator=(SecureBuffer&& other) noexcept {
        if (this != &other) {
            wipe();
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t   size() const noexcept { return size_; }
    bool     empty() const noexcept { return size_ == 0; }

    T&       operator[](size_t i)       noexcept { return data_[i]; }
    const T& operator[](size_t i) const noexcept { return data_[i]; }

    /**
     * Wipe the contents in place without releasing the allocation.
     * Useful when the buffer is reused for a sequence of secrets and the
     * caller wants the bytes gone immediately rather than at destructor
     * time. Idempotent; safe to call on an empty buffer.
     */
    void wipe() noexcept {
        if (data_ && size_ > 0) {
            secure_wipe(data_, size_ * sizeof(T));
        }
    }
};

/**
 * SecureString -- owning byte sequence with zero-on-free semantics.
 *
 * Exists because std::string makes no guarantee about wiping its
 * storage on destruction (and SSO can put the bytes inside the
 * std::string object itself, which is then memcpy'd around with no
 * way to wipe the source). For pool passwords and similar credentials
 * we accept the cost of a separate heap allocation and explicit wipe.
 *
 * The contained bytes are NOT NUL-terminated by default. Call
 * c_str_unsafe() only when you must hand the pointer to a C API that
 * truly needs a NUL terminator; the +1 byte is allocated and zeroed
 * so the cast is safe.
 */
class SecureString {
    char* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;  // includes the +1 NUL slot

public:
    SecureString() = default;

    SecureString(const char* src, size_t n) {
        assign(src, n);
    }

    explicit SecureString(const char* src) {
        assign(src, src ? std::strlen(src) : 0);
    }

    ~SecureString() {
        wipe();
        delete[] data_;
    }

    SecureString(const SecureString&) = delete;
    SecureString& operator=(const SecureString&) = delete;

    SecureString(SecureString&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    SecureString& operator=(SecureString&& other) noexcept {
        if (this != &other) {
            wipe();
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void assign(const char* src, size_t n) {
        wipe();
        if (capacity_ < n + 1) {
            delete[] data_;
            capacity_ = n + 1;
            data_ = new char[capacity_]();
        } else {
            std::memset(data_, 0, capacity_);
        }
        if (src && n > 0) {
            std::memcpy(data_, src, n);
        }
        size_ = n;
        if (data_) data_[size_] = '\0';
    }

    const char* data() const noexcept { return data_; }
    char*       data()       noexcept { return data_; }
    size_t      size() const noexcept { return size_; }
    bool        empty() const noexcept { return size_ == 0; }

    // For callsites that absolutely require a C string. The buffer holds
    // a NUL byte beyond size_, so this is well-formed.
    const char* c_str_unsafe() const noexcept {
        return data_ ? data_ : "";
    }

    void wipe() noexcept {
        if (data_ && capacity_ > 0) {
            secure_wipe(data_, capacity_);
        }
        size_ = 0;
    }
};

} // namespace collider
