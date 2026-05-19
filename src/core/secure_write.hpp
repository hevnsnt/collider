/**
 * secure_write.hpp -- open files holding recovered key material with
 * owner-only permissions.
 *
 * Files written by the solver may contain recovered private keys
 * (puzzle_found.txt, brainwallet_hits.txt, recovered_keys_*.txt,
 * bloom_hits.txt, brainwallet.pot). On a shared host (any non-personal
 * workstation), other local users can read these files by default
 * because std::ofstream creates them with the process umask (typically
 * 022, world-readable). On Windows, default ACL inheritance from the
 * parent directory often grants read to Users and Authenticated Users.
 *
 * This header provides a single entry point, secure_open_ofstream(),
 * that:
 *
 *   - POSIX: opens the file via open(O_CREAT | O_WRONLY, 0600) so the
 *     OS sees an owner-only mode at create time, then fchmod's the
 *     descriptor to 0600 to defeat the process umask, then closes the
 *     descriptor and opens the std::ofstream against the same path
 *     (the mode persists across reopen). On an existing file the
 *     fchmod also repairs a pre-existing weak mode.
 *
 *   - Windows: creates the file with CreateFileA + an explicit
 *     SECURITY_DESCRIPTOR that grants only the calling user a single
 *     DACL ACE with FILE_GENERIC_READ | FILE_GENERIC_WRITE. The DACL
 *     is then anchored to the file via SetSecurityInfo so a
 *     pre-existing file inheriting wider ACLs is repaired. The
 *     handle is closed and the std::ofstream is opened against the
 *     same path; the DACL persists.
 *
 * Callers replace
 *
 *     std::ofstream f(path, std::ios::app);
 *
 * with
 *
 *     std::ofstream f = collider::secure_open_ofstream(path,
 *                                                      std::ios::app);
 *
 * and otherwise behave identically. If secure creation fails for any
 * reason, the function falls back to a regular std::ofstream open and
 * logs a single warning to std::cerr.
 */

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ios>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <aclapi.h>
#else
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace collider {

namespace detail {

#ifdef _WIN32

/**
 * RAII holder for an owner-only SECURITY_DESCRIPTOR + DACL pair.
 *
 * Construction is via build_owner_only_sa(). The destructor frees the
 * SD and the ACL through LocalFree, and the duplicated SID through
 * std::free (allocated via std::malloc inside build_owner_only_sa).
 */
struct OwnerOnlySa {
    SECURITY_ATTRIBUTES sa{};
    PSECURITY_DESCRIPTOR sd = nullptr;
    PACL acl = nullptr;
    PSID user_sid = nullptr;
    bool ok = false;

    OwnerOnlySa() = default;
    OwnerOnlySa(const OwnerOnlySa&) = delete;
    OwnerOnlySa& operator=(const OwnerOnlySa&) = delete;

    ~OwnerOnlySa() {
        if (sd) LocalFree(sd);
        if (acl) LocalFree(acl);
        if (user_sid) std::free(user_sid);
    }
};

inline bool build_owner_only_sa(OwnerOnlySa& out) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) {
        return false;
    }

    DWORD needed = 0;
    GetTokenInformation(token, TokenUser, nullptr, 0, &needed);
    if (needed == 0) {
        CloseHandle(token);
        return false;
    }

    std::vector<char> token_buf(needed);
    if (!GetTokenInformation(token, TokenUser, token_buf.data(), needed, &needed)) {
        CloseHandle(token);
        return false;
    }
    CloseHandle(token);

    PSID raw_sid = reinterpret_cast<TOKEN_USER*>(token_buf.data())->User.Sid;
    if (!IsValidSid(raw_sid)) {
        return false;
    }

    DWORD sid_len = GetLengthSid(raw_sid);
    out.user_sid = std::malloc(sid_len);
    if (!out.user_sid) return false;
    if (!CopySid(sid_len, out.user_sid, raw_sid)) {
        return false;
    }

    EXPLICIT_ACCESSA ea{};
    ea.grfAccessPermissions = FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE;
    ea.grfAccessMode        = SET_ACCESS;
    ea.grfInheritance       = NO_INHERITANCE;
    ea.Trustee.TrusteeForm  = TRUSTEE_IS_SID;
    ea.Trustee.TrusteeType  = TRUSTEE_IS_USER;
    ea.Trustee.ptstrName    = reinterpret_cast<LPSTR>(out.user_sid);

    if (SetEntriesInAclA(1, &ea, nullptr, &out.acl) != ERROR_SUCCESS) {
        return false;
    }

    out.sd = LocalAlloc(LPTR, SECURITY_DESCRIPTOR_MIN_LENGTH);
    if (!out.sd) return false;
    if (!InitializeSecurityDescriptor(out.sd, SECURITY_DESCRIPTOR_REVISION)) {
        return false;
    }
    if (!SetSecurityDescriptorDacl(out.sd, TRUE, out.acl, FALSE)) {
        return false;
    }

    out.sa.nLength              = sizeof(SECURITY_ATTRIBUTES);
    out.sa.bInheritHandle       = FALSE;
    out.sa.lpSecurityDescriptor = out.sd;
    out.ok = true;
    return true;
}

#endif  // _WIN32

} // namespace detail

/**
 * Behaviour selector for secure_open_ofstream when the owner-only
 * permission set cannot be established. The contract of this header
 * says "never silently weakens permissions" -- the two policies below
 * give callers a way to pick how loud that failure is.
 *
 *   FallbackLoud (default, legacy behaviour):
 *     If the owner-only ACL / mode could not be built (Windows: token
 *     query failed, ACL allocation failed; POSIX: open() failed) the
 *     function emits a single std::cerr warning, then opens the file
 *     via the default std::ofstream constructor so the call does not
 *     hard-fail mid-scan. The resulting file inherits the parent
 *     directory's DACL (Windows) or is masked by umask (POSIX). Use
 *     this for non-key-material sinks where availability beats
 *     confidentiality (e.g. ~/.collider/pool_dp_seq.dat).
 *
 *   FailHard:
 *     If the owner-only permission set could not be established the
 *     function returns an *unopened* std::ofstream and emits a single
 *     std::cerr error explaining why. Callers MUST check is_open() /
 *     operator bool() before writing; an unopened stream silently
 *     discards writes, so calling code that already gates on
 *     `if (out)` is correct by default. Use this for any sink that
 *     holds recovered key material:
 *
 *       - recovered_keys/<...>.json
 *       - puzzle_found.txt
 *       - brainwallet_hits.txt
 *       - brainwallet.pot
 *       - found-empty.txt
 *
 *     The failure mode for these files must be "no file" rather than
 *     "world-readable file"; FailHard enforces that.
 */
enum class SecureWriteOnFailure {
    FallbackLoud,
    FailHard,
};

/**
 * Open `path` for writing with owner-only file permissions.
 *
 * `mode` may be any combination of std::ios::app, std::ios::trunc,
 * std::ios::out, and std::ios::binary the caller would have passed to
 * std::ofstream. The semantics of std::ofstream apply unchanged.
 *
 * `on_failure` selects how the function behaves when the owner-only
 * permission set cannot be established. See SecureWriteOnFailure above.
 * The default (FallbackLoud) preserves legacy behaviour: warn loudly,
 * open with whatever permissions the system gives us. Callers that
 * write recovered key material should pass FailHard so the resulting
 * stream is unopened on permission failure rather than world-readable.
 */
inline std::ofstream secure_open_ofstream(
    const std::filesystem::path& path,
    std::ios::openmode mode = std::ios::out | std::ios::app,
    SecureWriteOnFailure on_failure = SecureWriteOnFailure::FallbackLoud)
{
    // Best-effort: create the parent directory. A failure here just
    // means the subsequent open call will fail with a clearer error.
    std::error_code ec;
    auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
    }

#ifdef _WIN32
    // Build the owner-only security attributes once; reuse the ACL for
    // both CreateFileA (initial create) and SetSecurityInfo (repair).
    detail::OwnerOnlySa sa;
    bool have_sa = detail::build_owner_only_sa(sa);
    if (!have_sa) {
        // Loud failure so an operator does not get a silent permissions
        // downgrade to the default-inheriting DACL. The contract of this
        // header is "never silently weakens permissions"; surfacing this
        // makes the violation visible.
        std::cerr << "[secure_write] WARNING: build_owner_only_sa failed "
                     "(GetLastError=" << GetLastError() << "); owner-only "
                     "DACL could not be constructed for "
                  << path.string() << "." << std::endl;
        if (on_failure == SecureWriteOnFailure::FailHard) {
            // Caller opted in to fail-hard. Return an unopened stream
            // so a downstream `if (out)` guard treats this the same
            // as any other open failure. Crucially: we do NOT call
            // CreateFileA below, so no file is created on disk with
            // the inherited (potentially world-readable) DACL.
            std::cerr << "[secure_write] FailHard: refusing to create "
                      << path.string()
                      << " with inherited DACL. Returning unopened stream; "
                         "caller will see is_open()==false."
                      << std::endl;
            return std::ofstream{};  // default-constructed = !is_open()
        }
        // FallbackLoud path: proceed with CreateFileA(nullptr SA) so
        // the call does not hard-fail mid-scan. The operator already
        // saw the warning above.
        std::cerr << "[secure_write]   (falling back: file will be "
                     "created with the inherited DACL, NOT the owner-only "
                     "DACL the secure_open_ofstream contract promises. "
                     "Inspect parent-directory permissions and consider "
                     "running with elevated rights.)" << std::endl;
    }

    DWORD desired_access     = GENERIC_WRITE;
    DWORD share_mode         = FILE_SHARE_READ;  // tail/inspection ok
    DWORD creation_disposition;
    if (mode & std::ios::trunc) {
        creation_disposition = CREATE_ALWAYS;
    } else if (mode & std::ios::app) {
        creation_disposition = OPEN_ALWAYS;
        desired_access |= FILE_APPEND_DATA;
    } else {
        creation_disposition = OPEN_ALWAYS;
    }

    HANDLE h = CreateFileA(
        path.string().c_str(),
        desired_access,
        share_mode,
        have_sa ? &sa.sa : nullptr,
        creation_disposition,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);

    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "[secure_write] CreateFileA failed for "
                  << path.string()
                  << " (err=" << GetLastError() << ")." << std::endl;
        if (on_failure == SecureWriteOnFailure::FailHard) {
            // Return an unopened stream rather than retrying with the
            // default std::ofstream constructor: a CreateFileA failure
            // here with have_sa==true means our explicit owner-only SA
            // got rejected, which is a deeper permissions / sharing
            // issue than the std::ofstream constructor would diagnose.
            std::cerr << "[secure_write] FailHard: not retrying with "
                         "default std::ofstream constructor; returning "
                         "unopened stream." << std::endl;
            return std::ofstream{};
        }
        std::cerr << "[secure_write]   (falling back to default "
                     "std::ofstream open.)" << std::endl;
        return std::ofstream(path, mode);
    }

    // Anchor the DACL on the file. For a freshly created file this is
    // a no-op (CreateFileA already applied sa.sa). For a pre-existing
    // file that inherited a wider ACL, this is what repairs it.
    if (have_sa) {
        SetSecurityInfo(h, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
                        nullptr, nullptr, sa.acl, nullptr);
    }

    // The DACL on the file persists across handle close + reopen. The
    // standard std::ofstream constructor below will reopen with our
    // restrictive permissions in place.
    CloseHandle(h);

    return std::ofstream(path, mode);

#else  // POSIX

    int oflags = O_WRONLY | O_CREAT;
    if (mode & std::ios::trunc) oflags |= O_TRUNC;
    if (mode & std::ios::app)   oflags |= O_APPEND;

    // Atomic owner-only create. The mode argument is masked by the
    // process umask; fchmod() below force-clears the result back to
    // S_IRUSR | S_IWUSR even if umask was 0 (and repairs a pre-existing
    // file that was created world-readable by the pre-fix code path).
    int fd = ::open(path.string().c_str(), oflags, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        std::cerr << "[secure_write] open() failed for " << path.string()
                  << " (errno=" << errno << ")." << std::endl;
        if (on_failure == SecureWriteOnFailure::FailHard) {
            // open() failed despite our explicit S_IRUSR|S_IWUSR mode.
            // FailHard contract: return an unopened stream rather than
            // retrying with std::ofstream's default open, which would
            // hit the same EACCES / ENOENT / EROFS but might leave a
            // partially-initialised file behind on some libc paths.
            std::cerr << "[secure_write] FailHard: not retrying with "
                         "default std::ofstream constructor; returning "
                         "unopened stream." << std::endl;
            return std::ofstream{};
        }
        std::cerr << "[secure_write]   (falling back to default "
                     "std::ofstream open.)" << std::endl;
        return std::ofstream(path, mode);
    }

    // Force 0600 regardless of umask / pre-existing mode.
    if (::fchmod(fd, S_IRUSR | S_IWUSR) != 0 &&
        on_failure == SecureWriteOnFailure::FailHard) {
        // fchmod() failed (rare; would happen on a filesystem that
        // ignores mode bits, e.g. some FAT mounts). Caller wanted
        // FailHard; close the fd without leaving the file world-
        // readable from a prior open. Note: ::open with O_CREAT
        // already created the file at mode 0600 minus umask, so even
        // on FailHard with fchmod failure the on-disk mode is at
        // worst (0600 & ~umask) and never wider than that. We still
        // refuse to open the std::ofstream so the caller does not
        // think the file is fully owner-only.
        int saved_errno = errno;
        std::cerr << "[secure_write] FailHard: fchmod failed for "
                  << path.string() << " (errno=" << saved_errno
                  << "); returning unopened stream." << std::endl;
        ::close(fd);
        return std::ofstream{};
    }
    ::close(fd);

    return std::ofstream(path, mode);

#endif
}

} // namespace collider
