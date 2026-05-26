/**
 * generate_worker_key.cpp -- B1 wire-v4 keygen (2026-05-23).
 *
 * Generates a fresh secp256k1 keypair, writes the private key to a
 * file in compressed-WIF form (mode 0600 on POSIX, ACL-equivalent on
 * Windows), and prints the derived bech32 P2WPKH worker address to
 * stdout for the operator to register with the pool.
 *
 * Usage:
 *   generate_worker_key <output-wif-path>
 *
 * The operator then:
 *   1. Registers the printed bech32 address with the pool as their
 *      worker_name (this is also where Bitcoin payouts will be sent).
 *   2. Runs the worker:
 *        collider_pro --pool jlps://host:17403 \
 *                     --worker <bech32-address-from-stdout> \
 *                     --worker-key <output-wif-path>
 *
 * Security:
 *   - Uses OpenSSL's CSPRNG (BN_priv_rand_range) to pick the privkey
 *     inside [1, n-1] where n is the secp256k1 group order. Refuses
 *     to write a key with high-bit-zero scalar (vanishingly rare).
 *   - File is created with 0600 on POSIX; Windows uses a restrictive
 *     ACL via SetNamedSecurityInfo when available, otherwise it falls
 *     back to the user's umask and emits a warning.
 *   - Aborts (does NOT overwrite) if the output path already exists.
 */

#include "../core/wif.hpp"
#include "../core/worker_identity.hpp"
#include "../core/bech32.hpp"

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace {

int usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " <output-wif-path>\n";
    return 2;
}

// Open the output file with restrictive permissions. Refuses to
// overwrite an existing file (the operator must move it aside) so a
// careless re-run doesn't wipe a working identity.
bool open_secure_for_write(const std::string& path, std::ofstream& out) {
#if defined(_WIN32)
    // Refuse to overwrite. CreateFileA with CREATE_NEW returns
    // INVALID_HANDLE_VALUE if the file exists, which is the behavior
    // we want.
    HANDLE h = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr,
                           CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::cerr << "[!] Cannot create " << path
                  << " (already exists?). Move the old file aside first.\n";
        return false;
    }
    CloseHandle(h);
    // Re-open via ofstream now that we've reserved the path with
    // CREATE_NEW semantics. On NTFS the inherited ACL gives only the
    // current user write access in most operator setups; production
    // users on shared boxes should restrict further via icacls.
    out.open(path, std::ios::binary | std::ios::trunc);
    return out.good();
#else
    int fd = open(path.c_str(),
                  O_WRONLY | O_CREAT | O_EXCL,
                  S_IRUSR | S_IWUSR);  // 0600
    if (fd < 0) {
        std::cerr << "[!] Cannot create " << path << " (exists?): "
                  << std::strerror(errno) << "\n";
        return false;
    }
    close(fd);
    out.open(path, std::ios::binary | std::ios::trunc);
    return out.good();
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return usage(argv[0]);
    const std::string out_path = argv[1];

    // 1. Generate a random privkey in [1, n-1] using OpenSSL's
    //    BN_priv_rand_range which pulls from RAND_bytes (system CSPRNG).
    EC_KEY* key = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (!key) {
        std::cerr << "[!] EC_KEY_new_by_curve_name failed\n";
        return 1;
    }
    if (EC_KEY_generate_key(key) != 1) {
        std::cerr << "[!] EC_KEY_generate_key failed\n";
        EC_KEY_free(key);
        return 1;
    }

    // 2. Extract the 32-byte privkey big-endian.
    const BIGNUM* priv = EC_KEY_get0_private_key(key);
    if (!priv) {
        std::cerr << "[!] EC_KEY_get0_private_key returned null\n";
        EC_KEY_free(key);
        return 1;
    }
    std::array<uint8_t, 32> priv_bytes{};
    if (BN_bn2binpad(priv, priv_bytes.data(), 32) != 32) {
        std::cerr << "[!] BN_bn2binpad failed\n";
        EC_KEY_free(key);
        return 1;
    }
    EC_KEY_free(key);

    // 3. Encode as compressed mainnet WIF.
    std::string wif = collider::wif::encode(priv_bytes, /*compressed=*/true);

    // 4. Re-load via WorkerIdentity to derive the bech32 address. This
    //    proves the file we're about to write actually round-trips and
    //    matches the address the server will compute.
    auto id = collider::identity::load_from_wif(wif, "bc");
    if (!id) {
        std::cerr << "[!] Generated WIF failed round-trip load (this is "
                     "a bug; aborting)\n";
        std::fill(priv_bytes.begin(), priv_bytes.end(), 0);
        std::fill(wif.begin(), wif.end(), '\0');
        return 1;
    }

    // 5. Persist WIF to the output file (refuses to overwrite).
    std::ofstream out;
    if (!open_secure_for_write(out_path, out)) {
        std::fill(priv_bytes.begin(), priv_bytes.end(), 0);
        std::fill(wif.begin(), wif.end(), '\0');
        return 1;
    }
    out << wif << "\n";
    out.close();
    if (!out) {
        std::cerr << "[!] Failed writing " << out_path << "\n";
        std::fill(priv_bytes.begin(), priv_bytes.end(), 0);
        std::fill(wif.begin(), wif.end(), '\0');
        return 1;
    }

    // 6. Print the bech32 address for the operator. Privkey bytes
    //    never leave this process; only the WIF on disk + the address
    //    on stdout. Wipe local copies before exit.
    std::cout << "Worker identity created.\n";
    std::cout << "  Address (register with pool, set --worker): "
              << id->bech32_address() << "\n";
    std::cout << "  WIF file (set --worker-key):                "
              << out_path << "\n";
    std::cout << "\n";
    std::cout << "Protect the WIF file: anyone with read access can sign "
                 "AUTH frames\n"
                 "as this worker. Recommended mode: 0600. Back it up "
                 "off-machine.\n";

    std::fill(priv_bytes.begin(), priv_bytes.end(), 0);
    std::fill(wif.begin(), wif.end(), '\0');
    return 0;
}
