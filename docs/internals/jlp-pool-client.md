# JLP pool client hardening notes

`jlp_pool_client.cpp` implements the JeanLucPons Kangaroo protocol on the
client side. The TLS / auth hardening summary below documents the
defense-in-depth measures applied across multiple security review passes.

## Hardening summary

| Area      | Change                                                         | Location                               |
| --------- | -------------------------------------------------------------- | -------------------------------------- |
| TLS       | TLS hostname verification + SNI + default trust store          | `init_tls()`                           |
| TLS       | `verify_cert` default flipped to `true`                        | header                                 |
| Auth      | `AuthState` machine gates work-affecting messages on `AUTH_OK` | `handle_server_message()`              |
| Reconnect | Bounded reconnect after `AUTH_FAIL` + jittered backoff         | `receiver_loop()`                      |
| Concur    | `ssl_write_mutex_` / `ssl_read_mutex_` split (concurrent r/w)  | `send_message()` / `receive_message()` |
| Auth      | `authenticate()` actually waits for `AUTH_OK` / `AUTH_FAIL`    | `authenticate()`                       |
| Threads   | `thread::operator=` safety on reconnect                        | `replace_thread()`                     |
| Types     | `pool_speed` type fixed to `uint64_t`                          | `handle_server_message()` STATS_RSP    |

## References

- RFC 6125 ("Representation and Verification of Domain-Based Application
  Service Identity within Internet PKI Using X.509 Certs")
- RFC 6066 sec 3 (Server Name Indication TLS extension)
- OpenSSL 1.1.1+ docs: `SSL_set_tlsext_host_name(3)`,
  `X509_VERIFY_PARAM_set1_host(3)`, `SSL_CTX_set_default_verify_paths(3)`
- C++23 [thread.thread.assign]/p2 (assigning to joinable thread terminates)
