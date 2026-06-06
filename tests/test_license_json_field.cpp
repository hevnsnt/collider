/**
 * Direct adversarial tests for the hardened top-level JSON field extractor
 * collider::license::detail::json_field (adversarial review v1.5.5, finding
 * R6 / the earlier LOW fix).
 *
 * json_field gates the --activate path: the client reads `"valid":true`,
 * `"email"`, and the signed-assertion fields out of the license-check
 * response with it. The previous implementation was a naive substring scan
 * (`response.find("\"valid\":")`) that false-accepted a `"valid":true`
 * occurrence ANYWHERE in the body, including buried inside an attacker
 * controlled error-message string or a nested sub-object. The hardened
 * version is a depth/string/escape-aware scanner that matches a key only at
 * depth 1 of the outermost object. These tests pin that behavior with the
 * exact adversarial inputs the fix was made for; until now it was verified
 * only by inspection.
 *
 * This is a Pro-only surface (license verification is Pro). On Free builds the
 * test compiles to a skip (exit 77) so ctest records it as SKIPPED.
 */

#ifndef COLLIDER_PRO
int main() { return 77; }   // skip on Free builds
#else

#include "license/license_check.hpp"

#include <cstdio>
#include <string>

using collider::license::detail::json_field;

namespace {

int g_failures = 0;

void check_eq(const std::string& got, const std::string& want,
              const char* what) {
    if (got != want) {
        std::fprintf(stderr,
                     "FAIL: %s\n  got  = \"%s\"\n  want = \"%s\"\n",
                     what, got.c_str(), want.c_str());
        ++g_failures;
    }
}

}  // namespace

int main() {
    std::printf("=== json_field adversarial tests ===\n");

    // 1. The attacker controls an error-message STRING that literally contains
    //    the bytes `"valid":true`. Because that text lives inside a string
    //    value (not at object depth 1 as a key), the top-level key "valid"
    //    MUST NOT be found. A naive substring scan would false-accept here.
    //    Wire JSON: {"error":"\"valid\":true"}
    check_eq(json_field("{\"error\":\"\\\"valid\\\":true\"}", "valid"), "",
             "\"valid\" buried in an error-message string is not matched");

    // 2. A NESTED sub-object has valid:true, but the real TOP-LEVEL valid is
    //    false. Only the top-level field counts.
    //    Wire JSON: {"x":{"valid":true},"valid":false}
    check_eq(json_field("{\"x\":{\"valid\":true},\"valid\":false}", "valid"),
             "false",
             "nested valid:true ignored; top-level valid:false wins");

    // 3. A top-level ARRAY whose element object has valid:true. There is no
    //    top-level OBJECT key, so the field is absent (fail-safe "").
    //    Wire JSON: [{"valid":true}]
    check_eq(json_field("[{\"valid\":true}]", "valid"), "",
             "top-level array yields no top-level field match");

    // 4. Escaped quotes inside a string value spell out `"valid":true`. The
    //    backslash-escaped quotes must NOT terminate the string early, so the
    //    embedded text is not parsed as a key. Field stays absent.
    //    Wire JSON: {"a":"\"valid\":true"}
    check_eq(json_field("{\"a\":\"\\\"valid\\\":true\"}", "valid"), "",
             "escaped quotes do not split the string into a fake key");

    // 5. An UNTERMINATED string must not read out of bounds or crash; it
    //    simply ends the scan with no match. (The CHECK is that we return
    //    cleanly; reaching this line means no crash occurred.)
    //    Wire JSON (truncated): {"email":"a@b
    check_eq(json_field("{\"email\":\"a@b", "email"), "a@b",
             "unterminated string yields the bytes read so far, no crash");
    //    A key whose name itself is unterminated cannot match either.
    //    Wire JSON (truncated): {"vali
    check_eq(json_field("{\"vali", "valid"), "",
             "unterminated key does not crash and does not match");

    // 6. A normal, well-formed response: both a bool token and a string field
    //    are extracted correctly.
    //    Wire JSON: {"valid":true,"email":"a@b"}
    check_eq(json_field("{\"valid\":true,\"email\":\"a@b\"}", "valid"), "true",
             "normal top-level valid:true extracted as \"true\"");
    check_eq(json_field("{\"valid\":true,\"email\":\"a@b\"}", "email"), "a@b",
             "normal top-level email string extracted");

    // Extra guards that lock the depth-1 contract from both sides.
    // 7. Whitespace around the colon/value is tolerated.
    check_eq(json_field("{ \"valid\" : true }", "valid"), "true",
             "whitespace around top-level value tolerated");
    // 8. A key that only appears as a NESTED key is absent at top level.
    check_eq(json_field("{\"data\":{\"email\":\"x@y\"}}", "email"), "",
             "nested-only key is absent at top level");

    if (g_failures == 0) {
        std::printf("all json_field tests passed\n");
        return 0;
    }
    std::printf("%d json_field test(s) FAILED\n", g_failures);
    return 1;
}

#endif  // COLLIDER_PRO
