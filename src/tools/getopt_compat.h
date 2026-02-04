/**
 * Minimal getopt implementation for Windows compatibility.
 * On POSIX systems, this just includes <getopt.h>.
 */

#pragma once

#ifdef _WIN32

#include <string.h>

extern char* optarg;
extern int optind;
extern int opterr;
extern int optopt;

struct option {
    const char* name;
    int has_arg;
    int* flag;
    int val;
};

#define no_argument 0
#define required_argument 1
#define optional_argument 2

inline char* optarg = nullptr;
inline int optind = 1;
inline int opterr = 1;
inline int optopt = '?';

inline int getopt_long(int argc, char* const argv[], const char* optstring,
                       const struct option* longopts, int* longindex) {
    static int pos = 1;

    if (optind >= argc || argv[optind] == nullptr) {
        return -1;
    }

    const char* arg = argv[optind];

    // Check for long option
    if (arg[0] == '-' && arg[1] == '-') {
        const char* opt_name = arg + 2;

        // Check for end of options
        if (*opt_name == '\0') {
            optind++;
            return -1;
        }

        // Find the option
        for (int i = 0; longopts[i].name != nullptr; i++) {
            size_t len = strlen(longopts[i].name);
            if (strncmp(opt_name, longopts[i].name, len) == 0) {
                if (opt_name[len] == '\0' || opt_name[len] == '=') {
                    if (longindex) *longindex = i;

                    if (longopts[i].has_arg == required_argument) {
                        if (opt_name[len] == '=') {
                            optarg = const_cast<char*>(opt_name + len + 1);
                        } else if (optind + 1 < argc) {
                            optarg = argv[++optind];
                        } else {
                            return '?';
                        }
                    }

                    optind++;
                    return longopts[i].val;
                }
            }
        }
        optind++;
        return '?';
    }

    // Check for short option
    if (arg[0] == '-' && arg[1] != '\0') {
        char c = arg[pos];
        const char* p = strchr(optstring, c);

        if (p == nullptr) {
            optopt = c;
            if (arg[pos + 1] == '\0') {
                optind++;
                pos = 1;
            } else {
                pos++;
            }
            return '?';
        }

        if (p[1] == ':') {
            // Requires argument
            if (arg[pos + 1] != '\0') {
                optarg = const_cast<char*>(arg + pos + 1);
                optind++;
                pos = 1;
            } else if (optind + 1 < argc) {
                optarg = argv[++optind];
                optind++;
                pos = 1;
            } else {
                optopt = c;
                optind++;
                pos = 1;
                return '?';
            }
        } else {
            if (arg[pos + 1] == '\0') {
                optind++;
                pos = 1;
            } else {
                pos++;
            }
        }

        return c;
    }

    return -1;
}

#else
#include <getopt.h>
#endif
