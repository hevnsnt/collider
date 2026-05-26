// settings_sidecar.hpp -- TR-5: persist TUI Settings edits to
// ~/.collider/settings.json so they survive across launches.
//
// Why JSON sidecar instead of editing config.yml in-place: the YAML
// loader uses a hand-rolled parser that handles indented blocks +
// comments; round-trip editing without breaking existing comments
// is non-trivial. A sidecar JSON file is trivially round-trippable
// and the loader merges its values OVER the YAML defaults at startup
// (so config.yml stays the operator's hand-curated source, sidecar
// holds the TUI-modal edits).
//
// Schema (all fields optional, defaults match SettingsValues):
//   {
//     "backend_kind":  "cuda" | "cpu" | "metal" | "",
//     "solver":        "kangaroo" | "bsgs" | "",
//     "num_kangaroos": int,
//     "batch_size":    int,
//     "dp_bits":       int,
//     "refresh_hz":    int,
//     "theme":         "Default" | "HighContrast" | "Monochrome" | "Light",
//     "verbose":       bool
//   }

#pragma once

#include "core/paths.hpp"
#include "ui/tui/panels/settings_panel.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace collider::settings_sidecar {

inline std::filesystem::path sidecar_path() {
    return collider::paths::collider_home() / "settings.json";
}

// Map a ThemeVariant enum to its serialized string name.
inline const char* theme_name(::collider::ui::tui::ThemeVariant v) {
    switch (v) {
        case ::collider::ui::tui::ThemeVariant::Default:      return "Default";
        case ::collider::ui::tui::ThemeVariant::HighContrast: return "HighContrast";
        case ::collider::ui::tui::ThemeVariant::Monochrome:   return "Monochrome";
        case ::collider::ui::tui::ThemeVariant::Light:        return "Light";
    }
    return "Default";
}

inline ::collider::ui::tui::ThemeVariant theme_from_name(const std::string& s) {
    if (s == "HighContrast") return ::collider::ui::tui::ThemeVariant::HighContrast;
    if (s == "Monochrome")   return ::collider::ui::tui::ThemeVariant::Monochrome;
    if (s == "Light")        return ::collider::ui::tui::ThemeVariant::Light;
    return ::collider::ui::tui::ThemeVariant::Default;
}

// Write the current SettingsValues to ~/.collider/settings.json.
// Returns true on success. The write is best-effort atomic via
// the standard write-tmp + rename pattern; partial writes are
// cleaned up on failure.
inline bool save(const ::collider::ui::tui::panels::SettingsValues& v) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(sidecar_path().parent_path(), ec);
    if (ec && !fs::exists(sidecar_path().parent_path())) return false;

    fs::path tmp = sidecar_path();
    tmp += ".tmp";

    auto escape = [](const std::string& s) {
        std::string out;
        out.reserve(s.size() + 2);
        for (char c : s) {
            if (c == '"' || c == '\\') out.push_back('\\');
            out.push_back(c);
        }
        return out;
    };

    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"backend_kind\":  \"" << escape(v.backend_kind) << "\",\n";
    oss << "  \"solver\":        \"" << escape(v.solver) << "\",\n";
    oss << "  \"num_kangaroos\": " << v.num_kangaroos << ",\n";
    oss << "  \"batch_size\":    " << v.batch_size << ",\n";
    oss << "  \"dp_bits\":       " << v.dp_bits << ",\n";
    oss << "  \"refresh_hz\":    " << v.refresh_hz << ",\n";
    oss << "  \"theme\":         \"" << theme_name(v.theme) << "\",\n";
    oss << "  \"verbose\":       " << (v.verbose ? "true" : "false") << "\n";
    oss << "}\n";

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << oss.str();
        if (!f) {
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            return false;
        }
    }

    fs::rename(tmp, sidecar_path(), ec);
    if (ec) {
        std::error_code rm_ec;
        fs::remove(sidecar_path(), rm_ec);
        fs::rename(tmp, sidecar_path(), ec);
        if (ec) {
            fs::remove(tmp, rm_ec);
            return false;
        }
    }
    return true;
}

// Load the sidecar into `v`. Returns true if the file existed AND was
// parseable; on missing file or malformed JSON returns false and leaves
// `v` untouched. The parser is intentionally minimal -- field-by-field
// regex-style scan because the schema is fixed and we control the
// writer.
inline bool load(::collider::ui::tui::panels::SettingsValues& v) {
    std::ifstream f(sidecar_path(), std::ios::binary);
    if (!f) return false;
    std::ostringstream oss;
    oss << f.rdbuf();
    const std::string text = oss.str();
    if (text.empty()) return false;

    auto find_string_field = [&](const char* key, std::string& out) -> bool {
        const std::string k = std::string("\"") + key + "\":";
        size_t p = text.find(k);
        if (p == std::string::npos) return false;
        p += k.size();
        while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) ++p;
        if (p >= text.size() || text[p] != '"') return false;
        ++p;
        size_t end = p;
        while (end < text.size() && text[end] != '"') {
            if (text[end] == '\\' && end + 1 < text.size()) ++end;
            ++end;
        }
        if (end > text.size()) return false;
        out.assign(text.begin() + p, text.begin() + end);
        return true;
    };

    auto find_int_field = [&](const char* key, long long& out) -> bool {
        const std::string k = std::string("\"") + key + "\":";
        size_t p = text.find(k);
        if (p == std::string::npos) return false;
        p += k.size();
        while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) ++p;
        size_t end = p;
        while (end < text.size() &&
               (text[end] == '-' || (text[end] >= '0' && text[end] <= '9'))) {
            ++end;
        }
        if (end == p) return false;
        try {
            out = std::stoll(text.substr(p, end - p));
        } catch (const std::exception&) {
            return false;
        }
        return true;
    };

    auto find_bool_field = [&](const char* key, bool& out) -> bool {
        const std::string k = std::string("\"") + key + "\":";
        size_t p = text.find(k);
        if (p == std::string::npos) return false;
        p += k.size();
        while (p < text.size() && (text[p] == ' ' || text[p] == '\t')) ++p;
        if (text.compare(p, 4, "true") == 0)  { out = true;  return true; }
        if (text.compare(p, 5, "false") == 0) { out = false; return true; }
        return false;
    };

    std::string s;
    long long n = 0;
    bool b = false;
    if (find_string_field("backend_kind", s)) v.backend_kind = s;
    if (find_string_field("solver",       s)) v.solver = s;
    if (find_int_field("num_kangaroos",   n)) v.num_kangaroos = static_cast<int>(n);
    if (find_int_field("batch_size",      n)) v.batch_size    = static_cast<size_t>(n);
    if (find_int_field("dp_bits",         n)) v.dp_bits       = static_cast<int>(n);
    if (find_int_field("refresh_hz",      n)) v.refresh_hz    = static_cast<int>(n);
    if (find_string_field("theme",        s)) v.theme         = theme_from_name(s);
    if (find_bool_field("verbose",        b)) v.verbose       = b;
    return true;
}

}  // namespace collider::settings_sidecar
