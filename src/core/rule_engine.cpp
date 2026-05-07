/**
 * Rule Engine Implementation
 *
 * Loads rule files and provides additional utilities.
 */

#include "rule_engine.hpp"
#include <fstream>
#include <sstream>

namespace collider {

RuleSet RuleEngine::load_ruleset(const std::string& path) {
    RuleSet ruleset;
    ruleset.name = path;

    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Could not open rule file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        Rule rule;
        rule.definition = line;
        rule.efficiency = 1.0f;  // Default efficiency

        ruleset.rules.push_back(rule);
    }

    ruleset.total_efficiency = static_cast<float>(ruleset.rules.size());
    return ruleset;
}

}  // namespace collider
