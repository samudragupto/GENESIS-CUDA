#include "config_manager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <set>

std::string ConfigManager::trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

bool ConfigManager::parseBool(const std::string& s) {
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower == "true" || lower == "1" || lower == "yes" || lower == "on");
}

bool ConfigManager::isInteger(const std::string& s) {
    if (s.empty()) return false;
    size_t start = 0;
    if (s[0] == '-' || s[0] == '+') start = 1;
    if (start >= s.size()) return false;
    for (size_t i = start; i < s.size(); i++) {
        if (!std::isdigit(s[i])) return false;
    }
    return true;
}

bool ConfigManager::isFloat(const std::string& s) {
    if (s.empty()) return false;
    bool has_dot = false;
    bool has_e = false;
    size_t start = 0;
    if (s[0] == '-' || s[0] == '+') start = 1;
    if (start >= s.size()) return false;
    for (size_t i = start; i < s.size(); i++) {
        if (s[i] == '.') {
            if (has_dot || has_e) return false;
            has_dot = true;
        } else if (s[i] == 'e' || s[i] == 'E') {
            if (has_e) return false;
            has_e = true;
            if (i + 1 < s.size() && (s[i+1] == '+' || s[i+1] == '-')) i++;
        } else if (!std::isdigit(s[i])) {
            return false;
        }
    }
    return has_dot || has_e;
}

std::string ConfigManager::makeFullKey(const char* section, const char* key) const {
    return std::string(section) + "." + std::string(key);
}

bool ConfigManager::loadFromFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ConfigManager: Cannot open " << filename << std::endl;
        return false;
    }

    filepath = filename;
    current_section = "";

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        line = trim(line);

        if (line.empty()) continue;
        if (line[0] == '#') continue;

        if (line[0] == '[') {
            size_t end_bracket = line.find(']');
            if (end_bracket == std::string::npos) {
                std::cerr << "ConfigManager: Malformed section at line " << line_num << std::endl;
                continue;
            }
            current_section = trim(line.substr(1, end_bracket - 1));
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            std::cerr << "ConfigManager: No '=' at line " << line_num << std::endl;
            continue;
        }

        std::string key = trim(line.substr(0, eq_pos));
        std::string val_str = trim(line.substr(eq_pos + 1));

        if (key.empty()) continue;

        if (val_str.size() >= 2 && val_str.front() == '"' && val_str.back() == '"') {
            val_str = val_str.substr(1, val_str.size() - 2);
        }

        size_t comment_pos = val_str.find('#');
        if (comment_pos != std::string::npos) {
            bool in_string = false;
            for (size_t i = 0; i < comment_pos; i++) {
                if (val_str[i] == '"') in_string = !in_string;
            }
            if (!in_string) {
                val_str = trim(val_str.substr(0, comment_pos));
            }
        }

        std::string full_key = current_section.empty() ? key : (current_section + "." + key);

        Value v;

        std::string lower_val = val_str;
        std::transform(lower_val.begin(), lower_val.end(), lower_val.begin(), ::tolower);

        if (lower_val == "true" || lower_val == "false" ||
            lower_val == "yes" || lower_val == "no" ||
            lower_val == "on" || lower_val == "off") {
            v.type = Value::TYPE_BOOL;
            v.bool_val = parseBool(val_str);
            v.int_val = v.bool_val ? 1 : 0;
            v.float_val = v.bool_val ? 1.0f : 0.0f;
            v.string_val = val_str;
        } else if (isInteger(val_str)) {
            v.type = Value::TYPE_INT;
            v.int_val = std::stoi(val_str);
            v.float_val = (float)v.int_val;
            v.bool_val = (v.int_val != 0);
            v.string_val = val_str;
        } else if (isFloat(val_str)) {
            v.type = Value::TYPE_FLOAT;
            v.float_val = std::stof(val_str);
            v.int_val = (int)v.float_val;
            v.bool_val = (v.float_val != 0.0f);
            v.string_val = val_str;
        } else {
            v.type = Value::TYPE_STRING;
            v.string_val = val_str;
            v.int_val = 0;
            v.float_val = 0.0f;
            v.bool_val = false;
        }

        values[full_key] = v;
    }

    file.close();
    return true;
}

bool ConfigManager::saveToFile(const char* filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return false;

    file << "# GENESIS Configuration File" << std::endl;
    file << "# Auto-generated" << std::endl;
    file << std::endl;

    std::set<std::string> sections;
    std::unordered_map<std::string, std::vector<std::pair<std::string, Value>>> section_entries;

    for (auto& kv : values) {
        size_t dot = kv.first.find('.');
        std::string section = "";
        std::string key = kv.first;
        if (dot != std::string::npos) {
            section = kv.first.substr(0, dot);
            key = kv.first.substr(dot + 1);
        }
        sections.insert(section);
        section_entries[section].push_back({key, kv.second});
    }

    for (auto& sec : sections) {
        if (!sec.empty()) {
            file << "[" << sec << "]" << std::endl;
        }

        for (auto& entry : section_entries[sec]) {
            file << entry.first << " = ";
            switch (entry.second.type) {
                case Value::TYPE_INT:
                    file << entry.second.int_val;
                    break;
                case Value::TYPE_FLOAT:
                    file << entry.second.float_val;
                    break;
                case Value::TYPE_BOOL:
                    file << (entry.second.bool_val ? "true" : "false");
                    break;
                case Value::TYPE_STRING:
                    file << "\"" << entry.second.string_val << "\"";
                    break;
            }
            file << std::endl;
        }
        file << std::endl;
    }

    file.close();
    return true;
}

int ConfigManager::getInt(const char* section, const char* key, int default_val) const {
    auto it = values.find(makeFullKey(section, key));
    if (it == values.end()) return default_val;
    return it->second.int_val;
}

float ConfigManager::getFloat(const char* section, const char* key, float default_val) const {
    auto it = values.find(makeFullKey(section, key));
    if (it == values.end()) return default_val;
    return it->second.float_val;
}

bool ConfigManager::getBool(const char* section, const char* key, bool default_val) const {
    auto it = values.find(makeFullKey(section, key));
    if (it == values.end()) return default_val;
    return it->second.bool_val;
}

std::string ConfigManager::getString(const char* section, const char* key, const char* default_val) const {
    auto it = values.find(makeFullKey(section, key));
    if (it == values.end()) return std::string(default_val);
    return it->second.string_val;
}

void ConfigManager::setInt(const char* section, const char* key, int val) {
    Value v;
    v.type = Value::TYPE_INT;
    v.int_val = val;
    v.float_val = (float)val;
    v.bool_val = (val != 0);
    v.string_val = std::to_string(val);
    values[makeFullKey(section, key)] = v;
}

void ConfigManager::setFloat(const char* section, const char* key, float val) {
    Value v;
    v.type = Value::TYPE_FLOAT;
    v.float_val = val;
    v.int_val = (int)val;
    v.bool_val = (val != 0.0f);
    v.string_val = std::to_string(val);
    values[makeFullKey(section, key)] = v;
}

void ConfigManager::setBool(const char* section, const char* key, bool val) {
    Value v;
    v.type = Value::TYPE_BOOL;
    v.bool_val = val;
    v.int_val = val ? 1 : 0;
    v.float_val = val ? 1.0f : 0.0f;
    v.string_val = val ? "true" : "false";
    values[makeFullKey(section, key)] = v;
}

void ConfigManager::setString(const char* section, const char* key, const char* val) {
    Value v;
    v.type = Value::TYPE_STRING;
    v.string_val = val;
    v.int_val = 0;
    v.float_val = 0.0f;
    v.bool_val = false;
    values[makeFullKey(section, key)] = v;
}

bool ConfigManager::hasKey(const char* section, const char* key) const {
    return values.find(makeFullKey(section, key)) != values.end();
}

std::vector<std::string> ConfigManager::getSections() const {
    std::set<std::string> section_set;
    for (auto& kv : values) {
        size_t dot = kv.first.find('.');
        if (dot != std::string::npos) {
            section_set.insert(kv.first.substr(0, dot));
        }
    }
    return std::vector<std::string>(section_set.begin(), section_set.end());
}

std::vector<std::string> ConfigManager::getKeysInSection(const char* section) const {
    std::vector<std::string> keys;
    std::string prefix = std::string(section) + ".";
    for (auto& kv : values) {
        if (kv.first.substr(0, prefix.size()) == prefix) {
            keys.push_back(kv.first.substr(prefix.size()));
        }
    }
    return keys;
}

void ConfigManager::printAll() const {
    std::cout << std::endl;
    std::cout << "========== Configuration ==========" << std::endl;

    std::string last_section = "";
    std::vector<std::pair<std::string, Value>> sorted_entries(values.begin(), values.end());
    std::sort(sorted_entries.begin(), sorted_entries.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& kv : sorted_entries) {
        size_t dot = kv.first.find('.');
        std::string section = "";
        std::string key = kv.first;
        if (dot != std::string::npos) {
            section = kv.first.substr(0, dot);
            key = kv.first.substr(dot + 1);
        }

        if (section != last_section) {
            if (!section.empty()) {
                std::cout << std::endl << "[" << section << "]" << std::endl;
            }
            last_section = section;
        }

        std::cout << "  " << key << " = ";
        switch (kv.second.type) {
            case Value::TYPE_INT:
                std::cout << kv.second.int_val << " (int)";
                break;
            case Value::TYPE_FLOAT:
                std::cout << kv.second.float_val << " (float)";
                break;
            case Value::TYPE_BOOL:
                std::cout << (kv.second.bool_val ? "true" : "false") << " (bool)";
                break;
            case Value::TYPE_STRING:
                std::cout << "\"" << kv.second.string_val << "\" (string)";
                break;
        }
        std::cout << std::endl;
    }
    std::cout << "===================================" << std::endl << std::endl;
}

void ConfigManager::clear() {
    values.clear();
    current_section.clear();
    filepath.clear();
}