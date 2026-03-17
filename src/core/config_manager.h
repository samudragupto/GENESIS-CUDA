#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>

class ConfigManager {
public:
    struct Value {
        enum Type { TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING };
        Type type;
        int int_val;
        float float_val;
        bool bool_val;
        std::string string_val;
    };

    std::unordered_map<std::string, Value> values;
    std::string current_section;
    std::string filepath;

    bool loadFromFile(const char* filename);
    bool saveToFile(const char* filename) const;

    int getInt(const char* section, const char* key, int default_val = 0) const;
    float getFloat(const char* section, const char* key, float default_val = 0.0f) const;
    bool getBool(const char* section, const char* key, bool default_val = false) const;
    std::string getString(const char* section, const char* key, const char* default_val = "") const;

    void setInt(const char* section, const char* key, int val);
    void setFloat(const char* section, const char* key, float val);
    void setBool(const char* section, const char* key, bool val);
    void setString(const char* section, const char* key, const char* val);

    bool hasKey(const char* section, const char* key) const;
    std::vector<std::string> getSections() const;
    std::vector<std::string> getKeysInSection(const char* section) const;

    void printAll() const;
    void clear();

private:
    std::string makeFullKey(const char* section, const char* key) const;
    static std::string trim(const std::string& s);
    static bool parseBool(const std::string& s);
    static bool isInteger(const std::string& s);
    static bool isFloat(const std::string& s);
};

#endif