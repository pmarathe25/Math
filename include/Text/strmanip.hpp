#ifndef STRMANIP_H
#define STRMANIP_H
#include <string>
#include <vector>

namespace strmanip {
    struct Indices {
        int first, second;
    };
    // Contains.
    bool contains(const std::string& expression, const std::vector<std::string>& toFind);
    bool contains(const std::string& expression, const std::string& toFind);
    bool contains(const std::string& expression, char toFind);
    std::string remove(std::string expression, const std::string& toRemove);
    // Split.
    std::vector<std::string> split(const std::string& expression, char delim);
    std::vector<std::string> split(const std::string& expression, const std::string& delims);
    // Parentheses methods.
    std::string balance(std::string expression, char openingChar, char closingChar);
    strmanip::Indices findInnermost(const std::string& expression, char openingChar, char closingChar);
}

#endif
