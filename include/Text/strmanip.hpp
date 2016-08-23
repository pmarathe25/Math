#ifndef STRMANIP_H
#define STRMANIP_H
#include <string>
#include <vector>

namespace strmanip {
    struct Indices {
        int first, second;
    };
    bool contains(const std::string& expression, const std::vector<std::string>& toFind);
    bool contains(const std::string& expression, const std::string& toFind);
    std::string remove(std::string expression, const std::string& toRemove);
    std::string balance(std::string expression, char openingChar, char closingChar);
    strmanip::Indices findInnermost(const std::string& expression, char openingChar, char closingChar);
}

#endif
