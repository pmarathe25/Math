#ifndef MATH_PARSER_H
#define MATH_PARSER_H
#include <string>
#include <utility>
#include <vector>
#include "math.hpp"

class MathParser {
    public:
        MathParser();
        double parse(const std::string& expression);
    private:
        std::string operators = "/*+-";
        bool containsOperators(const std::string& expression);
        double getDoubleValue(const std::string& expression);
        std::pair<int, int> findInnermostParens(const std::string& expression);
        std::pair<std::pair<double, double>, std::pair<int, int> > findOperands(const std::string& expression, int operatorLocation);
        std::vector<double (*)(std::pair<double, double>)> operatorFunctions = {&divide, &multiply, &add, &subtract};
};

#endif
