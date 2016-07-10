#ifndef MATH_PARSER_H
#define MATH_PARSER_H
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "math.hpp"

class MathParser {
    public:
        MathParser();
        double parse(const std::string& expression);
    private:
        int numUnaryOperators = 1;
        std::vector<std::string> operatorPrecedenceList = {"!", "/", "*", "+", "-"};
        std::map<std::string, double (*)(std::pair<double, double>)> binaryOperatorFunctions = {{"/", &divide}, {"*", &multiply}, {"+", &add}, {"-", &subtract}};
        std::map<std::string, double (*)(double)> unaryOperatorFunctions = {{"!", &factorial}};
        bool containsOperators(const std::string& expression);
        double getDoubleValue(const std::string& expression);
        // Methods.
        std::pair<int, int> findInnermostParens(const std::string& expression);
        std::pair<std::pair<double, double>, std::pair<int, int> > findBinaryOperands(const std::string& expression, const std::string& op, int operatorLocation);
        std::pair<double, std::pair<int, int> > findUnaryOperand(const std::string& expression, const std::string& op, int operatorLocation);
};

#endif
