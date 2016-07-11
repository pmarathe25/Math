#ifndef MATH_PARSER_H
#define MATH_PARSER_H
#include <string>
#include <vector>
#include <map>
#include "math.hpp"

class MathParser {
    public:
        MathParser();
        double parse(std::string expression);
        void setOperatorPrecedenceList(const std::vector<std::string>& newList);
        void addBinaryOperator(const std::string& op, double (*func)(double, double));
        void addBinaryOperator(const std::map<std::string, double (*)(double, double)> ops);
        void addUnaryOperator(const std::string& op, double (*func)(double));
        void addUnaryOperator(const std::map<std::string, double (*)(double)>& ops);
    private:
        struct Indices {
            int first, second;
        };
        struct BinaryOperands {
            Indices indices;
            double firstOperand, secondOperand;
        };
        struct UnaryOperand {
            Indices indices;
            double firstOperand;
        };
        // Order of operations.
        std::vector<std::string> operatorPrecedenceList = {"!", "/", "*", "+", "-"};
        // Operator functions.
        std::map<std::string, double (*)(double, double)> binaryOperatorFunctions = {{"/", &divide}, {"*", &multiply}, {"+", &add}, {"-", &subtract}};
        std::map<std::string, double (*)(double)> unaryOperatorFunctions = {{"!", &factorial}};
        // Methods.
        double parseClean(const std::string&  expression);
        bool containsOperators(const std::string& expression);
        double removeParentheses(std::string expression);
        std::string balanceParentheses(std::string expression);
        Indices findInnermostParentheses(const std::string& expression);
        BinaryOperands findBinaryOperands(const std::string& expression, const std::string& op, int operatorLocation);
        UnaryOperand findUnaryOperand(const std::string& expression, const std::string& op, int operatorLocation);
};

#endif
