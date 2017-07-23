#ifndef MATH_PARSER_H
#define MATH_PARSER_H
#include <string>
#include <vector>
#include <map>
#include "Math.hpp"
#include "strmanip.hpp"

namespace math {
    class MathParser {
        public:
            MathParser();
            double parse(std::string expression);
            void setOperatorPrecedenceList(const std::vector<std::string>& newList);
            // Add operators.
            void addBinaryOperator(const std::string& op, double (*func)(double, double));
            void addBinaryOperator(const std::map<std::string, double (*)(double, double)> ops);
            void addUnaryOperator(const std::string& op, double (*func)(double));
            void addUnaryOperator(const std::map<std::string, double (*)(double)>& ops);
        private:
            // Contains indices of the expression and values of operands.
            struct BinaryOperands {
                strmanip::Indices indices;
                double firstOperand, secondOperand;
            };
            // Contains indices of the expression and the value of the operand.
            struct UnaryOperand {
                strmanip::Indices indices;
                double firstOperand;
            };
            // Order of operations.
            std::vector<std::string> operatorPrecedenceList = {"!", "/", "*", "+", "-"};
            // Operator functions.
            std::map<std::string, double (*)(double, double)> binaryOperatorFunctions = {{"/", &math::divide}, {"*", &math::multiply}, {"+", &math::add}, {"-", &math::subtract}};
            std::map<std::string, double (*)(double)> unaryOperatorFunctions = {{"!", &math::factorial}};
            // Methods.
            double parseClean(const std::string& expression);
            BinaryOperands findBinaryOperands(const std::string& expression, const std::string& op, int operatorLocation);
            UnaryOperand findUnaryOperand(const std::string& expression, const std::string& op, int operatorLocation);
    };
}

#endif
