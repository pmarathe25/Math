#include "mathParser.hpp"
#include <iostream>
#include <chrono>
#include <thread>

MathParser::MathParser() {

}

double MathParser::parse(const std::string& expression) {
    if (!containsOperators(expression)) {
        return getDoubleValue(expression);
    }
    // If there is an expression in parentheses, evaluate it first.
    std::pair<int, int> inner = findInnermostParens(expression);
    if (inner.first != -1) {
        // Recursively parse the inner expression excluding the parentheses and then parse the new expression.
        return parse(expression.substr(0, inner.first) + std::to_string(parse(expression.substr(inner.first + 1, inner.second - inner.first - 1)))
            + expression.substr(inner.second + 1));
    } else {
        // Evaluate unary operators.
        for (std::vector<std::string>::iterator op = operatorPrecedenceList.begin(); op != operatorPrecedenceList.begin() + numUnaryOperators; op++) {
            int operatorLocation = expression.find(*op);
            if (operatorLocation != std::string::npos) {
                // Create a pair to hold the operator and the start and end indices of the expression.
                std::pair<double, std::pair<int, int> > operand = findUnaryOperand(expression, *op, operatorLocation);


                return parse(expression.substr(0, operand.second.first) + std::to_string((*unaryOperatorFunctions.at(*op))(operand.first))
                    + expression.substr(operand.second.second));
            }
        }
        // Evaluate binary operators in the order they are provided in the precedence list.
        for (std::vector<std::string>::iterator op = operatorPrecedenceList.begin() + numUnaryOperators; op != operatorPrecedenceList.end(); op++) {
            // Find the first instance of the operator.
            int operatorLocation = expression.find(*op);
            if (operatorLocation != std::string::npos) {
                // Create a pair to hold the two operators and the start and end indices of the expression.
                std::pair<std::pair<double, double>, std::pair<int, int> > operands = findBinaryOperands(expression, *op, operatorLocation);

                std::cout << expression.substr(0, operands.second.first) << "Value of 7!: " << std::to_string((*binaryOperatorFunctions.at(*op))(operands.first))
                    << " Rest of string: " << expression.substr(operands.second.second) << std::endl;
                std::this_thread::sleep_for (std::chrono::seconds(1));


                return parse(expression.substr(0, operands.second.first) + std::to_string((*binaryOperatorFunctions.at(*op))(operands.first))
                    + expression.substr(operands.second.second));
            }
        }
    }
}

bool MathParser::containsOperators(const std::string& expression) {
    for (std::vector<std::string>::iterator op = operatorPrecedenceList.begin(); op != operatorPrecedenceList.end(); op++) {
        if (expression.find_first_of(*op) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Accepts a single string value and returns the double equivalent after removing extraneous parentheses.
double MathParser::getDoubleValue(const std::string& expression) {
    // Remove all parenthesis
    std::string temp = expression;
    for (std::string::iterator character = temp.begin(); character != temp.end(); character++) {
        if ((*character) == '(' || (*character) == ')') {
            temp.erase(character);
        }
    }
    return std::stod(temp);
}

std::pair<int, int> MathParser::findInnermostParens(const std::string& expression) {
    int maxDepth = 0;
    int maxDepthStart = -1;
    int maxDepthEnd = -1;
    int depth = 0;
    for (int i = 0; i < expression.length(); i++) {
        if (expression.at(i) == '(') {
            // Keep max depth updated and let maxDepthStart be the index of the innermost parenthesis so far.
            depth++;
            if (depth >= maxDepth) {
                maxDepth = depth;
                maxDepthStart = i;
            }
        } else if (expression.at(i) == ')') {
            // If this is the matching end parenthesis for the innermost start parenthesis, update maxDepthEnd.
            if (depth == maxDepth) {
                maxDepthEnd = i;
            }
            depth--;
        }
    }
    return std::make_pair(maxDepthStart, maxDepthEnd);
}

std::pair<double, std::pair<int, int> > MathParser::findUnaryOperand(const std::string& expression, const std::string& op, int operatorLocation) {
    // Get the index of the beginning of the first operand..
    int operandStart = expression.find_last_not_of("0123456789.", operatorLocation - 1) + 1;
    // Get the index after the end of the unary operator.
    int operatorEnd = operatorLocation + op.size();
    // If this is the last operator in the string, set the index to the end of the string.
    operatorEnd = (operatorEnd == std::string::npos) ? expression.length() - 1 : operatorEnd;
    // Get the operand as a double.
    double operand = std::stod(expression.substr(operandStart, operatorLocation - operandStart));
    // Make pairs.
    std::pair<int, int> indices = std::make_pair(operandStart, operatorEnd);
    return std::make_pair(operand, indices);
}

// Gives the values and locations of the operands.
std::pair<std::pair<double, double>, std::pair<int, int> > MathParser::findBinaryOperands(const std::string& expression, const std::string& op, int operatorLocation) {
    // Get the index of the beginning of the first operand..
    int firstOperandStart = expression.find_last_not_of("0123456789.", operatorLocation - 1) + 1;
    // Get the index after the end of the second operand.
    int secondOperandEnd = expression.find_first_not_of("0123456789.", operatorLocation + op.size());
    // If this is the last operand in the string, set the index to the end of the string.
    secondOperandEnd = (secondOperandEnd == std::string::npos) ? expression.length() : secondOperandEnd;
    // Get the two operands as doubles.
    double firstOperand = std::stod(expression.substr(firstOperandStart, operatorLocation - firstOperandStart));
    double secondOperand = std::stod(expression.substr(operatorLocation + op.size(), secondOperandEnd + 1 - operatorLocation));
    // Make pairs.
    std::pair<double, double> operands = std::make_pair(firstOperand, secondOperand);
    std::pair<int, int> indices = std::make_pair(firstOperandStart, secondOperandEnd);
    return std::make_pair(operands, indices);
}
