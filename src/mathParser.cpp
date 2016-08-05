#include "Math/mathParser.hpp"
#include <algorithm>

MathParser::MathParser() {

}

double MathParser::parse(std::string expression) {
    // Remove spaces.
    expression = strmanip::remove(expression, " ");
    // Balance parentheses.
    expression = strmanip::balance(expression, '(', ')');
    return parseClean(expression);
}

void MathParser::setOperatorPrecedenceList(const std::vector<std::string>& newList) {
    operatorPrecedenceList = newList;
}

void MathParser::addBinaryOperator(const std::string& op, double (*func)(double, double)) {
    binaryOperatorFunctions[op] = func;
    // Push the operator to the front of the precedence list.
    operatorPrecedenceList.insert(operatorPrecedenceList.begin(), op);
}

void MathParser::addBinaryOperator(const std::map<std::string, double (*)(double, double)> ops) {
    binaryOperatorFunctions.insert(ops.begin(), ops.end());
    // Push all provided operators onto the front of the precedence list.
    for (std::map<std::string, double (*)(double, double)>::const_iterator op = ops.begin(); op != ops.end(); op++) {
        operatorPrecedenceList.insert(operatorPrecedenceList.begin(), op -> first);
    }
}

void MathParser::addUnaryOperator(const std::string& op, double (*func)(double)) {
    unaryOperatorFunctions[op] = func;
    // Push the operator to the front of the precedence list.
    operatorPrecedenceList.insert(operatorPrecedenceList.begin(), op);
}

void MathParser::addUnaryOperator(const std::map<std::string, double (*)(double)>& ops) {
    unaryOperatorFunctions.insert(ops.begin(), ops.end());
    // Push all provided operators onto the front of the precedence list.
    for (std::map<std::string, double (*)(double)>::const_iterator op = ops.begin(); op != ops.end(); op++) {
        operatorPrecedenceList.insert(operatorPrecedenceList.begin(), op -> first);
    }
}

double MathParser::parseClean(const std::string&  expression) {
    // If there are no operators, return the double value of the expression.
    if (!strmanip::contains(expression, operatorPrecedenceList)) {
        return std::stod(strmanip::remove(expression, "()"));
    }
    // Evaluate innermost parentheses first.
    strmanip::Indices inner = strmanip::findInnermost(expression, '(', ')');
    if (inner.first != -1) {
        // Recursively parse the inner expression excluding the parentheses and then parse the new expression.
        return parseClean(expression.substr(0, inner.first) + std::to_string(parseClean(expression.substr(inner.first + 1, inner.second - inner.first - 1)))
            + expression.substr(inner.second + 1));
    }
    // Evaluate operators in the order they are provided in the precedence list.
    for (std::vector<std::string>::iterator op = operatorPrecedenceList.begin(); op != operatorPrecedenceList.end(); op++) {
        // Find the first instance of the operator.
        int operatorLocation = expression.find(*op);
        if (operatorLocation != std::string::npos) {
            // If this is a binary operator call a binary function, otherwise call a unary function.
            if (binaryOperatorFunctions.count(*op) > 0) {
                // Create a struct to hold the two operators and the start and end indices of the expression.
                BinaryOperands binaryOperands = findBinaryOperands(expression, *op, operatorLocation);
                return parseClean(expression.substr(0, binaryOperands.indices.first) +
                    std::to_string((*binaryOperatorFunctions.at(*op))(binaryOperands.firstOperand, binaryOperands.secondOperand))
                    + expression.substr(binaryOperands.indices.second));
            } else if (unaryOperatorFunctions.count(*op) > 0) {
                // Create a struct to hold the operator and the start and end indices of the expression.
                UnaryOperand unaryOperand = findUnaryOperand(expression, *op, operatorLocation);
                return parseClean(expression.substr(0, unaryOperand.indices.first)
                    + std::to_string((*unaryOperatorFunctions.at(*op))(unaryOperand.firstOperand))
                    + expression.substr(unaryOperand.indices.second));
            }
        }
    }
}

MathParser::UnaryOperand MathParser::findUnaryOperand(const std::string& expression, const std::string& op, int operatorLocation) {
    UnaryOperand temp;
    // Get the index of the beginning of the first operand..
    int operandStart = expression.find_last_not_of("0123456789.", operatorLocation - 1) + 1;
    // Get the index after the end of the unary operator.
    int operatorEnd = operatorLocation + op.size();
    // If this is the last operator in the string, set the index to the end of the string.
    operatorEnd = (operatorEnd == std::string::npos) ? expression.length() - 1 : operatorEnd;
    // Get the operand as a double.
    double operand = std::stod(expression.substr(operandStart, operatorLocation - operandStart));

    temp.indices.first = operandStart;
    temp.indices.second = operatorEnd;
    temp.firstOperand = operand;
    return temp;
}

// Gives the values and locations of the operands.
MathParser::BinaryOperands MathParser::findBinaryOperands(const std::string& expression, const std::string& op, int operatorLocation) {
    BinaryOperands temp;
    // Get the index of the beginning of the first operand..
    int firstOperandStart = expression.find_last_not_of("0123456789.", operatorLocation - 1) + 1;
    // Get the index after the end of the second operand.
    int secondOperandEnd = expression.find_first_not_of("0123456789.", operatorLocation + op.size());
    // If this is the last operand in the string, set the index to the end of the string.
    secondOperandEnd = (secondOperandEnd == std::string::npos) ? expression.length() : secondOperandEnd;
    // Get the two operands as doubles.
    double firstOperand = std::stod(expression.substr(firstOperandStart, operatorLocation - firstOperandStart));
    double secondOperand = std::stod(expression.substr(operatorLocation + op.size(), secondOperandEnd + 1 - operatorLocation));
    temp.indices.first = firstOperandStart;
    temp.indices.second = secondOperandEnd;
    temp.firstOperand = firstOperand;
    temp.secondOperand = secondOperand;
    return temp;
}
