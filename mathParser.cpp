#include "mathParser.hpp"
#include <algorithm>

MathParser::MathParser() {

}

double MathParser::parse(std::string expression) {
    // Remove spaces.
    expression.erase(std::remove(expression.begin(), expression.end(), ' '), expression.end());
    // Balance parentheses.
    expression = balanceParentheses(expression);
    return parseClean(expression);
}

void MathParser::setOperatorPrecedenceList(const std::vector<std::string>& newList) {
    operatorPrecedenceList = newList;
}

void MathParser::addBinaryOperator(const std::string& op, double (*func)(double, double)) {
    binaryOperatorFunctions.at(op) = func;
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
    if (!containsOperators(expression)) {
        return removeParentheses(expression);
    }
    // If there is an expression in parentheses, evaluate it first.
    Indices inner = findInnermostParentheses(expression);
    if (inner.first != -1) {
        // Recursively parse the inner expression excluding the parentheses and then parse the new expression.
        return parseClean(expression.substr(0, inner.first) + std::to_string(parseClean(expression.substr(inner.first + 1, inner.second - inner.first - 1)))
            + expression.substr(inner.second + 1));
    } else {
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
double MathParser::removeParentheses(std::string expression) {
    // Remove all parentheses
    expression.erase(std::remove(expression.begin(), expression.end(), '('), expression.end());
    expression.erase(std::remove(expression.begin(), expression.end(), ')'), expression.end());
    return std::stod(expression);
}

std::string MathParser::balanceParentheses(std::string expression) {
    int parenthesesBalance = 0;
    // Remove extra right parentheses.
    for (std::string::iterator it = expression.begin(); it != expression.end(); it++) {
        parenthesesBalance += (*it == '(');
        parenthesesBalance -= (*it == ')');
        // If there are more right parentheses than left as the string is traversed left to right, remove them.
        if (parenthesesBalance < 0) {
            expression.erase(it);
            parenthesesBalance++;
            it--;
        }
    }
    // Remove extra left parentheses.
    parenthesesBalance = 0;
    for (std::string::iterator it = expression.end() - 1; it != expression.begin() - 1; it--) {
        parenthesesBalance += (*it == '(');
        parenthesesBalance -= (*it == ')');
        // If there are more left parentheses than right as the string is traversed right to left, remove them.
        if (parenthesesBalance > 0) {
            expression.erase(it);
            // Subtracting two, because the next iteration will once again consider the previous parenthesis.
            parenthesesBalance -= 2;
            it++;
        }
    }
    return expression;
}

MathParser::Indices MathParser::findInnermostParentheses(const std::string& expression) {
    Indices temp;
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
    temp.first = maxDepthStart;
    temp.second = maxDepthEnd;
    return temp;
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
