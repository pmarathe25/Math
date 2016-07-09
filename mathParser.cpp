#include "mathParser.hpp"

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
        // Evaluate operators in the order they are provided.
        for (int i = 0; i < operators.length(); i++) {
            int operatorLocation = expression.find(operators.at(i));
            if (operatorLocation != std::string::npos) {
                std::pair<std::pair<double, double>, std::pair<int, int> > operands = findOperands(expression, operatorLocation);
                return parse(expression.substr(0, operands.second.first) + std::to_string((*operatorFunctions.at(i))(operands.first))
                    + expression.substr(operands.second.second));
            }
        }
    }
}

bool MathParser::containsOperators(const std::string& expression) {
    if (expression.find_first_of(operators) != std::string::npos) {
        return true;
    } else {
        return false;
    }
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

// Gives the values and locations of the operands.
std::pair<std::pair<double, double>, std::pair<int, int> > MathParser::findOperands(const std::string& expression, int operatorLocation) {
    // Get the index of the beginning of the first operand..
    int firstOperandStart = expression.find_last_not_of("0123456789.", operatorLocation - 1) + 1;
    // Get the index of the end of the second operand.
    int secondOperandEnd = expression.find_first_not_of("0123456789.", operatorLocation + 1);
    // If this is the last operand in the string, set the index to the end of the string.
    secondOperandEnd = (secondOperandEnd == std::string::npos) ? expression.length() - 1 : secondOperandEnd - 1;
    // Get the two operands as doubles.
    double firstOperand = std::stod(expression.substr(firstOperandStart, operatorLocation - firstOperandStart));
    double secondOperand = std::stod(expression.substr(operatorLocation + 1, secondOperandEnd - operatorLocation));
    // Make pairs.
    std::pair<double, double> operands = std::make_pair(firstOperand, secondOperand);
    std::pair<int, int> indices = std::make_pair(firstOperandStart, secondOperandEnd);
    return std::make_pair(operands, indices);
}
