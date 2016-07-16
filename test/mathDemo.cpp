#include "Math/mathParser.hpp"
#include <iostream>

double postIncrement(double operand) {
    return operand + 1;
}

int main() {
    MathParser parser = MathParser();
    parser.addUnaryOperator("++", &postIncrement);
    std::cout << parser.parse("1++*(3++!)++/9*4++-1+++++3") << std::endl;
    return 0;
}
