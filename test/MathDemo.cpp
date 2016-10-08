#include "Math/MathParser.hpp"
#include "Math/Matrix.hpp"
#include <iostream>

double postIncrement(double operand) {
    return operand + 1;
}

int main() {
    math::MathParser parser = math::MathParser();
    parser.addUnaryOperator("++", &postIncrement);
    std::cout << parser.parse("1++*(3++!)++/9*4++-1+++++3") << std::endl;
    math::Matrix<int> mat = math::Matrix<int>(5, 5);
    mat.at(0, 4) = 5;
    mat.display();
    mat.display(mat.getRow(0));
    return 0;
}
