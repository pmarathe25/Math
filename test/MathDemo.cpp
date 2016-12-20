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
    math::Matrix<int> mat = math::Matrix<int>(1, 100000); // math::Matrix<int>({{0, 1}, {2, 3}, {4, 5}});
    math::Matrix<int> other = math::Matrix<int>(100000, 1); // math::Matrix<int>({{0, 1, 2}, {3, 4, 5}});
    mat.at(0, 1) = 5;
    // math::display(mat);
    std::cout << std::endl;
    // math::display(other);
    std::cout << std::endl;
    math::display(mat * other);
    std::vector<double> a {2, 0};
    std::vector<double> b = {2, 0};
    std::cout << math::innerProduct(a, b) << std::endl;
    return 0;
}
