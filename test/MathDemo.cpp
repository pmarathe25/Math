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
    math::Matrix<float> mat = math::Matrix<float>(20, 100000); // math::Matrix<float>({{0, 1}, {2, 3}, {4, 5}});
    math::Matrix<float> other = math::Matrix<float>(100000, 20); // math::Matrix<float>({{0, 1, 2}, {3, 4, 5}});
    int testSize = 40;
    math::Matrix<float> toTranspose = math::Matrix<float>(testSize, testSize);
    math::Matrix<float> toTranspose2 = math::Matrix<float>(5, 4);
    math::Matrix<float> toTranspose3 = math::Matrix<float>(5, 5);
    for (int i = 0; i < testSize * testSize; ++i) {
        toTranspose.at(i) = i;
    }
    for (int i = 0; i < 20; ++i) {
        toTranspose2.at(i) = i;
    }
    for (int i = 0; i < 25; ++i) {
        toTranspose3.at(i) = i;
    }
    mat.at(0, 1) = 5;
    // math::display(mat);
    // std::cout << std::endl;
    // math::display(other);
    // std::cout << std::endl;
    // math::display(mat * other);
    math::display(toTranspose);
    std::cout << std::endl;
    math::display(toTranspose.transpose());
    std::cout << std::endl;
    math::display(toTranspose2);
    std::cout << std::endl;
    math::display(toTranspose2.transpose());
    std::cout << std::endl;
    math::display(toTranspose3);
    std::cout << std::endl;
    math::display(toTranspose3.transpose());
    std::cout << std::endl;
    std::vector<float> a {2, 0}; //, 4, 5, 6, 7, 2, 3, 4, 5, 4, 5, 6};
    std::vector<float> b = {2, 0}; //, 5, 6, 7, 8, 3, 2, 4, 5, 6, 76, 54};
    std::cout << math::innerProduct(a, b) << std::endl;
    return 0;
}
