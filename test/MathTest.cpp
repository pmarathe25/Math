#include "Math/MathParser.hpp"
#include "Math/Matrix.hpp"
#include <iostream>
#include <chrono>

double postIncrement(double operand) {
    return operand + 1;
}

int main() {
    math::Matrix<float> mat = math::Matrix<float>({{0, 1, 0}, {0, 2, 3}});
    math::Matrix<float> other = math::Matrix<float>({0, 1, 2, 3, 4, 5}, 2, 3);
    int testSize = 40;
    math::Matrix<double> toTranspose = math::Matrix<double>(testSize, testSize);
    math::Matrix<float> toTranspose2 = math::Matrix<float>(5, 30);
    math::Matrix<float> toTranspose3 = math::Matrix<float>(15, 5);
    std::cout << "Matrices created." << std::endl;
    for (int i = 0; i < testSize; ++i) {
        for (int j = 0; j < testSize; ++j) {
            toTranspose.at(i, j) = 1;
        }
    }
    for (int i = 0; i < 150; ++i) {
        toTranspose2.at(i) = i;
    }
    for (int i = 0; i < 75; ++i) {
        toTranspose3.at(i) = i;
    }
    toTranspose.display();
    toTranspose2.display();
    toTranspose3.display();
}
