#include "Math/MathParser.hpp"
#include "Math/Matrix.hpp"
#include <iostream>

double postIncrement(double operand) {
    return operand + 1;
}

int main() {
    // math::MathParser parser = math::MathParser();
    // parser.addUnaryOperator("++", &postIncrement);
    // std::cout << parser.parse("1++*(3++!)++/9*4++-1+++++3") << std::endl;
    // math::Matrix<float> mat = math::Matrix<float>({{0, 1}, {2, 3}, {4, 5}});
    // math::Matrix<float> other = math::Matrix<float>({0, 1, 2, 3, 4, 5}, 2, 3);
    int testSize = 40;
    math::Matrix<double> toTranspose = math::Matrix<double>(testSize, testSize);
    math::Matrix<float> toTranspose2 = math::Matrix<float>(5, 30);
    math::Matrix<float> toTranspose3 = math::Matrix<float>(15, 5);
    math::display(toTranspose3 * toTranspose2);
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
    math::Matrix<double> test = toTranspose2;
    math::Matrix<double> rnd = math::Matrix<double>(1, 15);
    std::cout << "Finished creating vector matrix." << std::endl;
    rnd.randomizeNormal(5, 0.1);
    std::cout << "Finished normal randomization." << std::endl;
    std::cout << rnd.numColumnsRaw() << std::endl;
    std::cout << rnd.numColumns() << std::endl;
    std::vector<float> a = {0, 0, 1, 0}; //, 4, 5, 6, 7, 2, 3, 4, 5, 4, 5, 6};
    std::vector<float> b = {2, 0, 5, 6}; //, 5, 6, 7, 8, 3, 2, 4, 5, 6, 76, 54};
    // math::display(mat);
    // std::cout << std::endl;
    // math::display(other);
    // std::cout << std::endl;
    // math::display(mat * other);
    // std::cout << std::endl;
    math::display(toTranspose);
    std::cout << std::endl;
    math::display(toTranspose.transpose());
    std::cout << std::endl;
    math::display(toTranspose2);
    std::cout << std::endl;
    math::display(toTranspose2.transpose().transpose());
    std::cout << std::endl;
    math::display(toTranspose3);
    std::cout << std::endl;
    math::display(toTranspose3.transpose());
    std::cout << std::endl;
    math::display(toTranspose * toTranspose * toTranspose);
    std::cout << std::endl;
    math::display(toTranspose2.getElements());
    std::cout << std::endl;
    math::display(toTranspose2 + toTranspose2);
    std::cout << std::endl;
    math::display(1.5 * test + toTranspose2);
    std::cout << std::endl;
    math::display(toTranspose2 + 1.5 * test);
    std::cout << std::endl;
    math::display(rnd);
    std::cout << std::endl;
    std::cout << math::innerProduct(a, b) << std::endl;
    // Test File I/O.
    std::ofstream saveFile("test/matrix");
    rnd.write(saveFile);
    saveFile.close();
    math::Matrix<int> rndRead;
    std::ifstream readFile("test/matrix");
    rndRead.read(readFile);
    readFile.close();
    math::display(rndRead);
    // Equality test.
    math::Matrix<int> equalTest = math::Matrix<int>(4, 4);
    math::Matrix<int> equalTest2 = math::Matrix<int>(4, 4);
    std::cout << std::endl;
    if (equalTest == equalTest2) {
        std::cout << "Success! Matrices are equal!" << std::endl;
    }
    std::cout << std::endl;
    // Vector testing.
    math::Matrix<float> vec = math::Matrix<float>({0, 1, 2, 3, 4});
    math::Matrix<float> vec2 = math::Matrix<float>({0, 1, 2, 3, 4});
    math::display(vec);
    // toTranspose2 = math::Matrix<int>(5, 30);
    math::display(toTranspose2);
    std::cout << std::endl;
    math::display(vec * toTranspose2);
    return 0;
}
