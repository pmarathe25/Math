#include "Math/MathParser.hpp"
#include "Math/Matrix.hpp"
#include <iostream>
#include <chrono>

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
    math::Matrix<double> rnd2 = math::Matrix<double>(1, 15);
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
    math::display(toTranspose2 - toTranspose2);
    std::cout << std::endl;
    math::display(1.5 * test + toTranspose2);
    std::cout << std::endl;
    math::display(toTranspose2 - 1.5 * test);
    std::cout << "Dot product." << std::endl;
    // Test File I/O.
    math::Matrix<double> rnd = math::Matrix<double>(1, 15);
    rnd.randomizeUniform(-100, 100);
    std::ofstream saveFile("test/matrix");
    rnd.write(saveFile);
    saveFile.close();
    math::Matrix<double> rndRead;
    std::ifstream readFile("test/matrix");
    rndRead.read(readFile);
    readFile.close();
    math::display(rnd);
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
    math::Matrix<double> vec = math::Matrix<double>({0.0, 0.1, 0.2});
    math::Matrix<double> vec2 = math::Matrix<double>({0.0, 0.1, 0.2});
    vec.randomizeNormal(5, 0.1);
    vec2.randomizeNormal(10, 2.5);
    math::display(vec);
    std::cout << std::endl;
    math::display(vec.raw());
    std::cout << std::endl;
    math::display(vec2);
    std::cout << std::endl;
    math::display(vec * vec2);
    // toTranspose2 = math::Matrix<int>(5, 30);
    std::cout << std::endl;
    math::display((2 * vec).raw());
    std::cout << std::endl;
    math::display((vec + vec2).raw());
    math::display((vec - vec2).raw());
    toTranspose2.randomizeNormal();
    std::cout << std::endl;
    math::display(toTranspose2);
    // math::display(toTranspose3.transpose());
    // std::cout << std::endl;
    // math::display(vec * vec2);
    // std::cout << std::endl;
    // rnd2.randomizeUniform();
    // std::cout << std::endl;
    // math::display(rnd2);
    std::cout << std::endl;
    // Test dot product speed.
    math::Matrix<float> dotTest = math::Matrix<float>(1024 * 16, 1);
    dotTest.randomizeNormal();
    math::Matrix<float> dotTest2 = dotTest;
    // Begin timing.
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    math::display(vec + vec);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    // End timing
    std::cout << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    // CUDA based dot product.
    std::cout << std::endl;
    // Begin timing.
    toTranspose.at(0) = 3;
    math::display(toTranspose);
    t1 = std::chrono::high_resolution_clock::now();
    math::display(toTranspose.dot(toTranspose));
    t2 = std::chrono::high_resolution_clock::now();
    // End timing
    std::cout << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;

}
