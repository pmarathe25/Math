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
    std::cout << "Matrices Assigned." << std::endl;
    math::Matrix<float> test = toTranspose2;
    std::cout << "Matrix copied." << std::endl;
    math::display(mat);
    std::cout << std::endl;
    math::display(other);
    std::cout << std::endl;
    // math::display(mat * other);
    std::cout << std::endl;
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

    math::display(toTranspose3);
    std::cout << std::endl;

    math::display(toTranspose * toTranspose * toTranspose);
    std::cout << std::endl;
    std::cout << "RAW" << std::endl;
    math::display(toTranspose2.raw());
    std::cout << std::endl;
    math::display(toTranspose2 - toTranspose2);
    std::cout << std::endl;
    math::display((test * 1.5) + toTranspose2);
    std::cout << std::endl;
    math::display(toTranspose2 - (1.5 * test));
    std::cout << "Dot product." << std::endl;
    // Test File I/O.
    math::Matrix<double> rndRead;
    // Equality test.
    math::Matrix<int> equalTest = math::Matrix<int>(4, 4);
    math::Matrix<int> equalTest2 = math::Matrix<int>(4, 4);
    equalTest.at(0) = 1;
    equalTest2.at(0) = 0;
    std::cout << std::endl;
    if (equalTest == equalTest2) {
        std::cout << "Success! Matrices are equal!" << std::endl;
    } else {
        std::cout << "Matrices are NOT equal!" << std::endl;
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
    math::display(vec.dot(vec2));
    // toTranspose2 = math::Matrix<int>(5, 30);
    std::cout << std::endl;
    math::display((2 * vec).raw());
    std::cout << std::endl;
    std::cout << "SUM" << std::endl;
    math::display((vec + vec2).raw());
    math::display((vec - vec2).raw());
    toTranspose2.randomizeNormal();
    std::cout << std::endl;
    math::display(toTranspose2);
    // math::display(toTranspose3.transpose());
    // std::cout << std::endl;
    // math::display(vec * vec2);
    // std::cout << std::endl;
    // std::cout << std::endl;
    // math::display(rnd2);
    std::cout << std::endl;
    // Test dot product speed.
    math::Matrix<double> dotTest = math::Matrix<double>(2, 1024 * 8);
    dotTest.randomizeNormal();

    // Begin timing.
    math::Matrix<double> rnd = math::Matrix<double>(1024, 16);
    std::cout << "RANDOMIZING" << std::endl;
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    rnd.randomizeUniform(-1, 1);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    std::cout << std::endl;
    // Sum.
    // std::cout << "DOT PRODUCT" << std::endl;
    // t1 = std::chrono::high_resolution_clock::now();
    // math::display(toTranspose.dot(toTranspose));
    // t2 = std::chrono::high_resolution_clock::now();
    // std::cout << std::endl;
    // duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    // std::cout << duration << std::endl;
    // std::cout << std::endl;
    // I/O.
    std::cout << "WRITING MATRIX" << std::endl;
    std::ofstream saveFile("test/matrix");
    t1 = std::chrono::high_resolution_clock::now();
    rnd.write(saveFile);
    t2 = std::chrono::high_resolution_clock::now();
    saveFile.close();
    // math::display(rnd);
    // End timing
    std::cout << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    // CUDA based dot product.
    std::cout << std::endl;
    // Begin timing.
    // toTranspose.at(0) = 3;
    // math::display(toTranspose);
    std::cout << "READING MATRIX" << std::endl;
    std::ifstream readFile("test/matrix");
    t1 = std::chrono::high_resolution_clock::now();
    rndRead.read(readFile);
    t2 = std::chrono::high_resolution_clock::now();
    readFile.close();
    // math::display(rndRead);
    // End timing
    std::cout << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    math::Matrix<double> rnd2 = math::Matrix<double>(1, 40);
    rnd2.randomizeUniform();
    // rnd2.transpose();
    // std::cout << "SUBTRACTION" << std::endl;
    // math::display(toTranspose);
    // math::display(toTranspose);
    // math::display(hadamardTest);
    std::cout << std::endl;
    std::cout << "HADAMARD PRODUCT" << std::endl;
    math::display(1 - toTranspose3.hadamard(toTranspose3));
    std::cout << std::endl;
    std::cout << "MATRIX MULTIPLICATION" << std::endl;
    t1 = std::chrono::high_resolution_clock::now();
    math::display(toTranspose * toTranspose);
    t2 = std::chrono::high_resolution_clock::now();
    // math::display(rndRead);
    // End timing
    std::cout << std::endl;
    duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << duration << std::endl;
    double someDouble = 2.0;
    math::display(toTranspose3);
    std::cout << std::endl;
    math::display(toTranspose3 * 2.0);
    std::cout << std::endl;
    math::display(toTranspose3);
    std::cout << std::endl;
    math::Matrix<double> newVec = math::Matrix<double>({1, 2, 3});
    std::cout << std::endl;
    math::display(newVec.transpose() * newVec);
    std::cout << std::endl;
    math::display(toTranspose);
    std::cout << std::endl;
    math::Matrix<double> rowMean = toTranspose.rowMean();
    math::display(rowMean);

}
