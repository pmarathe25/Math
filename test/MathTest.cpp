#include "Math/Matrix.hpp"
#include <iostream>
#include <chrono>

template <typename T>
math::Matrix<T> createOnesMatrix(int rows, int cols) {
    math::Matrix<T> output = math::Matrix<T>(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output.at(i, j) = 1;
        }
    }
    return output;
}

template <typename T>
math::Matrix<T> createSequentialMatrix(int rows, int cols) {
    math::Matrix<T> output = math::Matrix<T>(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output.at(i, j) = i * cols + j;
        }
    }
    return output;
}

int testMatrixCreation() {
    std::cout << "Testing Matrix Creation." << std::endl;
    std::cout << "Vector of Vectors Creation 2x3" << std::endl;
    math::Matrix<float> creationTest0 = math::Matrix<float>({{0, 1, 0}, {0, 2, 3}});
    creationTest0.display();
    std::cout << "Vector Creation 2x3" << std::endl;
    math::Matrix<float> creationTest1 = math::Matrix<float>({0, 1, 2, 3, 4, 5}, 2, 3);
    creationTest1.display();
    std::cout << "Empty Initialization with Assignment 40x40" << std::endl;
    math::Matrix<double> creationTest2 = createOnesMatrix<double>(40, 40);
    creationTest2.display();
    std::cout << "Empty Initialization with Assignment 5x30" << std::endl;
    math::Matrix<float> creationTest3 = createSequentialMatrix<float>(5, 30);
    creationTest3.display();
    std::cout << "Empty Initialization with Assignment 15x5" << std::endl;
    math::Matrix<float> creationTest4 = createSequentialMatrix<float>(15, 5);
    creationTest4.display();
    return 0;
}

int testMatrixCopy() {
    std::cout << "Testing Matrix Copy" << std::endl;
    std::cout << "Copying Matrices of the same type" << std::endl;
    math::Matrix<float> copyTest0 = createSequentialMatrix<float>(2, 10);
    math::Matrix<float> copyTest1 = copyTest0;
    std::cout << "Original" << std::endl;
    copyTest0.display();
    std::cout << "Copy" << std::endl;
    copyTest1.display();
    std::cout << "Modifying First Matrix (Should not affect second)" << std::endl;
    copyTest0.at(0) = 4056;
    std::cout << "Original" << std::endl;
    copyTest0.display();
    std::cout << "Copy" << std::endl;
    copyTest1.display();
    std::cout << "Copying Matrices of different types" << std::endl;
    math::Matrix<double> copyTest2 = createSequentialMatrix<double>(2, 10);
    math::Matrix<int> copyTest3 = copyTest2;
    std::cout << "Original" << std::endl;
    copyTest2.display();
    std::cout << "Copy" << std::endl;
    copyTest3.display();
    std::cout << "Modifying First Matrix (Should not affect second)" << std::endl;
    copyTest2.at(0) = 4056;
    std::cout << "Original" << std::endl;
    copyTest2.display();
    std::cout << "Copy" << std::endl;
    copyTest3.display();
    return 0;
}

int testMatrixReshape() {
    std::cout << "Testing Matrix Reshape" << std::endl;
    math::Matrix<float> reshapeTest0 = createSequentialMatrix<float>(5, 30);
    reshapeTest0.display();
    std::cout << "Reshaping 5x30 Matrix into 10x15" << std::endl;
    reshapeTest0.reshape(10, 15);
    reshapeTest0.display();
    return 0;
}

int testMatrixTranspose() {
    std::cout << "Testing Matrix Transpose" << std::endl;
    math::Matrix<float> transposeTest0 = createSequentialMatrix<float>(5, 30);
    transposeTest0.display();
    std::cout << "Transpose" << std::endl;
    transposeTest0.transpose().display();
    return 0;
}

int testMatrixRandomize() {
    std::cout << "Testing Matrix Randomize" << std::endl;
    std::cout << "Normal Randomization | Mean 0 | Standard Deviation 1" << std::endl;
    math::Matrix<float> randomizeTest0 = createSequentialMatrix<float>(5, 30);
    randomizeTest0.randomizeNormal(0, 1);
    randomizeTest0.display();
    std::cout << "Uniform Randomization | Range [0, 1]" << std::endl;
    randomizeTest0.randomizeUniform(0, 1);
    randomizeTest0.display();
    return 0;
}

int testRowMean() {
    std::cout << "Testing Matrix Row Mean" << std::endl;
    std::cout << "5x30 Sequential Matrix" << std::endl;
    math::Matrix<int> rowMeanTest0 = createSequentialMatrix<int>(5, 30);
    rowMeanTest0.display();
    std::cout << "Row Mean" << std::endl;
    rowMeanTest0.rowMean().display();
    return 0;
}

int testRowWiseDot() {
    std::cout << "Testing Row-Wise Dot Product" << std::endl;
    std::cout << "30x5 Ones Matrix" << std::endl;
    math::Matrix<int> rowWiseDotTest0 = createOnesMatrix<int>(30, 5);
    rowWiseDotTest0.display();
    std::cout << "Row-Wise Dot Product of Matrix with itself" << std::endl;
    rowWiseDotTest0.rowWiseDot(rowWiseDotTest0).display();
    return 0;
}

int main() {
    int success = 0;
    success += testMatrixCreation();
    success += testMatrixCopy();
    success += testMatrixReshape();
    success += testMatrixTranspose();
    success += testMatrixRandomize();
    success += testRowMean();
    success += testRowWiseDot();
}
