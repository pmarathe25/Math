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
    std::cout << "========================Testing Matrix Creation.========================" << std::endl;
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
    std::cout << "========================Testing Matrix Copy========================" << std::endl;
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
    return 0;
}

int testMatrixReshape() {
    std::cout << "========================Testing Matrix Reshape========================" << std::endl;
    math::Matrix<float> reshapeTest0 = createSequentialMatrix<float>(5, 30);
    reshapeTest0.display();
    std::cout << "Reshaping 5x30 Matrix into 10x15" << std::endl;
    reshapeTest0.reshape(10, 15);
    reshapeTest0.display();
    return 0;
}

int testMatrixTranspose() {
    std::cout << "========================Testing Matrix Transpose========================" << std::endl;
    math::Matrix<float> transposeTest0 = createSequentialMatrix<float>(5, 30);
    transposeTest0.display();
    std::cout << "Transpose" << std::endl;
    transposeTest0.transpose().display();
    return 0;
}

int testMatrixRandomize() {
    std::cout << "========================Testing Matrix Randomize========================" << std::endl;
    std::cout << "Normal Randomization | Mean 0 | Standard Deviation 1" << std::endl;
    math::Matrix<float> randomizeTest0 = createSequentialMatrix<float>(5, 30);
    randomizeTest0 = randomNormalLike(randomizeTest0, 0, 1);
    randomizeTest0.display();
    std::cout << "Uniform Randomization | Range [0, 1]" << std::endl;
    randomizeTest0 = randomUniformLike(randomizeTest0, 0, 1);
    randomizeTest0.display();
    return 0;
}

int testRowMean() {
    std::cout << "========================Testing Matrix Row Mean========================" << std::endl;
    std::cout << "5x30 Sequential Matrix" << std::endl;
    math::Matrix<int> rowMeanTest0 = createSequentialMatrix<int>(5, 30);
    rowMeanTest0.display();
    std::cout << "Row Mean" << std::endl;
    rowMeanTest0.rowMean().display();
    return 0;
}

int testDotProduct() {
    std::cout << "========================Testing Row-Wise Dot Product========================" << std::endl;
    std::cout << "30x5 Ones Matrix" << std::endl;
    math::Matrix<int> dotTest0 = createOnesMatrix<int>(30, 5);
    dotTest0.display();
    std::cout << "Row-Wise Dot Product of Matrix with itself" << std::endl;
    dotTest0.dot(dotTest0).display();
    return 0;
}

int testMatrixMultiplication() {
    std::cout << "========================Testing Matrix Multiplication========================" << std::endl;
    std::cout << "10x5 Ones Matrix" << std::endl;
    math::Matrix<int> multiplicationTest0 = createOnesMatrix<int>(10, 5);
    multiplicationTest0.display();
    std::cout << "5x5 Sequential Matrix" << std::endl;
    math::Matrix<int> multiplicationTest1 = createSequentialMatrix<int>(5, 5);
    multiplicationTest1.display();
    std::cout << "Product" << std::endl;
    (multiplicationTest0 * multiplicationTest1).display();
    return 0;
}

int testMatrixArithmetic() {
    std::cout << "========================Testing Matrix Arithmetic========================" << std::endl;
    std::cout << "10x10 Ones Matrix" << std::endl;
    math::Matrix<float> arithmeticTest0 = createOnesMatrix<float>(10, 10);
    arithmeticTest0.display();
    std::cout << "10x10 Sequential Matrix" << std::endl;
    math::Matrix<float> arithmeticTest1 = createSequentialMatrix<float>(10, 10);
    arithmeticTest1.display();
    std::cout << "Sum" << std::endl;
    math::Matrix<float> sum = arithmeticTest0 + arithmeticTest1;
    sum.display();
    std::cout << "Difference" << std::endl;
    math::Matrix<float> difference = arithmeticTest0 - arithmeticTest1;
    difference.display();
    return 0;
}

int testMatrixVectorArithmetic() {
    std::cout << "========================Testing Matrix-Vector Arithmetic========================" << std::endl;
    std::cout << "10x10 Ones Matrix" << std::endl;
    math::Matrix<int> vectorArithmeticTest0 = createOnesMatrix<int>(10, 10);
    vectorArithmeticTest0.display();
    std::cout << "1x10 Sequential Column Vector" << std::endl;
    math::Matrix<int> vectorArithmeticTest1 = createSequentialMatrix<int>(10, 1);
    vectorArithmeticTest1.display();
    std::cout << "Matrix-Column Vector Addition" << std::endl;
    vectorArithmeticTest0.addVector(vectorArithmeticTest1).display();
    std::cout << "1x10 Sequential Row Vector" << std::endl;
    math::Matrix<int> vectorArithmeticTest2 = createSequentialMatrix<int>(1, 10);
    vectorArithmeticTest2.display();
    std::cout << "Matrix-Row Vector Addition" << std::endl;
    vectorArithmeticTest0.addVector(vectorArithmeticTest2).display();
    return 0;
}

int testMatrixScalarArithmetic() {
    std::cout << "========================Testing Matrix-Scalar Arithmetic========================" << std::endl;
    std::cout << "10x10 Ones Matrix" << std::endl;
    math::Matrix<int> scalarArithmeticTest0 = createOnesMatrix<int>(10, 10);
    scalarArithmeticTest0.display();
    std::cout << "Product with Scalar 2 (Right)" << std::endl;
    (scalarArithmeticTest0 * 2).display();
    std::cout << "Product with Scalar 2 (Left)" << std::endl;
    (2 * scalarArithmeticTest0).display();
    std::cout << "Sum with Scalar 2 (Right)" << std::endl;
    (scalarArithmeticTest0 + 2).display();
    std::cout << "Sum with Scalar 2 (Left)" << std::endl;
    (2 + scalarArithmeticTest0).display();
    std::cout << "Difference with Scalar 2 (Right)" << std::endl;
    (scalarArithmeticTest0 - 2).display();
    std::cout << "Difference with Scalar 2 (Left)" << std::endl;
    (2 - scalarArithmeticTest0).display();
    return 0;
}

int testMatrixHadamardProduct() {
    std::cout << "========================Testing Matrix Hadamard Product========================" << std::endl;
    std::cout << "10x10 Sequential Matrix" << std::endl;
    math::Matrix<int> hadamardProductTest0 = createSequentialMatrix<int>(10, 10);
    hadamardProductTest0.display();
    std::cout << "Matrix Hadamard Product with itself" << std::endl;
    hadamardProductTest0.hadamard(hadamardProductTest0).display();
    return 0;
}

__device__ double sigmoid(double a) {
    return 1 / (1 + exp(-a));
}

int testMatrixApplyFunction() {
    std::cout << "========================Testing Matrix Apply Function========================" << std::endl;
    std::cout << "10x10 Sequential Matrix" << std::endl;
    math::Matrix<double> applyFunctionTest0 = createSequentialMatrix<double>(10, 10) - 50;
    applyFunctionTest0.display();
    std::cout << "Applying sigmoid function" << std::endl;
    applyFunctionTest0.applyFunction<sigmoid>().display();
    return 0;
}

int main() {
    int numFailed = 0;
    numFailed += testMatrixCreation();
    numFailed += testMatrixCopy();
    numFailed += testMatrixReshape();
    numFailed += testMatrixTranspose();
    numFailed += testMatrixRandomize();
    numFailed += testRowMean();
    numFailed += testDotProduct();
    numFailed += testMatrixMultiplication();
    numFailed += testMatrixArithmetic();
    numFailed += testMatrixVectorArithmetic();
    numFailed += testMatrixScalarArithmetic();
    numFailed += testMatrixHadamardProduct();
    numFailed += testMatrixApplyFunction();
    if (numFailed == 0) {
        std::cout << "All Tests Passed." << std::endl;
    } else {
        std::cout << numFailed << " Tests Failed." << std::endl;
    }
}
