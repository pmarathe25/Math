#include "Matrix.hpp"
#include <string>
#include <iostream>
#include <chrono>

int testMatrixCreation() {
    std::cout << "========================Testing Matrix Creation.========================" << '\n';
    std::cout << "Vector of Vectors Creation 2x3" << '\n';
    Matrix_F creationTest0 = Matrix_F({{0, 1, 0}, {0, 2, 3}});
    creationTest0.display();
    std::cout << "Vector Creation 2x3" << '\n';
    Matrix_F creationTest1 = Matrix_F({0, 1, 2, 3, 4, 5}, 2, 3);
    creationTest1.display();
    std::cout << "Empty Initialization with Assignment 40x40" << '\n';
    Matrix_D creationTest2 = Matrix_D::ones(40, 40);
    creationTest2.display();
    std::cout << "Empty Initialization with Assignment 5x30" << '\n';
    Matrix_F creationTest3 = Matrix_F::sequentialMatrix(5, 30);
    creationTest3.display();
    std::cout << "Empty Initialization with Assignment 15x5" << '\n';
    Matrix_F creationTest4 = Matrix_F::sequentialMatrix(15, 5);
    creationTest4.display();
    return 0;
}

int testMatrixCopy() {
    std::cout << "========================Testing Matrix Copy========================" << '\n';
    std::cout << "Copying Matrices of the same type" << '\n';
    Matrix_F copyTest0 = Matrix_F::sequentialMatrix(2, 10);
    Matrix_F copyTest1 = copyTest0;
    std::cout << "Original" << '\n';
    copyTest0.display();
    std::cout << "Copy" << '\n';
    copyTest1.display();
    std::cout << "Modifying First Matrix (Should not affect second)" << '\n';
    copyTest0.at(0) = 4056;
    std::cout << "Original" << '\n';
    copyTest0.display();
    std::cout << "Copy" << '\n';
    copyTest1.display();
    return 0;
}

int testMatrixReshape() {
    std::cout << "========================Testing Matrix Reshape========================" << '\n';
    Matrix_F reshapeTest0 = Matrix_F::sequentialMatrix(5, 30);
    reshapeTest0.display();
    std::cout << "Reshaping 5x30 Matrix into 10x15" << '\n';
    reshapeTest0.reshape(10, 15);
    reshapeTest0.display();
    return 0;
}

int testMatrixTranspose() {
    std::cout << "========================Testing Matrix Transpose========================" << '\n';
    Matrix_F transposeTest0 = Matrix_F::sequentialMatrix(5, 30);
    transposeTest0.display();
    std::cout << "Transpose" << '\n';
    transposeTest0.transpose().display();
    return 0;
}

int testMatrixRandomize() {
    std::cout << "========================Testing Matrix Randomize========================" << '\n';
    std::cout << "Normal Randomization | Mean 0 | Standard Deviation 1" << '\n';
    Matrix_F randomizeTest0 = Matrix_F::sequentialMatrix(5, 30);
    randomizeTest0 = Matrix_F::randomNormalLike(randomizeTest0, 0.0, 1.0);
    randomizeTest0.display();
    std::cout << "Uniform Randomization | Range [0, 1]" << '\n';
    randomizeTest0 = Matrix_F::randomUniformLike(randomizeTest0, 0.0, 1.0);
    randomizeTest0.display();
    return 0;
}

int testRowMean() {
    std::cout << "========================Testing Matrix Row Mean========================" << '\n';
    std::cout << "5x30 Sequential Matrix" << '\n';
    Matrix rowMeanTest0 = Matrix::sequentialMatrix(5, 30);
    rowMeanTest0.display();
    std::cout << "Row Mean" << '\n';
    rowMeanTest0.rowMean().display();
    return 0;
}

int testDotProduct() {
    std::cout << "========================Testing Row-Wise Dot Product========================" << '\n';
    std::cout << "30x5 Ones Matrix" << '\n';
    Matrix dotTest0 = Matrix::ones(30, 5);
    dotTest0.display();
    std::cout << "Row-Wise Dot Product of Matrix with itself" << '\n';
    dotTest0.dot(dotTest0).display();
    return 0;
}

int testMatrixMultiplication() {
    std::cout << "========================Testing Matrix Multiplication========================" << '\n';
    std::cout << "10x5 Ones Matrix" << '\n';
    Matrix multiplicationTest0 = Matrix::ones(10, 5);
    multiplicationTest0.display();
    std::cout << "5x5 Sequential Matrix" << '\n';
    Matrix multiplicationTest1 = Matrix::sequentialMatrix(5, 5);
    multiplicationTest1.display();
    std::cout << "Product" << '\n';
    (multiplicationTest0 * multiplicationTest1).display();
    return 0;
}

int testMatrixArithmetic() {
    std::cout << "========================Testing Matrix-Matrix Arithmetic========================" << '\n';
    std::cout << "10x10 Ones Matrix" << '\n';
    Matrix_F arithmeticTest0 = Matrix_F::ones(10, 10);
    arithmeticTest0.display();
    std::cout << "10x10 Sequential Matrix" << '\n';
    Matrix_F arithmeticTest1 = Matrix_F::sequentialMatrix(10, 10);
    arithmeticTest1.display();
    std::cout << "Sum" << '\n';
    Matrix_F sum = arithmeticTest0 + arithmeticTest1;
    sum.display();
    std::cout << "Difference" << '\n';
    Matrix_F difference = arithmeticTest0 - arithmeticTest1;
    difference.display();
    std::cout << "Sum in place" << '\n';
    arithmeticTest0 += arithmeticTest1;
    arithmeticTest0.display();
    std::cout << "Difference in place" << '\n';
    arithmeticTest0 -= arithmeticTest1;
    arithmeticTest0.display();
    return 0;
}

int testMatrixVectorArithmetic() {
    std::cout << "========================Testing Matrix-Vector Arithmetic========================" << '\n';
    std::cout << "10x10 Ones Matrix" << '\n';
    Matrix vectorArithmeticTest0 = Matrix::ones(10, 10);
    vectorArithmeticTest0.display();
    std::cout << "1x10 Sequential Column Vector" << '\n';
    Matrix vectorArithmeticTest1 = Matrix::sequentialMatrix(10, 1);
    vectorArithmeticTest1.display();
    std::cout << "Matrix-Column Vector Addition" << '\n';
    vectorArithmeticTest0.addVector(vectorArithmeticTest1).display();
    std::cout << "1x10 Sequential Row Vector" << '\n';
    Matrix vectorArithmeticTest2 = Matrix::sequentialMatrix(1, 10);
    vectorArithmeticTest2.display();
    std::cout << "Matrix-Row Vector Addition" << '\n';
    vectorArithmeticTest0.addVector(vectorArithmeticTest2).display();
    return 0;
}

int testMatrixScalarArithmetic() {
    std::cout << "========================Testing Matrix-Scalar Arithmetic========================" << '\n';
    std::cout << "10x10 Ones Matrix" << '\n';
    Matrix_F scalarArithmeticTest0 = Matrix_F::ones(10, 10);
    scalarArithmeticTest0.display();
    std::cout << "Division with Scalar 2 (Right)" << '\n';
    (scalarArithmeticTest0 / 2).display();
    std::cout << "Product with Scalar 2 (Right)" << '\n';
    (scalarArithmeticTest0 * 2).display();
    std::cout << "Product with Scalar 2 (Left)" << '\n';
    (2 * scalarArithmeticTest0).display();
    std::cout << "Sum with Scalar 2 (Right)" << '\n';
    (scalarArithmeticTest0 + 2).display();
    std::cout << "Sum with Scalar 2 (Left)" << '\n';
    (2 + scalarArithmeticTest0).display();
    std::cout << "Difference with Scalar 2 (Right)" << '\n';
    (scalarArithmeticTest0 - 2).display();
    std::cout << "Difference with Scalar 2 (Left)" << '\n';
    (2 - scalarArithmeticTest0).display();
    return 0;
}

int testMatrixHadamardProduct() {
    std::cout << "========================Testing Matrix Hadamard Product========================" << '\n';
    std::cout << "10x10 Sequential Matrix" << '\n';
    Matrix hadamardProductTest0 = Matrix::sequentialMatrix(10, 10);
    hadamardProductTest0.display();
    std::cout << "Matrix Hadamard Product with itself" << '\n';
    hadamardProductTest0.hadamard(hadamardProductTest0).display();
    return 0;
}

__device__ double sigmoid(double a) {
    return 1 / (1 + exp(-a));
}

int testMatrixApplyFunction() {
    std::cout << "========================Testing Matrix Apply Function========================" << '\n';
    std::cout << "10x10 Sequential Matrix" << '\n';
    Matrix_D applyFunctionTest0 = Matrix_D::sequentialMatrix(10, 10) - 50;
    applyFunctionTest0.display();
    std::cout << "Applying sigmoid function" << '\n';
    applyFunctionTest0.applyFunction<sigmoid>().display();
    return 0;
}

int testMatrixPower() {
    std::cout << "========================Testing Matrix Power========================" << '\n';
    std::cout << "10x10 Sequential Matrix" << '\n';
    Matrix_D powerTest0 = Matrix_D::sequentialMatrix(10, 10);
    powerTest0.display();
    std::cout << "Computing square" << '\n';
    math::pow(powerTest0, 2).display();
    std::cout << "Computing cube" << '\n';
    math::pow(powerTest0, 3).display();
    return 0;
}

int testMatrixFileIO() {
    std::string filePath = "./test/matrix";
    std::cout << "========================Testing Matrix File IO========================" << '\n';
    std::cout << "100x100 Random Normal Matrix" << '\n';
    Matrix_D fileIOTest0 = Matrix_D::randomNormal(6, 8);
    fileIOTest0.display();
    std::cout << "Writing matrix" << '\n';
    fileIOTest0.save(filePath);
    std::cout << "Loading matrix" << '\n';
    Matrix_D fileIOTest1(filePath);
    std::cout << "Loaded matrix of dimensions " << fileIOTest1.numRows() << "x" << fileIOTest1.numColumns() << '\n';
    fileIOTest1.display();
    std::cout << '\n';
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
    numFailed += testMatrixPower();
    numFailed += testMatrixFileIO();
    if (numFailed == 0) {
        std::cout << "All Tests Passed." << '\n';
    } else {
        std::cout << numFailed << " Tests Failed." << '\n';
    }
}
