#include "Matrix.hpp"
#include <string>
#include <iostream>
#include <chrono>

int testMatrixCreation() {
    std::cout << "========================Testing Matrix Creation.========================" << '\n';
    Matrix_F creationTest0 = Matrix_F({0, 1, 0, 0, 2, 3}, 2);
    creationTest0.display("Vector Creation 2x3 (Implicit columns)");
    Matrix_F creationTest1 = Matrix_F({0, 1, 2, 3, 4, 5}, 2, 3);
    creationTest1.display("Vector Creation 2x3");
    Matrix_D creationTest2 = Matrix_D::ones(40, 40);
    creationTest2.display("Empty Initialization with Assignment 40x40");
    Matrix_F creationTest3 = Matrix_F::sequential(5, 30);
    creationTest3.display("Empty Initialization with Assignment 5x30");
    Matrix_F creationTest4 = Matrix_F::sequential(15, 5);
    creationTest4.display("Empty Initialization with Assignment 15x5");
    return 0;
}

int testMatrixCopy() {
    std::cout << "========================Testing Matrix Copy========================" << '\n';
    std::cout << "Copying Matrices of the same type" << '\n';
    Matrix_F copyTest0 = Matrix_F::sequential(2, 10);
    Matrix_F copyTest1 = copyTest0;
    copyTest0.display("Original");
    copyTest1.display("Copy");
    std::cout << "Modifying First Matrix (Should not affect second)" << '\n';
    copyTest0.at(0) = 4056;
    copyTest0.display("Original");
    copyTest1.display("Copy");
    return 0;
}

int testMatrixReshape() {
    std::cout << "========================Testing Matrix Reshape========================" << '\n';
    Matrix_F reshapeTest0 = Matrix_F::sequential(5, 30);
    reshapeTest0.display("5x30 Sequential Matrix");
    reshapeTest0.reshape(10, 15);
    reshapeTest0.display("Reshaping 5x30 Matrix into 10x15");
    return 0;
}

int testMatrixTranspose() {
    std::cout << "========================Testing Matrix Transpose========================" << '\n';
    Matrix_F transposeTest0 = Matrix_F::sequential(5, 30);
    transposeTest0.display("5x30 Sequential Matrix");
    transposeTest0.transpose().display("Transpose");
    return 0;
}

int testMatrixRandomize() {
    std::cout << "========================Testing Matrix Randomize========================" << '\n';
    Matrix_F randomizeTest0 = Matrix_F::sequential(5, 30);
    randomizeTest0 = Matrix_F::randomNormalLike(randomizeTest0, 0.0, 1.0);
    randomizeTest0.display("Normal Randomization | Mean 0 | Standard Deviation 1");
    randomizeTest0 = Matrix_F::randomUniformLike(randomizeTest0, 0.0, 1.0);
    randomizeTest0.display("Uniform Randomization | Range [0, 1]");
    return 0;
}

int testWeightedSum() {
    std::cout << "========================Testing Matrix Weighted Sum========================" << '\n';
    Matrix weightedSumTest0 = Matrix::sequential(5, 30);
    weightedSumTest0.display("5x30 Sequential Matrix");
    weightedSumTest0.weightedSum(1 / (float) weightedSumTest0.numRows()).display("Row Mean");
    weightedSumTest0.weightedSum().display("Row Sum");
    return 0;
}

int testDotProduct() {
    std::cout << "========================Testing Row-Wise Dot Product========================" << '\n';
    Matrix dotTest0 = Matrix::ones(30, 5);
    dotTest0.display("30x5 Ones Matrix");
    dotTest0.dot(dotTest0).display("Row-Wise Dot Product of Matrix with itself");
    return 0;
}

int testMatrixMultiplication() {
    std::cout << "========================Testing Matrix Multiplication========================" << '\n';
    Matrix multiplicationTest0 = Matrix::ones(10, 5);
    multiplicationTest0.display("10x5 Ones Matrix");
    Matrix multiplicationTest1 = Matrix::sequential(5, 5);
    multiplicationTest1.display("5x5 Sequential Matrix");
    (multiplicationTest0 * multiplicationTest1).display("Product");
    return 0;
}

int testMatrixArithmetic() {
    std::cout << "========================Testing Matrix-Matrix Arithmetic========================" << '\n';
    Matrix_F arithmeticTest0 = Matrix_F::ones(10, 10);
    arithmeticTest0.display("10x10 Ones Matrix");
    Matrix_F arithmeticTest1 = Matrix_F::sequential(10, 10);
    arithmeticTest1.display("10x10 Sequential Matrix");
    Matrix_F sum = arithmeticTest0 + arithmeticTest1;
    sum.display("Sum");
    Matrix_F difference = arithmeticTest0 - arithmeticTest1;
    difference.display("Difference");
    arithmeticTest0 += arithmeticTest1;
    arithmeticTest0.display("Sum in place");
    arithmeticTest0 -= arithmeticTest1;
    arithmeticTest0.display("Difference in place");
    return 0;
}

int testMatrixVectorArithmetic() {
    std::cout << "========================Testing Matrix-Vector Arithmetic========================" << '\n';
    Matrix vectorArithmeticTest0 = Matrix::ones(10, 10);
    vectorArithmeticTest0.display("10x10 Ones Matrix");
    Matrix vectorArithmeticTest1 = Matrix::sequential(10, 1);
    vectorArithmeticTest1.display("1x10 Sequential Column Vector");
    vectorArithmeticTest0.addVector(vectorArithmeticTest1).display("Matrix-Column Vector Addition");
    Matrix vectorArithmeticTest2 = Matrix::sequential(1, 10);
    vectorArithmeticTest2.display("1x10 Sequential Row Vector");
    vectorArithmeticTest0.addVector(vectorArithmeticTest2).display("Matrix-Row Vector Addition");
    return 0;
}

int testMatrixScalarArithmetic() {
    std::cout << "========================Testing Matrix-Scalar Arithmetic========================" << '\n';
    Matrix_F scalarArithmeticTest0 = Matrix_F::ones(10, 10);
    scalarArithmeticTest0.display("10x10 Ones Matrix");
    (scalarArithmeticTest0 / 2).display("Division with Scalar 2 (Right)");
    (scalarArithmeticTest0 * 2).display("Product with Scalar 2 (Right)");
    (2 * scalarArithmeticTest0).display("Product with Scalar 2 (Left)");
    (scalarArithmeticTest0 + 2).display("Sum with Scalar 2 (Right)");
    (2 + scalarArithmeticTest0).display("Sum with Scalar 2 (Left)");
    (scalarArithmeticTest0 - 2).display("Difference with Scalar 2 (Right)");
    (2 - scalarArithmeticTest0).display("Difference with Scalar 2 (Left)");
    return 0;
}

int testMatrixHadamardProduct() {
    std::cout << "========================Testing Matrix Hadamard Product========================" << '\n';
    Matrix hadamardProductTest0 = Matrix::sequential(10, 10);
    hadamardProductTest0.display("10x10 Sequential Matrix");
    hadamardProductTest0.hadamard(hadamardProductTest0).display("Matrix Hadamard Product with itself");
    return 0;
}

__device__ double sigmoid(double a) {
    return 1 / (1 + exp(-a));
}

int testMatrixApplyFunction() {
    std::cout << "========================Testing Matrix Apply Function========================" << '\n';
    Matrix_D applyFunctionTest0 = Matrix_D::sequential(10, 10) - 50;
    applyFunctionTest0.display("10x10 Sequential Matrix");
    applyFunctionTest0.applyFunction<sigmoid>().display("Applying sigmoid function");
    return 0;
}

int testMatrixPower() {
    std::cout << "========================Testing Matrix Power========================" << '\n';
    Matrix_D powerTest0 = Matrix_D::sequential(10, 10);
    powerTest0.display("10x10 Sequential Matrix");
    powerTest0.pow(2).display("Computing square");
    powerTest0.pow(3).display("Computing cube");
    return 0;
}

int testMatrixFileIO() {
    std::string filePath = "./test/matrix.bin";
    std::cout << "========================Testing Matrix File IO========================" << '\n';
    Matrix_D fileIOTest0 = Matrix_D::randomNormal(6, 8);
    fileIOTest0.display("100x100 Random Normal Matrix");
    std::cout << "Writing matrix" << '\n';
    fileIOTest0.save(filePath);
    std::cout << "Loading matrix" << '\n';
    Matrix_D fileIOTest1(filePath);
    fileIOTest1.display("Loaded matrix of dimensions " + std::to_string(fileIOTest1.numRows()) + "x" + std::to_string(fileIOTest1.numColumns()));
    return 0;
}

int main() {
    int numFailed = 0;
    numFailed += testMatrixCreation();
    numFailed += testMatrixCopy();
    numFailed += testMatrixReshape();
    numFailed += testMatrixTranspose();
    numFailed += testMatrixRandomize();
    numFailed += testWeightedSum();
    numFailed += testDotProduct();
    numFailed += testMatrixMultiplication();
    numFailed += testMatrixArithmetic();
    numFailed += testMatrixVectorArithmetic();
    numFailed += testMatrixScalarArithmetic();
    numFailed += testMatrixHadamardProduct();
    numFailed += testMatrixApplyFunction();
    numFailed += testMatrixPower();
    numFailed += testMatrixFileIO();
    std::cout << '\n';
    if (numFailed == 0) {
        std::cout << "All Tests Passed." << '\n';
    } else {
        std::cout << numFailed << " Tests Failed." << '\n';
    }
}
