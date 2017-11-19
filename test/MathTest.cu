#include "StealthMatrix.hpp"
#include <string>
#include <iostream>
#include <chrono>

// int testStealthMatrixCreation() {
//     std::cout << "========================Testing StealthMatrix Creation.========================" << '\n';
//     StealthMatrix_F creationTest0 = StealthMatrix_F({0, 1, 0, 0, 2, 3}, 2);
//     creationTest0.display("Vector Creation 2x3 (Implicit columns)");
//     StealthMatrix_F creationTest1 = StealthMatrix_F({0, 1, 2, 3, 4, 5}, 2, 3);
//     creationTest1.display("Vector Creation 2x3");
//     StealthMatrix_D creationTest2 = StealthMatrix_D::ones(40, 40);
//     creationTest2.display("Empty Initialization with Assignment 40x40");
//     StealthMatrix_F creationTest3 = StealthMatrix_F::sequential(5, 30);
//     creationTest3.display("Empty Initialization with Assignment 5x30");
//     StealthMatrix_F creationTest4 = StealthMatrix_F::sequential(15, 5);
//     creationTest4.display("Empty Initialization with Assignment 15x5");
//     StealthMatrix_F creationTest5(15, 5);
//     creationTest5.display("Empty Initialization 15x5");
//     StealthMatrix_C creationTest6(30, 10);
//     creationTest6.display("Empty Initialization 15x5");
//     return 0;
// }
//
// int testStealthMatrixCopy() {
//     std::cout << "========================Testing StealthMatrix Copy========================" << '\n';
//     std::cout << "Copying Matrices of the same type" << '\n';
//     StealthMatrix_F copyTest0 = StealthMatrix_F::sequential(2, 10);
//     StealthMatrix_F copyTest1 = copyTest0;
//     copyTest0.display("Original");
//     copyTest1.display("Copy");
//     std::cout << "Modifying First StealthMatrix (Should not affect second)" << '\n';
//     copyTest0.at(0) = 4056;
//     copyTest0.display("Original");
//     copyTest1.display("Copy");
//     return 0;
// }
//
// int testStealthMatrixReshape() {
//     std::cout << "========================Testing StealthMatrix Reshape========================" << '\n';
//     StealthMatrix_F reshapeTest0 = StealthMatrix_F::sequential(5, 30);
//     reshapeTest0.display("5x30 Sequential StealthMatrix");
//     reshapeTest0.reshape(10, 15);
//     reshapeTest0.display("Reshaping 5x30 StealthMatrix into 10x15");
//     return 0;
// }
//
// int testStealthMatrixTranspose() {
//     std::cout << "========================Testing StealthMatrix Transpose========================" << '\n';
//     StealthMatrix_F transposeTest0 = StealthMatrix_F::sequential(5, 30);
//     transposeTest0.display("5x30 Sequential StealthMatrix");
//     transposeTest0.transpose().display("Transpose");
//     return 0;
// }
//
// int testStealthMatrixRandomize() {
//     std::cout << "========================Testing StealthMatrix Randomize========================" << '\n';
//     StealthMatrix_F randomizeTest0 = StealthMatrix_F::sequential(5, 30);
//     randomizeTest0 = StealthMatrix_F::randomNormalLike(randomizeTest0, 0.0, 1.0);
//     randomizeTest0.display("Normal Randomization | Mean 0 | Standard Deviation 1");
//     randomizeTest0 = StealthMatrix_F::randomUniformLike(randomizeTest0, 0.0, 1.0);
//     randomizeTest0.display("Uniform Randomization | Range [0, 1]");
//     return 0;
// }
//
// int testweightedSum() {
//     std::cout << "========================Testing StealthMatrix Weighted Row Sum========================" << '\n';
//     StealthMatrix weightedSumTest0 = StealthMatrix::sequential(5, 30);
//     weightedSumTest0.display("5x30 Sequential StealthMatrix");
//     weightedSumTest0.rowMean().display("Row Mean");
//     weightedSumTest0.weightedSum().display("Row Sum");
//     weightedSumTest0.weightedSum(0).display("Column Sum");
//     weightedSumTest0.columnMean().display("Column Mean");
//     return 0;
// }
//
// int testDotProduct() {
//     std::cout << "========================Testing Row-Wise Dot Product========================" << '\n';
//     StealthMatrix dotTest0 = StealthMatrix::ones(30, 5);
//     dotTest0.display("30x5 Ones StealthMatrix");
//     dotTest0.dot(dotTest0).display("Row-Wise Dot Product of StealthMatrix with itself");
//     return 0;
// }
//
// int testStealthMatrixMultiplication() {
//     std::cout << "========================Testing StealthMatrix Multiplication========================" << '\n';
//     StealthMatrix multiplicationTest0 = StealthMatrix::ones(10, 5);
//     multiplicationTest0.display("10x5 Ones StealthMatrix");
//     StealthMatrix multiplicationTest1 = StealthMatrix::sequential(5, 5);
//     multiplicationTest1.display("5x5 Sequential StealthMatrix");
//     (multiplicationTest0 * multiplicationTest1).display("Product");
//     return 0;
// }
//
// int testStealthMatrixArithmetic() {
//     std::cout << "========================Testing StealthMatrix-StealthMatrix Arithmetic========================" << '\n';
//     StealthMatrix_F arithmeticTest0 = StealthMatrix_F::ones(10, 10);
//     arithmeticTest0.display("10x10 Ones StealthMatrix");
//     StealthMatrix_F arithmeticTest1 = StealthMatrix_F::sequential(10, 10);
//     arithmeticTest1.display("10x10 Sequential StealthMatrix");
//     StealthMatrix_F sum = arithmeticTest0 + arithmeticTest1;
//     sum.display("Sum");
//     StealthMatrix_F difference = arithmeticTest0 - arithmeticTest1;
//     difference.display("Difference");
//     arithmeticTest0 += arithmeticTest1;
//     arithmeticTest0.display("Sum in place");
//     arithmeticTest0 -= arithmeticTest1;
//     arithmeticTest0.display("Difference in place");
//     return 0;
// }
//
// int testStealthMatrixVectorArithmetic() {
//     std::cout << "========================Testing StealthMatrix-Vector Arithmetic========================" << '\n';
//     StealthMatrix vectorArithmeticTest0 = StealthMatrix::ones(10, 10);
//     vectorArithmeticTest0.display("10x10 Ones StealthMatrix");
//     StealthMatrix vectorArithmeticTest1 = StealthMatrix::sequential(10, 1);
//     vectorArithmeticTest1.display("1x10 Sequential Column Vector");
//     vectorArithmeticTest0.addVector(vectorArithmeticTest1).display("StealthMatrix-Column Vector Addition");
//     StealthMatrix vectorArithmeticTest2 = StealthMatrix::sequential(1, 10);
//     vectorArithmeticTest2.display("1x10 Sequential Row Vector");
//     vectorArithmeticTest0.addVector(vectorArithmeticTest2).display("StealthMatrix-Row Vector Addition");
//     return 0;
// }
//
// int testStealthMatrixScalarArithmetic() {
//     std::cout << "========================Testing StealthMatrix-Scalar Arithmetic========================" << '\n';
//     StealthMatrix_F scalarArithmeticTest0 = StealthMatrix_F::ones(10, 10);
//     scalarArithmeticTest0.display("10x10 Ones StealthMatrix");
//     (scalarArithmeticTest0 / 2).display("Division with Scalar 2 (Right)");
//     (scalarArithmeticTest0 * 2).display("Product with Scalar 2 (Right)");
//     (2 * scalarArithmeticTest0).display("Product with Scalar 2 (Left)");
//     (scalarArithmeticTest0 + 2).display("Sum with Scalar 2 (Right)");
//     (2 + scalarArithmeticTest0).display("Sum with Scalar 2 (Left)");
//     (scalarArithmeticTest0 - 2).display("Difference with Scalar 2 (Right)");
//     (2 - scalarArithmeticTest0).display("Difference with Scalar 2 (Left)");
//     return 0;
// }
//
// int testStealthMatrixHadamardProduct() {
//     std::cout << "========================Testing StealthMatrix Hadamard Product========================" << '\n';
//     StealthMatrix hadamardProductTest0 = StealthMatrix::sequential(10, 10);
//     hadamardProductTest0.display("10x10 Sequential StealthMatrix");
//     hadamardProductTest0.hadamard(hadamardProductTest0).display("StealthMatrix Hadamard Product with itself");
//     return 0;
// }
//
// __device__ double sigmoid(double a) {
//     return 1 / (1 + exp(-a));
// }
//
// int testStealthMatrixApplyFunction() {
//     std::cout << "========================Testing StealthMatrix Apply Function========================" << '\n';
//     StealthMatrix_D applyFunctionTest0 = StealthMatrix_D::sequential(10, 10) - 50;
//     applyFunctionTest0.display("10x10 Sequential StealthMatrix");
//     applyFunctionTest0.applyFunction<sigmoid>().display("Applying sigmoid function");
//     return 0;
// }
//
// int testStealthMatrixPower() {
//     std::cout << "========================Testing StealthMatrix Power========================" << '\n';
//     StealthMatrix_D powerTest0 = StealthMatrix_D::sequential(10, 10);
//     powerTest0.display("10x10 Sequential StealthMatrix");
//     powerTest0.pow(2).display("Computing square");
//     powerTest0.pow(3).display("Computing cube");
//     return 0;
// }
//
// int testStealthMatrixFileIO() {
//     std::string filePath = "./test/matrix.bin";
//     std::cout << "========================Testing StealthMatrix File IO========================" << '\n';
//     StealthMatrix_D fileIOTest0 = StealthMatrix_D::randomNormal(6, 8);
//     fileIOTest0.display("6x8 Random Normal StealthMatrix");
//     std::cout << "Writing matrix" << '\n';
//     fileIOTest0.save(filePath);
//     std::cout << "Loading matrix" << '\n';
//     StealthMatrix_D fileIOTest1(filePath);
//     fileIOTest1.display("Loaded matrix of dimensions " + std::to_string(fileIOTest1.numRows()) + "x" + std::to_string(fileIOTest1.numColumns()));
//     return 0;
// }
//
// int testStealthMatrixTypeCasting() {
//     std::cout << "========================Testing StealthMatrix Type Casting========================" << '\n';
//     StealthMatrix_D typeCastingTest0 = StealthMatrix_D::randomNormal(4, 4);
//     typeCastingTest0.display("4x4 Random Normal StealthMatrix (double)");
//     StealthMatrix typeCastingTest1 = typeCastingTest0.asType<int>();
//     typeCastingTest1.display("Above StealthMatrix as type int");
//     StealthMatrix_C typeCastingTest2 = StealthMatrix_C::randomUniform(4, 4, 65, 91);
//     typeCastingTest2.display("4x4 Random Uniform StealthMatrix (char)");
//     StealthMatrix_F typeCastingTest3 = typeCastingTest2.asType<float>();
//     typeCastingTest3.display("Above StealthMatrix as type float");
//     StealthMatrix typeCastingTest4 = typeCastingTest2.asType<int>();
//     typeCastingTest4.display("Above StealthMatrix as type int");
//     return 0;
// }
//
// int testStealthMatrixArgmax() {
//     std::cout << "========================Testing StealthMatrix Argmax========================" << '\n';
//     StealthMatrix_F argmaxTest0 = StealthMatrix_F::sequential(10, 10);
//     argmaxTest0.argmax().display("Sequential Matrix Argmax");
//     return 0;
// }
//
// int testStealthMatrixMaxMask() {
//     std::cout << "========================Testing StealthMatrix Argmax========================" << '\n';
//     StealthMatrix_F maxMaskTest0 = StealthMatrix_F::sequential(10, 10);
//     maxMaskTest0.maxMask().display("Sequential Matrix Argmax");
//     return 0;
// }

int testStealthMatrixCreation() {
    StealthMath::StealthMatrix<float, 10, 10> mat{};
    mat.at(0, 0) = 1.5;

    StealthMath::StealthMatrix<float, 10, 10> mat2{};
    mat2 = mat;

    display(mat, "This matrix should equal...");
    display(mat2, "...this one");
    return 0;
}

int main() {
    int numFailed = 0;
    numFailed += testStealthMatrixCreation();
    // numFailed += testStealthMatrixCopy();
    // numFailed += testStealthMatrixReshape();
    // numFailed += testStealthMatrixTranspose();
    // numFailed += testStealthMatrixRandomize();
    // numFailed += testweightedSum();
    // numFailed += testDotProduct();
    // numFailed += testStealthMatrixMultiplication();
    // numFailed += testStealthMatrixArithmetic();
    // numFailed += testStealthMatrixVectorArithmetic();
    // numFailed += testStealthMatrixScalarArithmetic();
    // numFailed += testStealthMatrixHadamardProduct();
    // numFailed += testStealthMatrixApplyFunction();
    // numFailed += testStealthMatrixPower();
    // numFailed += testStealthMatrixFileIO();
    // numFailed += testStealthMatrixTypeCasting();
    // numFailed += testStealthMatrixArgmax();
    // numFailed += testStealthMatrixMaxMask();

    std::cout << '\n';
    if (numFailed == 0) {
        std::cout << "All Tests Passed." << '\n';
    } else {
        std::cout << numFailed << " Tests Failed." << '\n';
    }
}
