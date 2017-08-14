#ifndef MATH_H
#define MATH_H
#include <vector>
#include <iostream>

const int BLOCK_DIM = 32;
const int THREADS_PER_BLOCK = 1024;

namespace StealthMath {
    int fibonacci(int n);
    double factorial(double operand);
    double divide(double operand1, double operand2);
    double multiply(double operand1, double operand2);
    double add(double operand1, double operand2);
    double subtract(double operand1, double operand2);
}

#endif
