#include "math.hpp"
#include <iostream>

int fibonacci(int n) {
    if (n < 2) {
        return n;
    } else {
        // Only need to keep track of the previous 2 numbers in the sequence.
        int prev[2] = {0, 1};
        int result = 0;
        for (int i = 1; i < n; i++) {
            result = prev[0] + prev[1];
            prev[0] = prev[1];
            prev[1] = result;
        }
        return result;
    }
}

double factorial(double operand) {
    int temp;
    // Set to 1 if the operand is less.
    temp = (operand < 1) ? 1 : operand;
    for (int i = 1; i < operand; i++) {
        temp *= i;
    }
    return temp;
}

double divide(std::pair<double, double> operands) {
    return (operands.first / operands.second);
}

double multiply(std::pair<double, double> operands) {
    return (operands.first * operands.second);
}

double add(std::pair<double, double> operands) {
    return (operands.first + operands.second);
}

double subtract(std::pair<double, double> operands) {
    return (operands.first - operands.second);
}
