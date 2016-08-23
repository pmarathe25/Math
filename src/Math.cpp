#include "Math/Math.hpp"
#include <iostream>

namespace math {
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

    double fibonacci(double n) {
        return fibonacci((int) n);
    }

    double factorial(double operand) {
        int temp;
        // Set to 1 if the operand is less.
        temp = (operand < 1) ? 1 : operand;
        for (int i = 1; i < (int) operand; i++) {
            temp *= i;
        }
        return temp;
    }

    double divide(double operand1, double operand2) {
        return (operand1 / operand2);
    }

    double multiply(double operand1, double operand2) {
        return (operand1 * operand2);
    }

    double add(double operand1, double operand2) {
        return (operand1 + operand2);
    }

    double subtract(double operand1, double operand2) {
        return (operand1 - operand2);
    }
}
