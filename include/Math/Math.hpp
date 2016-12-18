#ifndef MATH_H
#define MATH_H
#include <vector>
#include <iostream>

namespace math {
    int fibonacci(int n);
    double fibonacci(double n);
    double factorial(double operand);
    double divide(double operand1, double operand2);
    double multiply(double operand1, double operand2);
    double add(double operand1, double operand2);
    double subtract(double operand1, double operand2);
    // Compute dot product.
    template <typename T>
    T innerProduct(const std::vector<T>& a, const std::vector<T>& b);
    // Display a vector.
    template <typename T>
    void display(const std::vector<T>& toDisplay) {
        for (typename std::vector<T>::const_iterator itVec = toDisplay.begin(); itVec != toDisplay.end(); ++itVec) {
            std::cout << *itVec << " ";
        }
        std::cout << std::endl;
    }

}

#endif
