#ifndef MATRIX_MODIFICATION_FUNCTIONS
#define MATRIX_MODIFICATION_FUNCTIONS
#include <chrono>
#include <random>

namespace math {
    template <typename T>
    void Matrix<T>::reshape(int rows, int cols) {
        if (rows * cols == size()) {
            this -> rows = rows;
            this -> cols = cols;
            this -> isVec = (rows == 1) || (cols == 1);
        } else {
            throw std::invalid_argument("Size mismatch in reshape.");
        }
    }

    template <typename T>
    void Matrix<T>::randomizeNormal(T mean, T stdDev) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < size(); ++i) {
            elements[i] = normalDistribution(generator);
        }
    }

    template <typename T>
    void Matrix<T>::randomizeUniform(T lowerBound, T upperBound) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < size(); ++i) {
            elements[i] = uniformDistribution(generator);
        }
    }
}

#endif
