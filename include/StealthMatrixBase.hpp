#ifndef STEALTH_MATRIX_BASE_H
#define STEALTH_MATRIX_BASE_H
#define THREADS_PER_BLOCK 1024
#define CUDA_CALLABLE __device__ __host__
#include <iostream>

namespace StealthMath {
    template <int numerator, int denominator>
    constexpr inline int ceilDivide() {
        return 1 + ((numerator - 1) / denominator);
    }

    template <typename Derived, typename type, int rowsAtCompileTime, int colsAtCompileTime,
        int sizeAtCompileTime = rowsAtCompileTime * colsAtCompileTime>
    class StealthMatrixBase {
        public:
            typedef type ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = sizeAtCompileTime
            };

            CUDA_CALLABLE StealthMatrixBase() {

            }

            CUDA_CALLABLE ScalarType& operator[](int i) {
                return static_cast<Derived*>(this) -> operator[](i);
            }

            CUDA_CALLABLE const ScalarType& operator[](int i) const {
                return static_cast<Derived*>(this) -> operator[](i);
            }

            CUDA_CALLABLE ScalarType& at(int i, int j) {
                return static_cast<Derived*>(this) -> operator[](i * cols + j);
            }

            CUDA_CALLABLE const ScalarType& at(int i, int j) const {
                return static_cast<const Derived*>(this) -> operator[](i * cols + j);
            }
    };
} /* StealthMath */

template <typename Matrix>
void display(Matrix& mat, const std::string& title = "") {
    std::cout << title << '\n';
    for (int i = 0; i < Matrix::rows; ++i) {
        for (int j = 0; j < Matrix::cols; ++j) {
            std::cout << mat.at(i, j) << " ";
        }
        std::cout << '\n';
    }
}

#endif
