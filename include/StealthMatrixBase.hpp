#ifndef STEALTH_MATRIX_BASE_H
#define STEALTH_MATRIX_BASE_H
#include "ForwardDeclarations.hpp"
#include "Ops.hpp"
#include <iostream>

namespace StealthMath {
    template <int numerator, int denominator>
    constexpr inline int ceilDivide() {
        return 1 + ((numerator - 1) / denominator);
    }

    template <typename Derived>
    class StealthMatrixBase {
        public:
            typedef typename internal::traits<Derived>::ScalarType ScalarType;

            enum {
                rows = internal::traits<Derived>::rows,
                cols = internal::traits<Derived>::cols,
                size = internal::traits<Derived>::size
            };

            StealthMatrixBase() {  }

            CUDA_CALLABLE Derived* deviceData() {
                return static_cast<Derived*>(this) -> deviceData();
            }

            CUDA_CALLABLE const Derived* deviceData() const {
                return static_cast<const Derived*>(this) -> deviceData();
            }

            // Access Functions
            CUDA_CALLABLE inline ScalarType operator[](int i) {
                return static_cast<Derived*>(this) -> operator[](i);
            }

            CUDA_CALLABLE inline const ScalarType operator[](int i) const {
                return static_cast<const Derived*>(this) -> operator[](i);
            }

            ScalarType& at(int i) {
                if (i >= size) {
                    throw std::out_of_range("Index out of bounds");
                }
                return static_cast<Derived*>(this) -> at_local(i);
            }

            const ScalarType at(int i) const {
                if (i >= size) {
                    throw std::out_of_range("Index out of bounds");
                }
                return static_cast<const Derived*>(this) -> at_local(i);
            }

            ScalarType& at(int i, int j) {
                if (i >= rows) {
                    throw std::out_of_range("Row index out of bounds");
                } else if (j >= cols) {
                    throw std::out_of_range("Columns index out of bounds");
                }
                return static_cast<Derived*>(this) -> at_local(i * cols + j);
            }

            const ScalarType at(int i, int j) const {
                if (i >= rows) {
                    throw std::out_of_range("Row index out of bounds");
                } else if (j >= cols) {
                    throw std::out_of_range("Columns index out of bounds");
                }
                return static_cast<const Derived*>(this) -> at_local(i * cols + j);
            }

            template <typename OtherDerived>
            CWiseBinaryOp<Derived, OtherDerived, CWiseBinaryOps::add> operator+(const StealthMatrixBase<OtherDerived>& other);
    };
} /* StealthMath */

template <typename Matrix>
void display(const Matrix& mat, const std::string& title = "") {
    std::cout << title << '\n';
    for (int i = 0; i < Matrix::rows; ++i) {
        for (int j = 0; j < Matrix::cols; ++j) {
            std::cout << mat.at(i, j) << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

#endif
