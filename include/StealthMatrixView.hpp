#ifndef STEALTH_MATRIX_VIEW_H
#define STEALTH_MATRIX_VIEW_H
#include "ForwardDeclarations.hpp"
#include <iostream>
#include <string>

namespace StealthMath {
    template <int numerator, int denominator>
    constexpr inline int ceilDivide() {
        return 1 + ((numerator - 1) / denominator);
    }

    namespace internal {
        template <typename MatrixType, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
        struct traits<StealthMatrixView<MatrixType, rowsAtCompileTime, colsAtCompileTime, Transposed>> {
            typedef typename internal::traits<MatrixType>::ScalarType ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = rows * cols
            };
        };
    } /* internal */

    template <typename MatrixType, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    class StealthMatrixView {
        public:
            typedef typename internal::traits<StealthMatrixView>::ScalarType ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = rows * cols
            };

            StealthMatrixView(const ScalarType* elementsView = 0) : elementsView(elementsView) {  }

            CUDA_CALLABLE const ScalarType& at(int row, int col) const {
                if (!Transposed) {
                    return elementsView[row * cols + col];
                } else {
                    return elementsView[col * rows + row];
                }
            }

            const ScalarType* data() const {
                return elementsView;
            }
        protected:
            void setViewAddress(const ScalarType* newAddress) {
                elementsView = newAddress;
            }

        private:
            const ScalarType* elementsView;
    };


    template <typename MatrixType, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    inline StealthMatrixView<MatrixType, colsAtCompileTime, rowsAtCompileTime, !Transposed>
        transpose(const StealthMatrixView<MatrixType, rowsAtCompileTime, colsAtCompileTime, Transposed>& mat) {
        return StealthMatrixView<MatrixType, colsAtCompileTime, rowsAtCompileTime, !Transposed>{mat.data()};
    }

    template <int newRows, int newCols = -1, typename MatrixType, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    inline StealthMatrixView<MatrixType, newRows, MatrixType::size / newRows, Transposed>
        reshape(const StealthMatrixView<MatrixType, rowsAtCompileTime, colsAtCompileTime, Transposed>& mat) {
        static_assert(newRows * (MatrixType::size / newRows) == rowsAtCompileTime * colsAtCompileTime, "Cannot reshape to incompatible dimensions.");
        return StealthMatrixView<MatrixType, newRows, MatrixType::size / newRows, Transposed>{mat.data()};
    }

    template <typename MatrixType>
    inline void display(const MatrixType& mat, const std::string& title = "") {
        std::cout << title << '\n';
        for (int i = 0; i < MatrixType::rows; ++i) {
            for (int j = 0; j < MatrixType::cols; ++j) {
                std::cout << mat.at(i, j) << " ";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
} /* StealthMath */

#endif
