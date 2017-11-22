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
                rows = (rowsAtCompileTime == -1) ? internal::traits<MatrixType>::rows : rowsAtCompileTime,
                cols = (rowsAtCompileTime == -1) ? internal::traits<MatrixType>::cols :
                    ((colsAtCompileTime == -1) ? internal::traits<MatrixType>::size / rowsAtCompileTime : colsAtCompileTime),
                size = rows * cols
            };
        };
    } /* internal */

    template <typename MatrixType, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    class StealthMatrixView {
        public:
            typedef typename internal::traits<StealthMatrixView>::ScalarType ScalarType;

            enum {
                rows = internal::traits<StealthMatrixView>::rows,
                cols = internal::traits<StealthMatrixView>::cols,
                size = internal::traits<StealthMatrixView>::size
            };

            StealthMatrixView(const ScalarType* elementsView = 0) : elementsView(elementsView) {  }

            CUDA_CALLABLE const ScalarType& at(int row, int col) const {
                if (!Transposed) {
                    return elementsView[row * cols + col];
                } else {
                    return elementsView[col * rows + row];
                }
            }

            StealthMatrixView<MatrixType, cols, rows, !Transposed> transpose() const {
                return StealthMatrixView<MatrixType, cols, rows, !Transposed>{elementsView};
            }

        protected:
            void setViewAddress(const ScalarType* newAddress) {
                elementsView = newAddress;
            }

        private:
            const ScalarType* elementsView;
    };

    template <typename MatrixType>
    void display(const MatrixType& mat, const std::string& title = "") {
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
