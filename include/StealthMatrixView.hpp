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
        template <typename type, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
        struct traits<StealthMatrixView<type, rowsAtCompileTime, colsAtCompileTime, Transposed>> {
            typedef type ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = rows * cols
            };
        };
    } /* internal */

    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    class StealthMatrixView {
        public:
            typedef type ScalarType;

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


    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    inline StealthMatrixView<type, colsAtCompileTime, rowsAtCompileTime, !Transposed>
        transpose(const StealthMatrixView<type, rowsAtCompileTime, colsAtCompileTime, Transposed>& mat) {
        return StealthMatrixView<type, colsAtCompileTime, rowsAtCompileTime, !Transposed>{mat.data()};
    }

    template <int newRows, int newCols = -1, typename type, int rowsAtCompileTime, int colsAtCompileTime, bool Transposed>
    inline StealthMatrixView<type, newRows, (rowsAtCompileTime * colsAtCompileTime) / newRows, Transposed>
        reshape(const StealthMatrixView<type, rowsAtCompileTime, colsAtCompileTime, Transposed>& mat) {
        static_assert(newRows * ((rowsAtCompileTime * colsAtCompileTime) / newRows) == rowsAtCompileTime * colsAtCompileTime, "Cannot reshape to incompatible dimensions.");
        return StealthMatrixView<type, newRows, (rowsAtCompileTime * colsAtCompileTime) / newRows, Transposed>{mat.data()};
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
