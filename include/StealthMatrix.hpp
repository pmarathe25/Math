#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "StealthMatrixView.hpp"

namespace StealthMath {
        namespace internal {
            template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime>
            struct traits<StealthMatrix<type, rowsAtCompileTime, colsAtCompileTime, sizeAtCompileTime>> {
                typedef type ScalarType;

                enum {
                    rows = rowsAtCompileTime,
                    cols = colsAtCompileTime,
                    size = sizeAtCompileTime
                };
            };
        } /* internal */

    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime>
    class StealthMatrix : public StealthMatrixView<type, rowsAtCompileTime, colsAtCompileTime, false> {
        public:
            typedef type ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = sizeAtCompileTime
            };

            StealthMatrix() {
                cudaMallocManaged(&elements, sizeAtCompileTime * sizeof(ScalarType));
                StealthMatrixView<ScalarType, rowsAtCompileTime, colsAtCompileTime, false>::setViewAddress(elements);
            }

            ~StealthMatrix() {
                cudaFree(elements);
            }

            CUDA_CALLABLE ScalarType& at(int row, int col) {
                return elements[row * cols + col];
            }

            CUDA_CALLABLE const ScalarType& at(int row, int col) const {
                return elements[row * cols + col];
            }

        private:
            ScalarType* elements;
    };
} /* StealthMath */

#endif
