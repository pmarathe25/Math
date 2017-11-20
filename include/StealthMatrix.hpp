#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "StealtMatrixView.hpp"

namespace StealthMath {
    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime>
    class StealthMatrix : public StealthMatrixView<StealthMatrix<type, rowsAtCompileTime, colsAtCompileTime>> {
        public:
            typedef type ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = sizeAtCompileTime
            };

            StealthMatrix() {
                cudaMallocManaged(&elements, sizeAtCompileTime * sizeof(ScalarType));
            }

            ~StealthMatrix() {
                cudaFree(elements);
            }

        private:
            ScalarType* elements;
    };
} /* StealthMath */

#endif
