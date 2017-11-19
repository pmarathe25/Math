#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "StealthMatrixBase.hpp"

namespace StealthMath {
    template <typename ScalarType, int rows, int cols>
    class StealthMatrix : public StealthMatrixBase<StealthMatrix, ScalarType, rows, cols> {
        public:
            CUDA_CALLABLE StealthMatrix() {
                cudaMallocManaged(&elements, rows * cols * sizeof(ScalarType));
            }

            template <typename OtherDerived>
            CUDA_CALLABLE void operator=(const StealthMatrixBase<OtherDerived, rows, cols>& other) {
                
            }
        private:
            ScalarType* elements;
    };

    template <typename Matrix>
    __global__ void copy(Matrix* A, const Matrix* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < A::Size) {
            A[index] = B[index];
        }
    }
} /* StealthMath */


#endif
