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
                dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                sumCUDA<<<blocks, threads>>>(data(), other.data(), size());
                cudaDeviceSynchronize();
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
