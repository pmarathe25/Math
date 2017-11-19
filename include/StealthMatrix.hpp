#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "StealthMatrixBase.hpp"

namespace StealthMath {

    template <typename Matrix>
    __global__ void copy(Matrix* A, const Matrix* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Matrix::size) {
            A[index] = B[index];
        }
    }

    template <typename ScalarType, int rows, int cols>
    class StealthMatrix : public StealthMatrixBase<StealthMatrix<ScalarType, rows, cols>, ScalarType, rows, cols> {
        public:
            StealthMatrix() {
                cudaMallocManaged(&elements, rows * cols * sizeof(ScalarType));
            }

            template <typename OtherDerived>
            CUDA_CALLABLE void operator=(const StealthMatrixBase<OtherDerived, ScalarType, rows, cols>& other) {
                dim3 blocks(ceilDivide<StealthMatrix::size, THREADS_PER_BLOCK>());
                dim3 threads(THREADS_PER_BLOCK);
                copy<<<blocks, threads>>>(this, &other);
                cudaDeviceSynchronize();
            }

            CUDA_CALLABLE ScalarType& operator[] (int i) {
                return elements[i];
            }

            CUDA_CALLABLE const ScalarType& operator[] (int i) const {
                return elements[i];
            }
        private:
            ScalarType* elements;
    };

} /* StealthMath */


#endif
