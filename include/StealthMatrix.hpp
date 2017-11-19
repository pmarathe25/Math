#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "StealthMatrixBase.hpp"

namespace StealthMath {
    template <typename ScalarType, int rows, int cols>
    class StealthMatrix : public StealthMatrixBase<StealthMatrix, ScalarType, rows, cols> {
        public:
            CUDA_CALLABLE

            template <typename OtherDerived>
            CUDA_CALLABLE operator=(const StealthMatrixBase<OtherDerived, rows, cols>& other) {

            }
        private:
            ScalarType* elements;
    };

    template <typename T>
    __global__ void copy(const T* A, const T* B, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = A[index] + B[index];
        }
    }
} /* StealthMath */


#endif
