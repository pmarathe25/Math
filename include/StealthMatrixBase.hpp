#ifndef STEALTH_MATRIX_BASE_H
#define STEALTH_MATRIX_BASE_H

namespace StealthMath {

    __global__ 

    template <typename Derived, int rows, int cols>
    class StealthMatrixBase {
        public:
            __device__ __host__ StealthMatrixBase() {

            }

            template <typename OtherDerived>
            __device__ __host__ operator=(const StealthMatrixBase<OtherDerived, rows, cols>& other) {

            }

    };
}

#endif
