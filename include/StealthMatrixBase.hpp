#ifndef STEALTH_MATRIX_BASE_H
#define STEALTH_MATRIX_BASE_H
#define CUDA_CALLABLE __device__ __host__

namespace StealthMath {
    template <typename Derived, typename ScalarType, int rows, int cols>
    class StealthMatrixBase {
        public:
            enum {
                Rows = rows,
                Columns = cols,
                Size = rows * cols
            };

            CUDA_CALLABLE StealthMatrixBase() {

            }

            CUDA_CALLABLE ScalarType operator[](int i) {
                return static_cast<Derived*>(this) -> [];
            }
    };
} /* StealthMath */

#endif
