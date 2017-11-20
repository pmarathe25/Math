#ifndef FORWARD_DECLARATIONS_H
#define FORWARD_DECLARATIONS_H
#define CUDA_CALLABLE __host__ __device__
#define THREADS_PER_BLOCK 1024

namespace StealthMath {
    namespace internal {
        template <typename T> struct traits;
    } /* internal */

    // Forward Declarations
    // MatrixBase
    template <typename Derived> class StealthMatrixView;
    // Matrix
    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime =
        rowsAtCompileTime * colsAtCompileTime> class StealthMatrix;
} /* StealthMath */

#endif
