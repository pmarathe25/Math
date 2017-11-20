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
    template <typename Derived> class StealthMatrixBase;
    // Matrix
    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime =
        rowsAtCompileTime * colsAtCompileTime> class StealthMatrix;
    // CWiseBinaryOp
    template <typename LHS, typename RHS>
    using Operation = LHS (*)(const LHS&, const RHS&);
    template <typename LHS, typename RHS,
        Operation<typename internal::traits<LHS>::ScalarType, typename internal::traits<RHS>::ScalarType> op,
        typename LHSType = typename internal::traits<LHS>::ScalarType, typename RHSType =
        typename internal::traits<RHS>::ScalarType, int rowsAtCompileTime =
        internal::traits<LHS>::rows, int colsAtCompileTime = internal::traits<LHS>::cols,
        int sizeAtCompileTime = internal::traits<LHS>::size> class CWiseBinaryOp;

} /* StealthMath */

#endif
