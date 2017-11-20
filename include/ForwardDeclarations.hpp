#ifndef FORWARD_DECLARATIONS_H
#define FORWARD_DECLARATIONS_H

namespace StealthMath {
    namespace internal {
        template <typename T> struct traits;
    } /* internal */

    // Matrix forward declaration
    template <typename Derived> class StealthMatrixBase;
    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime = rowsAtCompileTime * colsAtCompileTime> class StealthMatrix;


} /* StealthMath */

#endif
