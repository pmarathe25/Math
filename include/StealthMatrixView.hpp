#ifndef STEALTH_MATRIX_VIEW_H
#define STEALTH_MATRIX_VIEW_H
#include "ForwardDeclarations.hpp"

namespace StealthMath {
    template <typename Derived, bool Transposed, int newRows>
    class StealthMatrixView {

        enum {
            rows = rowsAtCompileTime,
            cols = colsAtCompileTime,
            size = sizeAtCompileTime
        };


    };
} /* StealthMath */

#endif
