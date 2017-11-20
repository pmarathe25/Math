#ifndef OPS_H
#define OPS_H
#include "StealthMatrixBase.hpp"
#include <iostream>

namespace StealthMath {
    namespace CWiseBinaryOps {
        template <typename LHS, typename RHS, typename ret = LHS>
        CUDA_CALLABLE ret add(const LHS& lhs, const RHS& rhs) {
            // printf("LHS is: %f\n", lhs);
            // printf("Calling add device function\n");
            return lhs + rhs;
        }
    } /* CWiseBinaryOps */

} /* StealthMath */

#endif
