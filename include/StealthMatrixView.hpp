#ifndef STEALTH_MATRIX_VIEW_H
#define STEALTH_MATRIX_VIEW_H
#include "ForwardDeclarations.hpp"

namespace StealthMath {
    template <int numerator, int denominator>
    constexpr inline int ceilDivide() {
        return 1 + ((numerator - 1) / denominator);
    }

    namespace internal {
        template <typename Derived, bool Transposed, int newRows, int newCols>
        struct traits<StealthMatrixView<Derived, Transposed, newRows, newCols>> {
            typedef typename internal::traits<Derived>::ScalarType ScalarType;

            enum {
                rows = (newRows == -1) ? internal::traits<Derived>::rows : newRows,
                cols = (newRows == -1) ? internal::traits<Derived>::cols : ((newCols == -1) ? internal::traits<Derived>::size / newRows : newCols),
                size = internal::traits<Derived>::size
            };
        };
    } /* internal */

    template <typename Derived, bool Transposed, int newRows, int newCols>
    class StealthMatrixView {
        public:
            typedef typename internal::traits<StealthMatrixView>::ScalarType ScalarType;

            enum {
                rows = internal::traits<StealthMatrixView>::rows,
                cols = internal::traits<StealthMatrixView>::cols,
                size = internal::traits<StealthMatrixView>::size
            };

            StealthMatrixView(const ScalarType* elementsView) : elementsView(elementsView) {  }

        private:
            const ScalarType* elementsView;
    };
} /* StealthMath */

#endif
