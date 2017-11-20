#ifndef C_WISE_BINARY_OP_H
#define C_WISE_BINARY_OP_H
#include "StealthMatrixBase.hpp"
#include "ForwardDeclarations.hpp"
#include "Ops.hpp"

namespace StealthMath {
    namespace internal {
        template <typename LHS, typename RHS,
            Operation<typename internal::traits<LHS>::ScalarType, typename internal::traits<RHS>::ScalarType> op,
            typename LHSType, typename RHSType, int rowsAtCompileTime, int colsAtCompileTime,
            int sizeAtCompileTime>
        struct traits<CWiseBinaryOp<LHS, RHS, op, LHSType, RHSType, rowsAtCompileTime, colsAtCompileTime, sizeAtCompileTime>> {
            typedef LHSType ScalarType;
            typedef LHSType ScalarTypeLHS;
            typedef RHSType ScalarTypeRHS;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = sizeAtCompileTime
            };
        };
    } /* internal */

    template <typename LHS, typename RHS,
        Operation<typename internal::traits<LHS>::ScalarType, typename internal::traits<RHS>::ScalarType> op,
        typename LHSType, typename RHSType,
        int rowsAtCompileTime,
        int colsAtCompileTime, int sizeAtCompileTime>
    class CWiseBinaryOp : public StealthMatrixBase<CWiseBinaryOp<LHS, RHS, op>> {
        public:
            typedef LHSType ScalarType;

            CWiseBinaryOp(const StealthMatrixBase<LHS>* lhs, const StealthMatrixBase<RHS>* rhs) :
                lhs(static_cast<const LHS*>(lhs)), rhs(static_cast<const RHS*>(rhs)),
                dev_lhs(static_cast<const LHS*>(lhs) -> deviceData()), dev_rhs(static_cast<const RHS*>(rhs) -> deviceData()) {
                std::cout << "Constructing Binary OP with matrices of size " << LHS::size << " and " << RHS::size << '\n';
                // Allocate dev_ptr
                cudaMalloc((void**) &dev_ptr, sizeof(this));
                cudaMemcpy(dev_ptr, this, sizeof(this), cudaMemcpyHostToDevice);
            }

            ~CWiseBinaryOp() {
                cudaFree(dev_ptr);
            }

            CUDA_CALLABLE CWiseBinaryOp* deviceData() {
                return dev_ptr;
            }

            CUDA_CALLABLE const CWiseBinaryOp* deviceData() const {
                return dev_ptr;
            }

            CUDA_CALLABLE inline ScalarType operator[] (int i) {
                // return op((*dev_lhs)[i], (*dev_rhs)[i]);
                // return op((*(dev_ptr -> lhs))[i], (*(dev_ptr -> rhs))[i]);
                return op((*lhs)[i], (*rhs)[i]);
            }

            CUDA_CALLABLE inline const ScalarType operator[] (int i) const {
                printf("Calling CWiseBinaryOp [] operator\n");
                // return op((*lhs)[i], (*rhs)[i]);
                return op((*dev_lhs)[i], (*dev_rhs)[i]);
                // return op((*(dev_ptr -> dev_lhs))[i], (*(dev_ptr -> dev_rhs))[i]);
                // return op(1.0, 1.0);
            }

            const ScalarType at_local(int i) const {
                return op(lhs -> at(i), rhs -> at(i));
            }

        private:
            const LHS* lhs, *dev_lhs;
            const RHS* rhs, *dev_rhs;
            CWiseBinaryOp* dev_ptr;
    };

    template <typename Derived>
    template <typename OtherDerived>
    CWiseBinaryOp<Derived, OtherDerived, CWiseBinaryOps::add> StealthMatrixBase<Derived>::operator+(const StealthMatrixBase<OtherDerived>& other) {
        return CWiseBinaryOp<Derived, OtherDerived, CWiseBinaryOps::add>(this, &other);
    }
} /* StealthMath */

#endif
