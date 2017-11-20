#ifndef STEALTH_MATRIX_H
#define STEALTH_MATRIX_H
#include "ForwardDeclarations.hpp"
#include "StealthMatrixBase.hpp"

namespace StealthMath {
    namespace internal {
        template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime>
        struct traits<StealthMatrix<type, rowsAtCompileTime, colsAtCompileTime, sizeAtCompileTime>> {
            typedef type ScalarType;

            enum {
                rows = rowsAtCompileTime,
                cols = colsAtCompileTime,
                size = sizeAtCompileTime
            };
        };
    } /* internal */

    template <typename Matrix, typename OtherMatrix>
    __global__ void copy(Matrix* A, const OtherMatrix* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Matrix::size) {
            printf("Index: %d\n", index);
            A -> setValue(index, (*B)[index]);
        }
    }

    template <typename type, int rowsAtCompileTime, int colsAtCompileTime, int sizeAtCompileTime>
    class StealthMatrix : public StealthMatrixBase<StealthMatrix<type, rowsAtCompileTime, colsAtCompileTime>> {
        public:
            typedef type ScalarType;

            StealthMatrix() {
                cudaMallocManaged(&elements, StealthMatrix::size * sizeof(ScalarType));
                // Allocate dev_ptr
                cudaMalloc((void**) &dev_ptr, sizeof(this));
                cudaMemcpy(dev_ptr, this, sizeof(this), cudaMemcpyHostToDevice);
            }

            ~StealthMatrix() {
                cudaFree(dev_ptr);
            }

            StealthMatrix* deviceData() {
                return dev_ptr;
            }

            const StealthMatrix* deviceData() const {
                return dev_ptr;
            }

            template <typename OtherDerived>
            StealthMatrix(const StealthMatrixBase<OtherDerived>& other) {
                *this = other;
            }

            void operator=(const StealthMatrix& other) {
                set(other);
            }

            template <typename OtherDerived>
            void operator=(const StealthMatrixBase<OtherDerived>& other) {
                set(other);
            }

            CUDA_CALLABLE ScalarType operator[] (int i) {
                return elements[i];
            }

            CUDA_CALLABLE const ScalarType operator[] (int i) const {
                // printf("Calling [] operator\n");
                return elements[i];
            }

            CUDA_CALLABLE void setValue(int index, ScalarType value) {
                elements[index] = value;
            }

            ScalarType& at_local(int i) {
                return elements[i];
            }

            const ScalarType at_local(int i) const {
                return elements[i];
            }

        private:
            template <typename OtherDerived>
            void set(const StealthMatrixBase<OtherDerived>& other) {
                std::cout << "Calling Copy Kernel" << '\n';
                // Launch kernel
                dim3 blocks(ceilDivide<StealthMatrix::size, THREADS_PER_BLOCK>());
                dim3 threads(THREADS_PER_BLOCK);
                copy<<<blocks, threads>>>((*this).deviceData(), other.deviceData());
                cudaDeviceSynchronize();
            }

            ScalarType* elements;
            StealthMatrix* dev_ptr;
    };

} /* StealthMath */


#endif
