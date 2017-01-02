#ifndef MATRIX_CPU_FUNCTIONS
#define MATRIX_CPU_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDifference(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] - other.data()[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] - other.data()[i + j];
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUScalarProduct(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] * other;
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] * other;
                }
            }
        }
        return output;
    }
}

#endif
