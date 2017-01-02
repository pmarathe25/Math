#ifndef MATRIX_CPU_FUNCTIONS
#define MATRIX_CPU_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        const T* otherData = other.data();
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    outputData[i + j] = thisData[i + j] + otherData[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    outputData[i + j] = thisData[i + j] + otherData[i + j];
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDifference(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        const T* otherData = other.data();
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    outputData[i + j] = thisData[i + j] - otherData[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    outputData[i + j] = thisData[i + j] - otherData[i + j];
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUScalarProduct(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    outputData[i + j] = thisData[i + j] * other;
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    outputData[i + j] = thisData[i + j] * other;
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDotProduct(const Matrix<T>& other) const {
        T product = T();
        // sizeRaw is guaranteed to be a multiple of 32.
        for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                product += data()[i + j] * other.data()[i + j];
            }
        }
        return Matrix<T>(product);
    }
}

#endif
