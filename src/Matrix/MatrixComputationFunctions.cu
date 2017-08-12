#ifndef MATRIX_COMPUTATION_FUNCTIONS
#define MATRIX_COMPUTATION_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::transpose() const {
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            Matrix<T> output(*this);
            output.reshape(numColumns(), numRows());
            return output;
        } else {
            Matrix<T> output(numColumns(), numRows());
            dim3 blocks(ceilDivide(numRows(), BLOCK_DIM), ceilDivide(numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            transposeCUDA<<<blocks, threads>>>(data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::weightedSum(float scaleFactor) const {
        if (numRows() == 1) {
            return (*this);
        } else {
            Matrix<T> output(1, numColumns());
            dim3 blocks(ceilDivide(output.size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            weightedSumCUDA<<<blocks, threads>>>(data(), scaleFactor, numColumns(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::dot(const Matrix& other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            Matrix<T> output(numRows(), 1);
            // Launch kernel where numThreads = numRows.
            dim3 blocks(ceilDivide(numRows(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            rowWiseDotProductCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
        if (numColumns() == other.numRows()) {
            Matrix<T> output(numRows(), other.numColumns());
            dim3 blocks(ceilDivide(output.numRows(), BLOCK_DIM), ceilDivide(output.numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            productCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), other.numColumns(), size(), other.size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(Matrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            sumCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }

    template <typename T>
    void Matrix<T>::operator+=(const Matrix& other) {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            sumCUDA<<<blocks, threads>>>(other.data(), data(), size());
            cudaDeviceSynchronize();
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }
    template <typename T>
    Matrix<T> Matrix<T>::operator-(Matrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            differenceCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }

    template <typename T>
    void Matrix<T>::operator-=(const Matrix& other) {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            differenceInPlaceCUDA<<<blocks, threads>>>(other.data(), data(), size());
            cudaDeviceSynchronize();
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }


    template <typename T>
    Matrix<T> Matrix<T>::hadamard(Matrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            hadamardProductCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Cannot find the Hadamard product of incompatible matrices.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::addVector(const Matrix& other) const {
        if (other.isVector() && (other.size() == numRows() || other.size() == numColumns())) {
            Matrix<T> output(numRows(), numColumns());
            dim3 blocks(ceilDivide(output.numRows(), BLOCK_DIM), ceilDivide(output.numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            if (other.numRows() == 1) {
                matrixVectorRowSumCUDA<<<blocks, threads>>>(data(), other.data(), numColumns(), numRows(), output.data());
            } else {
                matrixVectorColumnSumCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            }
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("addVector only accepts Matrices that are Vectors.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(T other) const {
        Matrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarProductCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator/(T other) const {
        Matrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarQuotientCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(T other) const {
        Matrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarSumCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(T other) const {
        Matrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarDifferenceCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::pow(int exponent) {
        Matrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(output.size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        powerCUDA<<<blocks, threads>>>(data(), exponent, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }
}

#endif
