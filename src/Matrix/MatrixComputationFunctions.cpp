#ifndef MATRIX_COMPUTATION_FUNCTIONS
#define MATRIX_COMPUTATION_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::transpose() const {
        Matrix<T> output;
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            output = (*this);
            output.reshape(numColumns(), numRows());
        } else {
            output = Matrix(numColumns(), numRows());
            dim3 blocks(std::ceil(numRows() / (float) BLOCK_DIM), std::ceil(numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeTranspose<<<blocks, threads>>>(data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::rowMean() const {
        if (numRows() == 1) {
            return (*this);
        } else {
            Matrix output = Matrix(1, numColumns());
            float scaleFactor = 1 / (float) numRows();
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeRowMean<<<blocks, threads>>>(data(), scaleFactor, numColumns(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::dot(const Matrix& other) const {
        try {
            Matrix output = Matrix(numRows(), 1);
            // Launch kernel where numThreads = numRows.
            dim3 blocks(std::ceil(numRows() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeRowWiseDotProduct<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
        try {
            Matrix output = Matrix(numRows(), other.numColumns());
            dim3 blocks(std::ceil(output.numRows() / (float) BLOCK_DIM), std::ceil(output.numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeProduct<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), other.numColumns(), size(), other.size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        try {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeSum<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        try {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeDifference<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::hadamard(const Matrix& other) const {
        try {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeHadamardProduct<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("Cannot find the Hadamard product of incompatible matrices.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::addVector(const Matrix<T>& other) const {
        try {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(output.numRows() / (float) BLOCK_DIM), std::ceil(output.numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            if (other.numRows() == 1) {
                computeMatrixVectorRowSum<<<blocks, threads>>>(data(), other.data(), numColumns(), numRows(), output.data());
            } else {
                computeMatrixVectorColumnSum<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            }
            cudaDeviceSynchronize();
            return output;
        } catch (const std::exception& e) {
            throw std::invalid_argument("addVector only accepts Matrices that are Vectors.");
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        computeScalarProduct<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        computeScalarSum<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        computeScalarSum<<<blocks, threads>>>(data(), -other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }
}

#endif
