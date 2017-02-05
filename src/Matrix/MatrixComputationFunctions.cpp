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
        if (isVector() && size() == other.size()) {
            Matrix output = T();
            for (int i = 0; i < size(); ++i) {
                output[0] += (*this)[i] * other[i];
            }
            return output;
        } else if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        } else {
            Matrix output = Matrix(numRows(), 1);
            // Launch kernel where numThreads = numRows.
            dim3 blocks(std::ceil(numRows() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeRowWiseDotProduct<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
        if (numColumns() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        } else if (isVector() && other.isVector()) {
            return dot(other);
        } else {
            Matrix output = Matrix(numRows(), other.numColumns());
            dim3 blocks(std::ceil(output.numRows() / (float) BLOCK_DIM), std::ceil(output.numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeProduct<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), other.numRows(), other.numColumns(), size(), other.size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        } else {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeSum<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        } else {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeDifference<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::hadamard(const Matrix& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Cannot find the Hadamard product of incompatible matrices.");
        } else {
            Matrix output = Matrix(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeHadamardProduct<<<blocks, threads>>>(data(), other.data(), size(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::addVector(const Matrix<T>& other) const {
        if (!other.isVector()) {
            throw std::invalid_argument("addVector only accepts Matrices that are Vectors.");
        } else {
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
