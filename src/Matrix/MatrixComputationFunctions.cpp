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
            output = Matrix<T>(numColumns(), numRows());
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
    Matrix<T> Matrix<T>::rowWiseDot(const Matrix& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        } else if (isVector()) {
            // For small matrices/double vectors, compute CPU dot product
            Matrix<T> output = T();
            for (int i = 0; i < size(); ++i) {
                output[0] += (*this)[i] * other[i];
            }
            return output;
        } else {
            Matrix<T> output = Matrix<T>(numRows(), 1);
            // Launch kernel where numThreads = numRows.
            dim3 blocks(std::ceil(numRows() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeRowWiseDotProduct<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    // template <typename T>
    // Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    //     if (numColumns() != other.numRows()) {
    //         throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
    //     } else if (isVector() && other.isVector() && numRows() == other.numRows() && numColumns() == other.numColumns()) {
    //         // If both are vectors, we just need to return the dot product.
    //         return dot(other);
    //     } else {
    //         Matrix product = Matrix(numRows(), other.numColumns());
    //         int Asize = size();
    //         int Bsize = other.size();
    //         int Csize = product.size();
    //         // Initialize device copies.
    //         T *dev_A, *dev_B, *dev_C;
    //         // Allocate memory for device ccpies.
    //         cudaMalloc((void**)&dev_A, Asize * sizeof(T));
    //         cudaMalloc((void**)&dev_B, Bsize * sizeof(T));
    //         cudaMalloc((void**)&dev_C, Csize * sizeof(T));
    //         // Copy inputs to device.
    //         cudaMemcpy(dev_A, data(), Asize * sizeof(T), cudaMemcpyHostToDevice);
    //         cudaMemcpy(dev_B, other.data(), Bsize * sizeof(T), cudaMemcpyHostToDevice);
    //         cudaMemcpy(dev_C, product.data(), Csize * sizeof(T), cudaMemcpyHostToDevice);
    //         // Launch kernel with only as many blocks as necessary.
    //         dim3 blocks(std::ceil(product.numRows() / (float) BLOCK_DIM), std::ceil(product.numColumns() / (float) BLOCK_DIM));
    //         dim3 threads(BLOCK_DIM, BLOCK_DIM);
    //         computeProduct<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns(), other.numRows(), other.numColumns(), size(), other.size(), dev_C);
    //         // Get result.
    //         cudaMemcpy(product.data(), dev_C, Csize * sizeof(T) , cudaMemcpyDeviceToHost);
    //         // Free memory.
    //         cudaFree(dev_A);
    //         cudaFree(dev_B);
    //         cudaFree(dev_C);
    //         // Return.
    //         return product;
    //     }
    // }

/*
    template <typename T>
    Matrix<T> Matrix<T>::hadamard(Matrix& other) {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Cannot find the Hadamard product of incompatible matrices.");
        } else {
            Matrix<T> output = Matrix<T>(numRows(), numColumns());
            dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            computeHadamardProduct<<<blocks, threads>>>(dataGPU(), other.dataGPU(), size(), output.dataGPU());
            output.updateCPUCopy();
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(T other) const {
        Matrix product = Matrix(numRows(), numColumns());
        dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        computeScalarProduct<<<blocks, threads>>>(dataGPU(), other, size(), product.dataGPU());
        return product;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(T other) const {
        if (size() < CPU_SATURATION_LIMIT) {
            return CPUSum(other);
        } else {
            return scalarArithmetic(other, SUM);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(T other) const {
        if (size() < CPU_SATURATION_LIMIT) {
            return CPUDifference(other);
        } else {
            return scalarArithmetic(other, DIFFERENCE);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (!isVector() && other.isVector() && (numColumns() == other.numColumns() || numRows() == other.numRows()) || other.isVector() &&
            (numColumns() != other.numColumns() || numRows() != other.numRows()) ) {
            if (size() < CPU_SATURATION_LIMIT) {
                return CPUMatrixVectorSum(other);
            } else {
                return matrixTiledArithmetic(other, SUM);
            }
        } else if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        } else if (size() < CPU_SATURATION_LIMIT) {
            // For small vectors, use CPU.
            return CPUSum(other);
        } else {
            // For large vectors and matrices, use CUDA.
            return matrixArithmetic(other, SUM);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        if (!isVector() && other.isVector() && (numColumns() == other.numColumns() || numRows() == other.numRows()) || other.isVector() &&
            (numColumns() != other.numColumns() || numRows() != other.numRows()) ) {
            if (size() < CPU_SATURATION_LIMIT) {
                return CPUMatrixVectorDifference(other);
            } else {
                return matrixTiledArithmetic(other, DIFFERENCE);
            }
        } else if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be subtracted.");
        } else if (size() < CPU_SATURATION_LIMIT) {
            // For small vectors, use CPU.
            return CPUDifference(other);
        } else {
            // For large vectors and matrices, use CUDA.
            return matrixArithmetic(other, DIFFERENCE);
        }
    }



    */

}

#endif
