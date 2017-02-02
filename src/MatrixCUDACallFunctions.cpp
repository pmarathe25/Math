#ifndef MATRIX_MATH_HELPER_FUNCTIONS
#define MATRIX_MATH_HELPER_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::transpose() {
        Matrix<T> out = Matrix<T>(numColumns(), numRows());
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            out.raw() = std::vector<T>(raw().begin(), raw().end());
        } else {
            dim3 blocks(std::ceil(numRows() / (float) BLOCK_DIM), std::ceil(numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeTranspose<<<blocks, threads>>>(dataGPU(), numRows(), numColumns(), out.dataGPU());
            out.updateCPUCopy();
        }
        return out;
    }

    template <typename T>
    Matrix<T> Matrix<T>::matrixArithmetic(const Matrix<T>& other, opMode mode) const {
        Matrix output = Matrix(numRows(), numColumns());
        int rawSize = size();
        // Initialize device copies.
        T *dev_A, *dev_B;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, rawSize * sizeof(T));
        cudaMalloc((void**)&dev_B, rawSize * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), rawSize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.data(), rawSize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(rawSize / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        switch (mode) {
            case SUM:
                computeSum<<<blocks, threads>>>(dev_A, dev_B, size());
                break;
            case DIFFERENCE:
                computeDifference<<<blocks, threads>>>(dev_A, dev_B, size());
                break;
        }
        // Get result.
        cudaMemcpy(output.data(), dev_A, rawSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        // Return.
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::matrixTiledArithmetic(const Matrix<T>& other, opMode mode) const {
        Matrix output = Matrix(numRows(), numColumns());
        int rawSize = size();
        // Initialize device copies.
        T *dev_A, *dev_B;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, rawSize * sizeof(T));
        cudaMalloc((void**)&dev_B, rawSize * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), rawSize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.data(), other.size() * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(numRows() / (float) BLOCK_DIM), std::ceil(numColumns() / (float) BLOCK_DIM));
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        switch (mode) {
            case SUM:
                if (other.numRows() == 1) {
                    computeMatrixVectorRowSum<<<blocks, threads>>>(dev_A, dev_B, numColumns(), numRows());
                } else {
                    computeMatrixVectorColumnSum<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns());
                }
                break;
            case DIFFERENCE:
                if (other.numRows() == 1) {
                    computeMatrixVectorRowDifference<<<blocks, threads>>>(dev_A, dev_B, numColumns(), numRows());
                } else {
                    computeMatrixVectorColumnDifference<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns());
                }
                break;
        }
        // Get result.
        cudaMemcpy(output.data(), dev_A, rawSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        // Return.
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::scalarArithmetic(T other, opMode mode) const {
        Matrix product = Matrix(numRows(), numColumns());
        int rawSize = size();
        // Initialize device copies.
        T *dev_A;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, rawSize * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), rawSize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(rawSize / (float) THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        switch (mode) {
            case SUM:
                computeScalarSum<<<blocks, threads>>>(dev_A, other, size());
                break;
            case DIFFERENCE:
                computeScalarDifference<<<blocks, threads>>>(dev_A, other, size());
                break;
        }
        // Get result.
        cudaMemcpy(product.data(), dev_A, rawSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        // Return.
        return product;
    }
}

#endif
