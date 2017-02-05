#ifndef MATRIX_MODIFICATION_FUNCTIONS
#define MATRIX_MODIFICATION_FUNCTIONS
#include <chrono>
#include <random>

namespace math {
    template <typename T>
    void Matrix<T>::reshape(int rows, int cols) {
        if (rows * cols == size()) {
            this -> rows = rows;
            this -> cols = cols;
            this -> isVec = (rows == 1) || (cols == 1);
        } else {
            throw std::invalid_argument("Size mismatch in reshape.");
        }
    }

    template <typename T>
    void Matrix<T>::randomizeNormal(T mean, T stdDev) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < size(); ++i) {
            elements[i] = normalDistribution(generator);
        }
    }

    template <typename T>
    void Matrix<T>::randomizeUniform(T lowerBound, T upperBound) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < size(); ++i) {
            elements[i] = uniformDistribution(generator);
        }
    }


    /*

    template <typename T>
    Matrix<T> Matrix<T>::matrixArithmetic(const Matrix<T>& other, int mode) const {
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
    Matrix<T> Matrix<T>::matrixTiledArithmetic(const Matrix<T>& other, int mode) const {
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
    Matrix<T> Matrix<T>::scalarArithmetic(T other, int mode) const {
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

    */
}

#endif
