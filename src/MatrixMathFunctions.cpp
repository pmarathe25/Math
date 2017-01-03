#ifndef MATRIX_MATH_FUNCTIONS
#define MATRIX_MATH_FUNCTIONS
#include <random>
#include <chrono>
#include <typeinfo>

namespace math {
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

    template <typename T>
    Matrix<T>& Matrix<T>::transpose() {
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            int temp = rows;
            rows = cols;
            cols = temp;
        } else {
            int matSize = size();
            // Initialize device copies.
            T *dev_original, *dev_transposed;
            // Allocate memory for device ccpies.
            cudaMalloc((void**)&dev_original, matSize * sizeof(T));
            cudaMalloc((void**)&dev_transposed, matSize * sizeof(T));
            // Copy inputs to device.
            cudaMemcpy(dev_original, data(), matSize * sizeof(T), cudaMemcpyHostToDevice);
            // Launch kernel with only as many blocks as necessary.
            dim3 blocks(std::ceil(numRows() / (float) BLOCK_DIM), std::ceil(numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeTranspose<<<blocks, threads>>>(dev_original, numRows(), numColumns(), matSize, dev_transposed);
            // Get result.
            init(numColumns(), numRows());
            cudaMemcpy(data(), dev_transposed, matSize * sizeof(T) , cudaMemcpyDeviceToHost);
            // Free memory.
            cudaFree(dev_original);
            cudaFree(dev_transposed);
        }
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
        if (numColumns() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        } else if (isVector() && other.isVector()) {
            // If both are vectors, we just need to return the dot product.
            return dot(other);
        } else {
            Matrix product = Matrix(numRows(), other.numColumns());
            int Asize = size();
            int Bsize = other.size();
            int Csize = product.size();
            // Initialize device copies.
            T *dev_A, *dev_B, *dev_C;
            // Allocate memory for device ccpies.
            cudaMalloc((void**)&dev_A, Asize * sizeof(T));
            cudaMalloc((void**)&dev_B, Bsize * sizeof(T));
            cudaMalloc((void**)&dev_C, Csize * sizeof(T));
            // Copy inputs to device.
            cudaMemcpy(dev_A, data(), Asize * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_B, other.data(), Bsize * sizeof(T), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_C, product.data(), Csize * sizeof(T), cudaMemcpyHostToDevice);
            // Launch kernel with only as many blocks as necessary.
            dim3 blocks(std::ceil(product.numRows() / (float) BLOCK_DIM), std::ceil(product.numColumns() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeProduct<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns(), other.numRows(), other.numColumns(), size(), other.size(), dev_C);
            // Get result.
            cudaMemcpy(product.data(), dev_C, Csize * sizeof(T) , cudaMemcpyDeviceToHost);
            // Free memory.
            cudaFree(dev_A);
            cudaFree(dev_B);
            cudaFree(dev_C);
            // Return.
            return product;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::dot(const Matrix& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        } else if (size() < CPU_SATURATION_LIMIT || (typeid(T) == typeid(double) && isVector())) {
            // For small matrices/double vectors, compute CPU dot product.
            return CPUDotProduct(other);
        } else if (isVector()) {
            // For large vectors, use CUDA.
            return math::innerProduct(raw(), other.raw());
        } else {
            // For matrices, also use CUDA.
            Matrix output = Matrix(numRows(), 1);
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
            computeDotProduct<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns());
            cudaMemcpy(output.data(), dev_A, output.size() * sizeof(T) , cudaMemcpyDeviceToHost);
            // Free memory.
            cudaFree(dev_A);
            cudaFree(dev_B);
            // Return.
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(T other) const  {
        if (size() < CPU_SATURATION_LIMIT) {
            // For small matrices, use CPU.
            return CPUScalarProduct(other);
        } else {
            // For large matrices, use CUDA.
            return scalarArithmetic(other, SCALAR_PRODUCT);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (!isVector() && other.isVector() && (numColumns() == other.numColumns() || numRows() == other.numRows())) {
            return matrixTiledArithmetic(other, SUM);
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
        if (!isVector() && other.isVector() && (numColumns() == other.numColumns() || numRows() == other.numRows())) {
            return matrixTiledArithmetic(other, DIFFERENCE);
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
        std::cout << "EXECUTING CUDA" << std::endl;
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
            case SCALAR_PRODUCT:
                computeScalarProduct<<<blocks, threads>>>(dev_A, other, size());
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
