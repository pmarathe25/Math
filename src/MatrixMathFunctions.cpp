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

}

#endif
