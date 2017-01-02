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
            at(i) = normalDistribution(generator);
        }
    }

    template <typename T>
    void Matrix<T>::randomizeUniform(T lowerBound, T upperBound) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < size(); ++i) {
            at(i) = uniformDistribution(generator);
        }
    }

    template <typename T>
    Matrix<T>& Matrix<T>::transpose() {
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            int temp = rowsRaw;
            rowsRaw = colsRaw;
            colsRaw = temp;
            temp = rows;
            rows = cols;
            cols = temp;
        } else {
            int size = sizeRaw();
            // Initialize device copies.
            T *dev_original, *dev_transposed;
            // Allocate memory for device ccpies.
            cudaMalloc((void**)&dev_original, size * sizeof(T));
            cudaMalloc((void**)&dev_transposed, size * sizeof(T));
            // Copy inputs to device.
            cudaMemcpy(dev_original, data(), size * sizeof(T), cudaMemcpyHostToDevice);
            // Launch kernel with only as many blocks as necessary.
            dim3 blocks(std::ceil(numRowsRaw() / (float) BLOCK_DIM), std::ceil(numColumnsRaw() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            computeTranspose<<<blocks, threads>>>(dev_original, numRowsRaw(), numColumnsRaw(), dev_transposed);
            // Get result.
            init(numColumns(), numRows());
            cudaMemcpy(data(), dev_transposed, size * sizeof(T) , cudaMemcpyDeviceToHost);
            // Free memory.
            cudaFree(dev_original);
            cudaFree(dev_transposed);
        }
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
        if (size() == 1) {
            // If there is only one element in one of the matrices, do scalar multiplication instead.
            return other * at(0);
        } else if (other.size() == 1) {
            return *this * other.at(0);
        } else if (isVector() && other.isVector()) {
            // If both are vectors, we just need to return the dot product.
            return Matrix<T>(dot(other));
        } else if (numColumns() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        } else {
            Matrix product = Matrix(numRows(), other.numColumns());
            int Asize = sizeRaw();
            int Bsize = other.sizeRaw();
            int Csize = product.sizeRaw();
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
            dim3 blocks(std::ceil(product.numRowsRaw() / (float) BLOCK_DIM), std::ceil(product.numColumnsRaw() / (float) BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            if (isVector()) {
                computeVectorProductLeft<<<blocks, threads>>>(dev_A, dev_B, numColumnsRaw(), other.numColumnsRaw(), dev_C);
            } else if (other.isVector()) {
                computeVectorProductRight<<<blocks, threads>>>(dev_A, dev_B, numColumnsRaw(), dev_C);
            } else {
                computeProduct<<<blocks, threads>>>(dev_A, dev_B, numColumnsRaw(), other.numColumnsRaw(), dev_C);
            }
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
    Matrix<T> Matrix<T>::operator*(T other) const  {
        if (isVector() && sizeRaw() < CPU_SATURATION_LIMIT) {
            // For small vectors, use CPU.
            return CPUScalarProduct(other);
        } else {
            // For large vectors and matrices, use CUDA.
            return scalarArithmetic(other, SCALAR_PRODUCT);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::dot(const Matrix& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        } else if ((sizeRaw() < CPU_SATURATION_LIMIT || typeid(T) == typeid(double)) && isVector()) {
            // For small vectors, compute CPU dot product.
            T product = T();
            // numColumnsRaw is guaranteed to be a multiple of 32.
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    product += data()[i + j] * other.data()[i + j];
                }
            }
            return Matrix<T>(product);
        } else if (isVector()) {
            // For large vectors, use CUDA.
            return math::innerProduct(raw(), other.raw());
        } else {
            // For matrices, also use CUDA.
            Matrix output = Matrix(numRows(), 1);
            int rawSize = sizeRaw();
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
            computeDotProduct<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumnsRaw());
            cudaMemcpy(output.data(), dev_A, output.size() * sizeof(T) , cudaMemcpyDeviceToHost);
            // Free memory.
            cudaFree(dev_A);
            cudaFree(dev_B);
            // Return.
            return output;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (isVector() && sizeRaw() < CPU_SATURATION_LIMIT) {
            // For small vectors, use CPU.
            return CPUSum(other);
        } else {
            // For large vectors and matrices, use CUDA.
            return matrxArithmetic(other, SUM);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        if (isVector() && sizeRaw() < CPU_SATURATION_LIMIT) {
            // For small vectors, use CPU.
            return CPUDifference(other);
        } else {
            // For large vectors and matrices, use CUDA.
            return matrxArithmetic(other, DIFFERENCE);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::matrxArithmetic(const Matrix<T>& other, opMode mode) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        } else {
            Matrix output = Matrix(numRows(), numColumns());
            int rawSize = sizeRaw();
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
                    if (isVector()) {
                        computeVectorSum<<<blocks, threads>>>(dev_A, dev_B, size());
                    } else {
                        computeSum<<<blocks, threads>>>(dev_A, dev_B);
                    }
                    break;
                case DIFFERENCE:
                    if (isVector()) {
                        computeVectorDifference<<<blocks, threads>>>(dev_A, dev_B, size());
                    } else {
                        computeDifference<<<blocks, threads>>>(dev_A, dev_B);
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
    }

    template <typename T>
    Matrix<T> Matrix<T>::scalarArithmetic(T other, opMode mode) const {
        Matrix product = Matrix(numRows(), numColumns());
        int rawSize = sizeRaw();
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
                if (isVector()) {
                    computeVectorScalarProduct<<<blocks, threads>>>(dev_A, other, size());
                } else {
                    computeScalarProduct<<<blocks, threads>>>(dev_A, other);
                }
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
