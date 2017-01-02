#include "Math/Matrix.hpp"
#include "Text/strmanip.hpp"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <typeinfo>
#include <random>
#include <chrono>

namespace math {

    template <typename T>
    void Matrix<T>::init(int rows, int cols) {
        int rowsPadded = rows;
        int colsPadded = cols;
        // Do not pad vectors.
        isVec = (rows == 1) || (cols == 1);
        if (rows % BLOCK_DIM != 0 && rows != 1) {
            rowsPadded += BLOCK_DIM - (rows % BLOCK_DIM);
        }
        if (cols % BLOCK_DIM != 0 && cols != 1) {
            colsPadded += BLOCK_DIM - (cols % BLOCK_DIM);
        }
        elements = std::vector<T> (rowsPadded * colsPadded);
        this -> rowsRaw = rowsPadded;
        this -> colsRaw = colsPadded;
        this -> rows = rows;
        this -> cols = cols;
    }

    template <typename T>
    Matrix<T>::Matrix() {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(0, 0);
    }

    template <typename T>
    Matrix<T>::Matrix(T elem) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(1, 1);
        elements[0] = elem;
    }

    template <typename T>
    Matrix<T>::Matrix(int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(rows, cols);
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(1, initialElements.size());
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        for (int i = 0; i < size(); ++i) {
            at(i) = initialElements.at(i);
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(rows, cols);
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        for (int i = 0; i < size(); ++i) {
            at(i) = initialElements.at(i);
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        init(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                at(row, col) = initialElements.at(row).at(col);
            }
        }
    }

    template <typename T>
    T& Matrix<T>::at(int row, int col) {
        if (row < numRows() && col < numColumns()) {
            return elements[row * numColumnsRaw() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    const T& Matrix<T>::at(int row, int col) const {
        if (row < numRows() && col < numColumns()) {
            return elements[row * numColumnsRaw() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    T& Matrix<T>::at(int index) {
        int row = index / numColumns();
        int col = index % numColumns();
        return at(row, col);
    }

    template <typename T>
    const T& Matrix<T>::at(int index) const {
        return at(index / numColumns(), index % numColumns());
    }

    template <typename T>
    T* Matrix<T>::data() {
        return elements.data();
    }

    template <typename T>
    const T* Matrix<T>::data() const {
        return elements.data();
    }

    template <typename T>
    std::vector<T>& Matrix<T>::raw() {
        return elements;
    }

    template <typename T>
    const std::vector<T>& Matrix<T>::raw() const {
        return elements;
    }

    template <typename T>
    std::vector<T> Matrix<T>::getElements() const {
        std::vector<T> temp;
        std::vector<T> tempRow;
        temp.reserve(size());
        for (int i = 0; i < numRows(); ++i) {
            tempRow = row(i);
            temp.insert(temp .end(), tempRow.cbegin(), tempRow.cend());
        }
        return temp;
    }

    template <typename T>
    int Matrix<T>::numRowsRaw() const {
        return rowsRaw;
    }

    template <typename T>
    int Matrix<T>::numColumnsRaw() const {
        return colsRaw;
    }

    template <typename T>
    int Matrix<T>::numRows() const {
        return rows;
    }

    template <typename T>
    int Matrix<T>::numColumns() const {
        return cols;
    }

    template <typename T>
    int Matrix<T>::sizeRaw() const {
        return numColumnsRaw() * numRowsRaw();
    }

    template <typename T>
    int Matrix<T>::size() const {
        return numColumns() * numRows();
    }

    template <typename T>
    bool Matrix<T>::isVector() const {
        return isVec;
    }

    template <typename T>
    std::vector<T> Matrix<T>::row(int row) const {
        std::vector<T> tempRow;
        tempRow.reserve(numColumns());
        for (int i = 0; i < numColumns(); ++i) {
            tempRow.push_back(at(row, i));
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) const {
        std::vector<T> tempCol;
        tempCol.reserve(numRows());
        for (int i = 0; i < numRows(); ++i) {
            tempCol.push_back(at(i, col));
        }
        return tempCol;
    }

    template <typename T>
    void Matrix<T>::write(std::ofstream& outFile) const {
        if (outFile.is_open()) {
            outFile << numRows() << "," << numColumns() << std::endl;
            int precision = 1;
            if (typeid(T) == typeid(double)) {
                precision = 15;
            } else if (typeid(T) == typeid(float)) {
                precision = 7;
            }
            for (int i = 0; i < elements.size() - 1; ++i) {
                outFile << std::fixed << std::setprecision(precision) << elements[i];
                outFile << ",";
            }
            outFile << elements.back() << std::endl;
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void Matrix<T>::read(std::ifstream& inFile) {
        if (inFile.is_open()) {
            // Declare temp variables.
            std::vector<std::string> tempElements;
            std::string temp;
            // Get size information.
            inFile >> temp;
            tempElements = strmanip::split(temp, ',');
            int rows = std::stoi(tempElements.at(0));
            int cols = std::stoi(tempElements.at(1));
            init(rows, cols);
            // Get elements.
            inFile >> temp;
            tempElements = strmanip::split(temp, ',');
            // Modify this matrix.
            for (int i = 0; i < tempElements.size(); ++i) {
                elements[i] = (T) std::stod(tempElements[i]);
            }
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

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
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tile[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute row and column of this block.
        int row = blockIdx.x * blockDim.x;
        int col = blockIdx.y * blockDim.y;
        // Load a (transposed) tile into shared memory.
        tile[threadIdx.y][threadIdx.x] = original[(row + threadIdx.x) * numCols + (col + threadIdx.y)];
        // Synchronize.
        __syncthreads();
        // Write the tiles into the output. Switch rows and columns to handle non-square matrices.
        transposed[(col + threadIdx.x) * numRows + (row + threadIdx.y)] = tile[threadIdx.x][threadIdx.y];
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
    __global__ void computeProduct(T* A, T* B, int numColsA, int numColsB, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A.
            tileA[threadIdx.x][threadIdx.y] = A[row * numColsA + (i + threadIdx.y)];
            // Load sub-matrix B.
            tileB[threadIdx.x][threadIdx.y] = B[(i + threadIdx.x) * numColsB + col];
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[threadIdx.x][j] * tileB[j][threadIdx.y];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[row * numColsB + col] = Cvalue;
    }

    template <typename T>
    __global__ void computeVectorProductRight(T* A, T* B, int numColsA, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A.
            tileA[threadIdx.x][threadIdx.y] = A[row * numColsA + (i + threadIdx.y)];
            // Load sub-matrix B - only load each element once.
            if (threadIdx.y == 0) {
                tileB[threadIdx.x] = B[i + threadIdx.x];
            }
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[threadIdx.x][j] * tileB[j];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[row] = Cvalue;
    }

    template <typename T>
    __global__ void computeVectorProductLeft(T* A, T* B, int numColsA, int numColsB, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A - only load each element once.
            if (threadIdx.x == 0) {
                tileA[threadIdx.y] = A[i + threadIdx.y];
            }
            // Load sub-matrix B.
            tileB[threadIdx.x][threadIdx.y] = B[(i + threadIdx.x) * numColsB + col];
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[j] * tileB[j][threadIdx.y];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[col] = Cvalue;
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
    __global__ void computeScalarProduct(T* A, T B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] * B;
    }

    template <typename T>
    __global__ void computeVectorScalarProduct(T* A, T B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] * B;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUScalarProduct(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] * other;
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] * other;
                }
            }
        }
        return output;
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
    __global__ void computeDotProduct(T* A, T* B, int numRows, int numColsRaw) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            int rowIndex = row * numColsRaw;
            int index = 0;
            T Cvalue = T();
            // numColsRaw is guaranteed to be a multiple of 32.
            for (int i = 0; i < numColsRaw; i += BLOCK_DIM) {
                index = rowIndex + i;
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    Cvalue += A[index] * B[index];
                    index++;
                }
            }
            A[row] = Cvalue;
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
    __global__ void computeSum(T* A, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] + B[index];
    }

    template <typename T>
    __global__ void computeVectorSum(T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] + B[index];
        }
    }

    template <typename T>
    __global__ void computeDifference(T* A, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] - B[index];
    }

    template <typename T>
    __global__ void computeVectorDifference(T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] - B[index];
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDifference(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        if (isVector()) {
            for (int i = 0; i < sizeRaw(); i += BLOCK_DIM) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        } else {
            for (int i = 0; i < sizeRaw(); i += THREADS_PER_BLOCK) {
                #pragma unroll
                for (int j = 0; j < THREADS_PER_BLOCK; ++j) {
                    output.data()[i + j] = data()[i + j] + other.data()[i + j];
                }
            }
        }
        return output;
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


    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
