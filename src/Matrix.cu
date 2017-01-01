#include "Math/Matrix.hpp"
#include "Text/strmanip.hpp"
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <iomanip>
#include <typeinfo>

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
        elements.at(0) = elem;
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
            return elements.at(row * numColumnsRaw() + col);
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    const T& Matrix<T>::at(int row, int col) const {
        if (row < numRows() && col < numColumns()) {
            return elements.at(row * numColumnsRaw() + col);
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
                outFile << std::fixed << std::setprecision(precision) << elements.at(i);
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
                elements.at(i) = (T) std::stod(tempElements.at(i));
            }
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }


    template <typename T>
    void Matrix<T>::randomizeNormal() {
        randomizeNormal(0, 1);
    }

    template <typename T>
    void Matrix<T>::randomizeNormal(T mean, T stdDev) {
        randomize(mean, stdDev, NORMAL);
    }


    template <typename T>
    void Matrix<T>::randomizeUniform() {
        randomizeUniform(0, 1);
    }

    template <typename T>
    void Matrix<T>::randomizeUniform(T lowerBound, T upperBound) {
        randomize(lowerBound, upperBound, UNIFORM);
    }

    template <typename T>
    __global__ void randomizeMatrixNormal(unsigned long seed, T* mat, T mean, T stdDev, int cols, int colsRaw, int unpaddedSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        curandState_t state;
        curand_init(seed, index, 0, &state);
        if (index < unpaddedSize && (index % colsRaw) < cols) {
            printf("%d\n", index);
            mat[index] = curand_normal(&state) * stdDev + mean;
        } else {
            mat[index] = 0;
        }
    }

    template <typename T>
    __global__ void randomizeMatrixUniform(unsigned long seed, T* mat, T lowerBound, T upperBound, int cols, int colsRaw, int unpaddedSize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        curandState_t state;
        curand_init(seed, index, index, &state);
        if (index < unpaddedSize && (index % colsRaw) < cols) {
            mat[index] = curand_uniform(&state) * (upperBound - lowerBound) + lowerBound;
        } else{
            mat[index] = 0;
        }
    }

    template <typename T>
    void Matrix<T>::randomize(T param1, T param2, randMode mode) {
        int size = sizeRaw();
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        // Initialize device copies.
        T *dev_mat;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_mat, size * sizeof(T));
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(size / (float) BLOCK_DIM));
        dim3 threads(BLOCK_DIM);
        switch (mode) {
            case UNIFORM:
                randomizeMatrixUniform<<<blocks, threads>>>(value.count(), dev_mat, param1, param2, numColumns(), numColumnsRaw(), numColumnsRaw() * numRows());
                break;
            case NORMAL:
                randomizeMatrixNormal<<<blocks, threads>>>(value.count(), dev_mat, param1, param2, numColumns(), numColumnsRaw(), numColumnsRaw() * numRows());
                break;
        }
        // Get result.
        cudaMemcpy(data(), dev_mat, size * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_mat);
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
            // If there is only one element in on the matrices, do scalar multiplication instead.
            return other * at(0);
        } else if (other.size() == 1) {
            return *this * other.at(0);
        } else if (isVector() && other.isVector()) {
            // If both are vectors, we just need to return the dot product.
            return Matrix<T>(math::innerProduct(raw(), other.raw()));
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
    __global__ void computeScalarProduct(T* A, T B, T* C) {
        C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] * B;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(T other) const {
        Matrix product = Matrix(numRows(), numColumns());
        int size = sizeRaw();
        // Initialize device copies.
        T *dev_A, *dev_C;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, size * sizeof(T));
        cudaMalloc((void**)&dev_C, size * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), size * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(size / (float) BLOCK_DIM));
        dim3 threads(BLOCK_DIM);
        computeScalarProduct<<<blocks, threads>>>(dev_A, other, dev_C);
        // Get result.
        cudaMemcpy(product.data(), dev_C, size * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_C);
        // Return.
        return product;
    }

    template <typename T>
    __global__ void computeSum(T* A, T* B, T* C) {
        C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] + B[blockIdx.x * blockDim.x + threadIdx.x];
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
        Matrix sum = Matrix(numRows(), numColumns());
        int size = sizeRaw();
        // Initialize device copies.
        T *dev_A, *dev_B, *dev_C;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, size * sizeof(T));
        cudaMalloc((void**)&dev_B, size * sizeof(T));
        cudaMalloc((void**)&dev_C, size * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(size / (float) BLOCK_DIM));
        dim3 threads(BLOCK_DIM);
        computeSum<<<blocks, threads>>>(dev_A, dev_B, dev_C);
        // Get result.
        cudaMemcpy(sum.data(), dev_C, size * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        // Return.
        return sum;
    }

    template <typename T>
    __global__ void computeDifference(T* A, T* B, T* C) {
        C[blockIdx.x * blockDim.x + threadIdx.x] = A[blockIdx.x * blockDim.x + threadIdx.x] - B[blockIdx.x * blockDim.x + threadIdx.x];
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        if (numColumns() != other.numColumns() || numRows() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be added.");
        }
        Matrix sum = Matrix(numRows(), numColumns());
        int size = sizeRaw();
        // Initialize device copies.
        T *dev_A, *dev_B, *dev_C;
        // Allocate memory for device copies.
        cudaMalloc((void**)&dev_A, size * sizeof(T));
        cudaMalloc((void**)&dev_B, size * sizeof(T));
        cudaMalloc((void**)&dev_C, size * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_A, data(), size * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel where numThreads = size of matrix.
        dim3 blocks(std::ceil(size / (float) BLOCK_DIM));
        dim3 threads(BLOCK_DIM);
        computeDifference<<<blocks, threads>>>(dev_A, dev_B, dev_C);
        // Get result.
        cudaMemcpy(sum.data(), dev_C, size * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        // Return.
        return sum;
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
