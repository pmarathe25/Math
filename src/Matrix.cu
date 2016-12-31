#include "Math/Matrix.hpp"
#include <iostream>
#define BLOCK_DIM 32

namespace math {

    template <typename T>
    void Matrix<T>::init(int rows, int cols) {
        int rowsPadded = rows;
        int colsPadded = cols;
        if (rows % BLOCK_DIM != 0) {
            rowsPadded += BLOCK_DIM - (rows % BLOCK_DIM);
        }
        if (cols % BLOCK_DIM != 0) {
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
    Matrix<T>::Matrix(int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(rows, cols);
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(rows, cols);
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
        return elements.at(row * numColumnsRaw() + col);
    }

    template <typename T>
    const T& Matrix<T>::at(int row, int col) const {
        return elements.at(row * numColumnsRaw() + col);
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

    template<typename T>
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tile[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute row and column of this block.
        int row = blockIdx.x * BLOCK_DIM;
        int col = blockIdx.y * BLOCK_DIM;
        // Load a (transposed) tile into shared memory.
        tile[threadIdx.y][threadIdx.x] = original[(row + threadIdx.x) * numCols + (col + threadIdx.y)];
        // Synchronize.
        __syncthreads();
        // Write the tiles into the output. Switch rows and columns to handle non-square matrices.
        transposed[(col + threadIdx.x) * numRows + (row + threadIdx.y)] = tile[threadIdx.x][threadIdx.y];
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const {
        int size = sizeRaw();
        Matrix<T> transpose = Matrix<T>(numColumns(), numRows());
        // Initialize device copies.
        T *dev_original, *dev_transposed;
        // Allocate memory for device ccpies.
        cudaMalloc((void**)&dev_original, size * sizeof(T));
        cudaMalloc((void**)&dev_transposed, size * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_original, data(), size * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel with only as many blocks as necessary.
        dim3 blocks(numRowsRaw() / BLOCK_DIM, numColumnsRaw() / BLOCK_DIM);
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeTranspose<<<blocks, threads>>>(dev_original, numRowsRaw(), numColumnsRaw(), dev_transposed);
        // Get result.
        cudaMemcpy(transpose.data(), dev_transposed, size * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_original);
        cudaFree(dev_transposed);
        // Return.
        return transpose;
    }

    template <typename T>
    __global__ void computeProduct(T* A, T* B, int numColsA, int numColsB, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
        int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
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
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
        if (numColumns() != other.numRows()) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        }
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
        dim3 blocks(product.numRowsRaw() / BLOCK_DIM, product.numColumnsRaw() / BLOCK_DIM);
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeProduct<<<blocks, threads>>>(dev_A, dev_B, numColumnsRaw(), other.numColumnsRaw(), dev_C);
        // Get result.
        cudaMemcpy(product.data(), dev_C, Csize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        // Return.
        return product;
    }

    template <typename T>
    __global__ void computeScalarProduct(T* A, T B, T* C) {
        C[blockIdx.x * BLOCK_DIM + threadIdx.x] = A[blockIdx.x * BLOCK_DIM + threadIdx.x] * B;
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
        dim3 blocks(sizeRaw() / BLOCK_DIM);
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
        C[blockIdx.x * BLOCK_DIM + threadIdx.x] = A[blockIdx.x * BLOCK_DIM + threadIdx.x] + B[blockIdx.x * BLOCK_DIM + threadIdx.x];
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
        if (size() != other.size()) {
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
        dim3 blocks(sizeRaw() / BLOCK_DIM);
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
        C[blockIdx.x * BLOCK_DIM + threadIdx.x] = A[blockIdx.x * BLOCK_DIM + threadIdx.x] - B[blockIdx.x * BLOCK_DIM + threadIdx.x];
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
        if (size() != other.size()) {
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
        dim3 blocks(sizeRaw() / BLOCK_DIM);
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
