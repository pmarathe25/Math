#include "Math/Matrix.hpp"
#include "Math/Math.hpp"
#define BLOCK_DIM 32
#define THREADS_PER_BLOCK 1024

namespace math {
    template <typename T>
    Matrix<T>::Matrix(int rowsRaw, int colsRaw) {
        // Initialize elements with size (rowsRaw, colsRaw).
        elements = std::vector<T> (rowsRaw * colsRaw);
        this -> rowsRaw = rowsRaw;
        this -> colsRaw = colsRaw;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rowsRaw, int colsRaw) {
        // Initialize elements with size (rowsRaw, colsRaw).
        elements = initialElements;
        this -> rowsRaw = rowsRaw;
        this -> colsRaw = colsRaw;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rowsRaw = initialElements.size();
        this -> colsRaw = initialElements.at(0).size();
        elements = std::vector<T> (rowsRaw * colsRaw);
        for (int row = 0; row < rowsRaw; ++row) {
            for (int col = 0; col < colsRaw; ++col) {
                elements.at(row * colsRaw + col) = initialElements.at(row).at(col);
            }
        }
    }

    template <typename T>
    T& Matrix<T>::at(int row, int col) {
        return elements.at(row * numColumnsRaw() + col);
    }

    template <typename T>
    T& Matrix<T>::at(int index) {
        return elements.at(index);
    }

    template <typename T>
    T* Matrix<T>::data() {
        return elements.data();
    }

    template <typename T>
    std::vector<T>& Matrix<T>::raw() {
        return elements;
    }

    template <typename T>
    const T* Matrix<T>::const_data() const {
        return elements.data();
    }

    template <typename T>
    const std::vector<T>& Matrix<T>::const_raw() const {
        return elements;
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
    int Matrix<T>::size() const {
        return numColumnsRaw() * numRowsRaw();
    }

    template <typename T>
    std::vector<T> Matrix<T>::row(int row) const {
        std::vector<T> tempRow;
        tempRow.reserve(numColumnsRaw());
        for (int i = 0; i < numColumnsRaw(); ++i) {
            tempRow.push_back(elements.at(row * numColumnsRaw() + i));
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) const {
        std::vector<T> tempCol;
        tempCol.reserve(numRowsRaw());
        for (int i = 0; i < numRowsRaw(); ++i) {
            tempCol.push_back(elements.at(i * numColumnsRaw() + col));
        }
        return tempCol;
    }

    template<typename T>
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        __shared__ T tile[BLOCK_DIM][BLOCK_DIM + 1];
        // Load a (transposed) tile into shared memory.
        int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
        int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
        if (x < numRows && y < numCols) {
            tile[threadIdx.y][threadIdx.x] = original[x * numCols + y];
        } else {
            tile[threadIdx.y][threadIdx.x] = 0;
        }
        // Synchronize.
        __syncthreads();
        // Write the tiles into the output. Switch rowsRaw and columns to handle non-square matrices.
        x = blockIdx.y * BLOCK_DIM + threadIdx.x;
        y = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if (x < numCols && y < numRows) {
            transposed[x * numRows + y] = tile[threadIdx.x][threadIdx.y];
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const {
        int matSize = size();
        Matrix<T> transpose = Matrix<T>(numColumnsRaw(), numRowsRaw());
        // Initialize device copies.
        T *dev_original, *dev_transposed;
        // Allocate memory for device ccpies.
        cudaMalloc((void**)&dev_original, matSize * sizeof(T));
        cudaMalloc((void**)&dev_transposed, matSize * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_original, const_data(), matSize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel with only as many blocks as necessary.
        int numBlocks = std::ceil(max(numColumnsRaw(), numRowsRaw()) / (double) BLOCK_DIM);
        dim3 blocks(numBlocks, numBlocks);
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeTranspose<<<blocks, threads>>>(dev_original, numRowsRaw(), numColumnsRaw(), dev_transposed);
        // Get result.
        cudaMemcpy(transpose.data(), dev_transposed, matSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_original);
        cudaFree(dev_transposed);
        // Return.
        return transpose;
    }

    template <typename T>
    __global__ void computeProduct(T* A, T* B, int numRowsA, int numColsA, int numRowsB, int numColsB, int Asize, int Bsize, T* C) {
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
        int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
        bool cValid = row < numRowsA && col < numColsB;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        int maxIterations = numColsA + BLOCK_DIM - 1;
        for (int i = 0; i < maxIterations; i += BLOCK_DIM) {
            // Compute indices.
            int indexA = row * numColsA + (i + threadIdx.y);
            int indexB = (i + threadIdx.x) * numColsB + col;
            // Load sub-matrix A.
            if (indexA < Asize) {
                tileA[threadIdx.x][threadIdx.y] = A[indexA];
            } else {
                tileA[threadIdx.x][threadIdx.y] = 0;
            }
            // Load sub-matrix B.
            if (indexB < Bsize) {
                tileB[threadIdx.x][threadIdx.y] = B[indexB];
            } else {
                tileB[threadIdx.x][threadIdx.y] = 0;
            }
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            if (cValid) {
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    Cvalue += tileA[threadIdx.x][j] * tileB[j][threadIdx.y];
                }
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        if (cValid) {
            C[row * numColsB + col] = Cvalue;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) {
        if (numColumnsRaw() != other.numRowsRaw()) {
            throw std::invalid_argument("Incompatible matrices cannot be multiplied.");
        }
        Matrix product = Matrix(numRowsRaw(), other.numColumnsRaw());
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
        cudaMemcpy(dev_A, const_data(), Asize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.const_data(), Bsize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_C, product.const_data(), Csize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel with only as many blocks as necessary.
        dim3 blocks(std::ceil(product.numRowsRaw() / (double) BLOCK_DIM), std::ceil(product.numColumnsRaw() / (double) BLOCK_DIM));
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeProduct<<<blocks, threads>>>(dev_A, dev_B, numRowsRaw(), numColumnsRaw(), other.numRowsRaw(), other.numColumnsRaw(), Asize, Bsize, dev_C);
        // Get result.
        cudaMemcpy(product.data(), dev_C, Csize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_A);
        cudaFree(dev_B);
        cudaFree(dev_C);
        // Return.
        return product;
    }

    template class Matrix<int>;
    template class Matrix<float>;

    template <typename T>
    void display(const Matrix<T>& toDisplay) {
        for (int i = 0; i < toDisplay.numRowsRaw(); ++i) {
            display(toDisplay.row(i));
        }
    }

    template void display(const Matrix<int>& toDisplay);
    template void display(const Matrix<float>& toDisplay);

}
