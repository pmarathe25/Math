#include "Math/Matrix.hpp"
#include "Math/Math.hpp"
#define BLOCK_DIM 32
#define GRID_DIM 1024


namespace math {
    template <typename T>
    Matrix<T>::Matrix(int rows, int cols) {
        // Initialize elements with size (rows, cols).
        elements = std::vector<T> (rows * cols);
        this -> rows = rows;
        this -> cols = cols;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rows, cols).
        elements = initialElements;
        this -> rows = rows;
        this -> cols = cols;
    }

    template <typename T>
    Matrix<T>::Matrix(const T* initialElements, int rows, int cols) {
        // Initialize elements with size (rows, cols).
        elements = std::vector<T> (rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            elements.at(i) = initialElements[i];
        }
        this -> rows = rows;
        this -> cols = cols;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        elements = std::vector<T> (rows * cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                elements.at(row * cols + col) = initialElements.at(row).at(col);
            }
        }
    }

    template <typename T>
    T& Matrix<T>::at(int row, int col) {
        return elements.at(row * numColumns() + col);
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
    const T* Matrix<T>::const_data() const {
        return elements.data();
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
    int Matrix<T>::size() const {
        return numColumns() * numRows();
    }

    template <typename T>
    std::vector<T> Matrix<T>::row(int row) const {
        std::vector<T> tempRow;
        tempRow.reserve(numColumns());
        for (int i = 0; i < numColumns(); ++i) {
            tempRow.push_back(elements.at(row * numColumns() + i));
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) const {
        std::vector<T> tempCol;
        tempCol.reserve(numRows());
        for (int i = 0; i < numRows(); ++i) {
            tempCol.push_back(elements.at(i * numColumns() + col));
        }
        return tempCol;
    }

    template<typename T>
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        __shared__ T tile[BLOCK_DIM][BLOCK_DIM];
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
        // Write the tiles into the output. Switch rows and columns to handle non-square matrices.
        x = blockIdx.y * BLOCK_DIM + threadIdx.x;
        y = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if (x < numCols && y < numRows) {
            transposed[x * numRows + y] = tile[threadIdx.x][threadIdx.y];
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const {
        int matSize = size();
        Matrix<T> transpose = Matrix<T>(numColumns(), numRows());
        // Initialize device copies.
        T *dev_original, *dev_transposed;
        // Allocate memory for device ccpies.
        cudaMalloc((void**)&dev_original, matSize * sizeof(T));
        cudaMalloc((void**)&dev_transposed, matSize * sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_original, const_data(), matSize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel.
        dim3 blocks(GRID_DIM, GRID_DIM);
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeTranspose<<<blocks, threads>>>(dev_original, numRows(), numColumns(), dev_transposed);
        // Get result.
        cudaMemcpy(transpose.data(), dev_transposed, matSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_original);
        cudaFree(dev_transposed);
        // Return.
        return transpose;
    }

    template <typename T>
    __global__ void computeProduct(T* A, T* B, int numRowsA, int numColsA, int numRowsB, int numColsB, T* C) {
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * BLOCK_DIM + threadIdx.x;
        int col = blockIdx.y * BLOCK_DIM + threadIdx.y;
        T Cvalue = T();
        // Only compute if that value is within the C matrix.
        if (row < numRowsA && col < numColsB) {
            // Iterate over the sub-matrices of A and B.
            for (int i = 0; i < (numColsA + BLOCK_DIM - 1); i += BLOCK_DIM) {
                // Load sub-matrices.
                if (row < numRowsA && i < numColsA) {
                    tileA[threadIdx.x][threadIdx.y] = A[row * numColsA + i];
                } else {
                    tileA[threadIdx.x][threadIdx.y] = 0;
                }
                if (i < numRowsB && col < numColsB) {
                    tileB[threadIdx.x][threadIdx.y] = B[i * numColsB + col];
                } else {
                    tileB[threadIdx.x][threadIdx.y] = 0;
                }
                // Synchronize.
                __syncthreads();
                // Compute dot product.
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    Cvalue += tileA[threadIdx.x][j] * tileB[j][threadIdx.y];
                }
                // Synchronize.
                __syncthreads();
            }
            // Write to output.
            C[row * numColsB + col] = Cvalue;
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) {
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
        cudaMemcpy(dev_A, const_data(), Asize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, other.const_data(), Bsize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_C, product.const_data(), Csize * sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel.
        dim3 blocks(GRID_DIM, GRID_DIM);
        dim3 threads(BLOCK_DIM, BLOCK_DIM);
        computeProduct<<<blocks, threads>>>(dev_A, dev_B, numRows(), numColumns(), other.numRows(), other.numColumns(), dev_C);
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
        for (int i = 0; i < toDisplay.numRows(); ++i) {
            display(toDisplay.row(i));
        }
    }

    template void display(const Matrix<int>& toDisplay);
    template void display(const Matrix<float>& toDisplay);

}
