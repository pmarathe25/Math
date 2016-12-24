#include "Math/Matrix.hpp"
#include "Math/Math.hpp"
#define TILE_DIM 32

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
        return &at(0, 0);
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
    __global__ void computeTranspose(T* original, int* numRows, int* numCols, int* size, T* transposed) {
        __shared__ T tile[TILE_DIM][TILE_DIM];
        // Load a (transposed) tile into shared memory.
        int y = blockIdx.y * TILE_DIM + threadIdx.y;
        int x = blockIdx.x * TILE_DIM + threadIdx.x;
        if (x < *numRows && y < *numCols) {
            tile[threadIdx.y][threadIdx.x] = original[x * *numCols + y];
        } else {
            tile[threadIdx.y][threadIdx.x] = 0;
        }
        // Synchronize.
        __syncthreads();
        // Write the tiles into the output. Switch rows and columns to handle non-square matrices.
        y = blockIdx.x * TILE_DIM + threadIdx.y;
        x = blockIdx.y * TILE_DIM + threadIdx.x;
        if (x < *numCols && y < *numRows) {
            transposed[x * *numRows + y] = tile[threadIdx.x][threadIdx.y];
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() {
        int matSize = size();
        int numCols = numColumns();
        int numRow = numRows();
        T transpose[matSize];
        // Initialize device copies.
        T *dev_original, *dev_transposed;
        int *dev_numCols, *dev_numRows, *dev_size;
        // Allocate memory for device ccpies.
        cudaMalloc((void**)&dev_original, matSize * sizeof(T));
        cudaMalloc((void**)&dev_transposed, matSize * sizeof(T));
        cudaMalloc((void**)&dev_numRows, sizeof(int));
        cudaMalloc((void**)&dev_numCols, sizeof(int));
        cudaMalloc((void**)&dev_size, sizeof(int));
        // Copy inputs to device.
        cudaMemcpy(dev_original, data(), matSize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_numRows, &numRow, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_numCols, &numCols, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_size, &matSize, sizeof(int), cudaMemcpyHostToDevice);
        // Launch kernel.
        dim3 blocks(512, 512);
        dim3 threads(TILE_DIM, TILE_DIM);
        computeTranspose<<<blocks, threads>>>(dev_original, dev_numRows, dev_numCols, dev_size, dev_transposed);
        // Get result.
        cudaMemcpy(transpose, dev_transposed, matSize * sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_original);
        cudaFree(dev_transposed);
        cudaFree(dev_numRows);
        cudaFree(dev_numCols);
        cudaFree(dev_size);
        // Return.
        std::cout << std::endl;
        return Matrix<T>(transpose, numColumns(), numRows());
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) {
        Matrix product = Matrix(numRows(), other.numColumns());
        for (int j = 0; j < product.numColumns(); ++j) {
            std::vector<T> otherColumn = other.column(j);
            for (int i = 0; i < product.numRows(); ++i) {
                product.at(i, j) = innerProduct(row(i), otherColumn);
            }
        }
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
