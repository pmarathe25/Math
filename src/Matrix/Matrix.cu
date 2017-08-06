#include "Matrix.hpp"
#include <iostream>
#include <iomanip>
#include <typeinfo>
// Include matrix functions.
#include "MatrixCUDAFunctions.cu"
#include "MatrixAccessFunctions.cu"
#include "MatrixModificationFunctions.cu"
#include "MatrixComputationFunctions.cu"

namespace math {
    template <typename T>
    void Matrix<T>::init(int rows, int cols) {
        isVec = (rows == 1) || (cols == 1);
        this -> rows = rows;
        this -> cols = cols;
        this -> matrixSize = rows * cols;
        cudaMallocManaged(&elements, matrixSize * sizeof(T));
    }

    template <typename T>
    Matrix<T>::Matrix(const std::string& filePath) {
        (*this) = load(filePath);
    }

    template <typename T>
    Matrix<T>::Matrix(T elem) {
        init(1, 1);
        elements[0] = elem;
    }

    template <typename T>
    Matrix<T>::Matrix(int rows, int cols) {
        init(rows, cols);
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(1, initialElements.size());
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        std::copy(initialElements.data(), initialElements.data() + size(), elements);
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        cols = (cols == -1) ? initialElements.size() / rows : cols;
        init(rows, cols);
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        std::copy(initialElements.data(), initialElements.data() + size(), elements);
    }

    template <typename T>
    Matrix<T>::Matrix(Matrix&& other) {
        if (elements) {
            cudaFree(elements);
        }
        elements = other.elements;
        other.elements = NULL;
        rows = other.numRows();
        cols = other.numColumns();
        matrixSize = rows * cols;
        isVec = (rows == 1) || (cols == 1);
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T>& other) {
        init(other.numRows(), other.numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        copyCUDA<<<blocks, threads>>>(data(), other.data(), size());
        cudaDeviceSynchronize();
    }

    template <typename T>
    void Matrix<T>::operator=(Matrix<T> other) {
        if (elements) {
            cudaFree(elements);
        }
        elements = other.elements;
        other.elements = NULL;
        rows = other.numRows();
        cols = other.numColumns();
        matrixSize = rows * cols;
        isVec = (rows == 1) || (cols == 1);
    }

    template <typename T>
    Matrix<T>::~Matrix() {
        if (elements) {
            cudaFree(elements);
        }
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
        return matrixSize;
    }

    template <typename T>
    bool Matrix<T>::isVector() const {
        return isVec;
    }

    template <typename T>
    bool Matrix<T>::sizeMatches(const Matrix<T>& other) const {
        return numRows() == other.numRows() && numColumns() == other.numColumns();
    }

    template <typename T>
    void Matrix<T>::display() const {
        for (int i = 0; i < numRows(); ++i) {
            for (int j = 0; j < numColumns(); ++j) {
                std::cout << elements[i * numColumns() + j] << " ";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    template <typename T>
    void Matrix<T>::save(const std::string& filePath) const {
        std::ofstream saveFile(filePath, std::ios::binary);
        if (saveFile.is_open()) {
            save(saveFile);
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void Matrix<T>::save(std::ofstream& outFile) const {
        if (outFile.is_open()) {
            // Write metadata
            int currentRows = rows, currentCols = cols;
            outFile.write(reinterpret_cast<char*>(&currentRows), sizeof currentRows);
            outFile.write(reinterpret_cast<char*>(&currentCols), sizeof currentCols);
            // Write elements
            for (int i = 0; i < size(); ++i) {
                // outFile << std::hexfloat << elements[i] << '\\';
                outFile.write(reinterpret_cast<char*>(&elements[i]), sizeof elements[i]);
                // outFile.write('\\');
            }
            // outFile << '\n';
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
