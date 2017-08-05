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
        for (int i = 0; i < matrixSize; ++i) {
            elements[i] = initialElements[i];
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        cols = (cols == -1) ? initialElements.size() / rows : cols;
        init(rows, cols);
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        for (int i = 0; i < matrixSize; ++i) {
            elements[i] = initialElements[i];
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        init(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                elements[row * cols + col] = initialElements[row][col];
            }
        }
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
        copy(other);
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
        std::ofstream saveFile(filePath);
        if (saveFile.is_open()) {
            save(saveFile);
            saveFile.close();
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void Matrix<T>::save(std::ofstream& outFile) const {
        if (outFile.is_open()) {
            outFile << std::hex << numRows() << "\\" << numColumns() << '\n';
            // Write elements
            for (int i = 0; i < size(); ++i) {
                outFile << elements[i] << '\\';
            }
            outFile << '\n';
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
