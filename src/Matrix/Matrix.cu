#include "Math/Matrix.hpp"
#include "Text/strmanip.hpp"
#include <iostream>
#include <iomanip>
#include <typeinfo>
// Include matrix functions.
#include "MatrixCUDAFunctions.cu"
#include "MatrixAccessFunctions.cpp"
#include "MatrixModificationFunctions.cpp"
#include "MatrixComputationFunctions.cpp"

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
        this -> elements = other.data();
        this -> rows = other.numRows();
        this -> cols = other.numColumns();
        this -> matrixSize = rows * cols;
        isVec = (rows == 1) || (cols == 1);
        this -> elements = other.elements;
        other.elements = NULL;
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T>& other) {
        copy(other);
    }

    template <typename T>
    void Matrix<T>::operator=(Matrix other) {
        this -> elements = other.data();
        this -> rows = other.numRows();
        this -> cols = other.numColumns();
        this -> matrixSize = rows * cols;
        isVec = (rows == 1) || (cols == 1);
        this -> elements = other.elements;
        other.elements = NULL;
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
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    template <typename T>
    void Matrix<T>::write(std::ofstream& outFile) const {
        if (outFile.is_open()) {
            outFile << numRows() << "," << numColumns() << std::endl;
            // Set precision
            int precision = 1;
            if (typeid(T) == typeid(double)) {
                precision = 15;
            } else if (typeid(T) == typeid(float)) {
                precision = 7;
            }
            // Write elements
            for (int i = 0; i < size(); ++i) {
                outFile << std::fixed << std::setprecision(precision) << elements[i] << ",";
            }
            outFile << std::endl;
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void Matrix<T>::read(std::ifstream& inFile) {
        if (inFile.is_open()) {
            // Declare temp variables.
            std::vector<std::string> tempElements (1);
            // Get size information.
            inFile >> tempElements[0];
            tempElements = strmanip::split(tempElements[0], ',');
            int rows = std::stoi(tempElements[0]);
            int cols = std::stoi(tempElements[1]);
            init(rows, cols);
            // Get elements.
            inFile >> tempElements[0];
            tempElements = strmanip::split(tempElements[0], ',');
            // Modify this matrix.
            for (int i = 0; i < size(); ++i) {
                elements[i] = (T) std::stod(tempElements[i]);
            }
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
