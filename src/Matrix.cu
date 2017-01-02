#include "Math/Matrix.hpp"
#include "Text/strmanip.hpp"
#include <iostream>
#include <iomanip>
// Include matrix functions.
#include "MatrixCUDAFunctions.cu"
#include "MatrixCPUFunctions.cpp"
#include "MatrixMathFunctions.cpp"

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
            (*this)[i] = initialElements[i];
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
            (*this)[i] = initialElements[i];
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        init(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                (*this)[row * cols + col] = initialElements[row][col];
            }
        }
    }

    // Indexing Functions.
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
        return at(index / numColumns(), index % numColumns());
    }

    template <typename T>
    const T& Matrix<T>::at(int index) const {
        return at(index / numColumns(), index % numColumns());
    }

    // Unsafe indexing functions.
    template <typename T>
    T& Matrix<T>::operator[](int index) {
        return elements[index / numColumns() * numColumnsRaw() + index % numColumns()];
    }

    template <typename T>
    const T& Matrix<T>::operator[](int index) const {
        return elements[index / numColumns() * numColumnsRaw() + index % numColumns()];
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
        int rowIndex = row * numColumns();
        for (int i = 0; i < numColumns(); ++i) {
            tempRow.push_back((*this)[rowIndex + i]);
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) const {
        std::vector<T> tempCol;
        tempCol.reserve(numRows());
        for (int i = 0; i < numRows(); ++i) {
            tempCol.push_back((*this)[i * numColumns() + col]);
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

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
