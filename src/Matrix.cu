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
        int i = 0;
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                elements[row + col] = initialElements[i++];
            }
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        init(rows, cols);
        int rowInitial = 0;
        for (int row = 0; row < rows * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < cols; ++col) {
                elements[row + col] = initialElements[rowInitial][col];
            }
            ++rowInitial;
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
        temp.reserve(size());
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                temp.push_back(elements[row + col]);
            }
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
        int rowIndex = row * numColumnsRaw();
        for (int i = 0; i < numColumns(); ++i) {
            tempRow.push_back(elements[rowIndex + i]);
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) const {
        std::vector<T> tempCol;
        tempCol.reserve(numRows());
        for (int i = 0; i < numRows() * numColumnsRaw(); i += numColumnsRaw()) {
            tempCol.push_back(elements[i + col]);
        }
        return tempCol;
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
            for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
                for (int col = 0; col < numColumns(); ++col) {
                    // elements[row + col] = uniformDistribution(generator);
                    outFile << std::fixed << std::setprecision(precision) << elements[row + col] << ",";
                }
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
            int i = 0;
            for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
                for (int col = 0; col < numColumns(); ++col) {
                    // elements[row + col] = uniformDistribution(generator);
                    elements[row + col] = (T) std::stod(tempElements[i++]);
                }
            }

        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
