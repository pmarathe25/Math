#include "Math/Matrix.hpp"
#include "Text/strmanip.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <typeinfo>
// Include matrix functions.
#include "MatrixCUDAFunctions.cu"
#include "MatrixCPUFunctions.cpp"
#include "MatrixMathFunctions.cpp"
#include "MatrixCUDACallFunctions.cpp"

namespace math {
    template <typename T>
    void Matrix<T>::init(int rows, int cols) {
        isVec = (rows == 1) || (cols == 1);
        this -> rows = rows;
        this -> cols = cols;
        this -> matrixSize = rows * cols;
        elements.reserve(rows * cols);
        // Allocate space on the GPU for this matrix.
        cudaMalloc((void**)&GPUPointer, matrixSize * sizeof(T));
        updateGPU = true;
    }

    template <typename T>
    Matrix<T>::Matrix() {
        init(0, 0);
    }

    template <typename T>
    Matrix<T>::Matrix(T elem) {
        init(1, 1);
        elements.push_back(elem);
    }

    template <typename T>
    Matrix<T>::Matrix(int rows, int cols) {
        // Zero-Initialize elements.
        init(rows, cols);
        elements = std::vector<T>(size());
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(1, initialElements.size());
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        elements = initialElements;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(rows, cols);
        if (size() != initialElements.size()) {
            throw std::invalid_argument("Matrix initialization dimension mismatch.");
        }
        elements = initialElements;
    }

    template <typename T>
    Matrix<T>::Matrix(const std::vector<std::vector<T> >& initialElements) {
        this -> rows = initialElements.size();
        this -> cols = initialElements.at(0).size();
        init(rows, cols);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                elements.push_back(initialElements[row][col]);
            }
        }
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T>& other) {
        rows = other.numRows();
        cols = other.numColumns();
        init(rows, cols);
        elements = other.raw();
        // Copy GPU data.
        copy<<<size() / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(other.dataGPU(), size(), dataGPU());
        updateGPU = other.isGPUCopyOld();
    }

    template <typename T>
    Matrix<T>::~Matrix() {
        cudaFree(GPUPointer);
    }

    // Indexing Functions.
    template <typename T>
    T& Matrix<T>::at(int row, int col) {
        if (row < numRows() && col < numColumns()) {
            updateCPUCopy();
            return elements[row * numColumns() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    const T& Matrix<T>::at(int row, int col) const {
        if (row < numRows() && col < numColumns()) {
            return elements[row * numColumns() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    T& Matrix<T>::at(int index) {
        updateCPUCopy();
        return elements.at(index);
    }

    template <typename T>
    const T& Matrix<T>::at(int index) const {
        return elements.at(index);
    }

    // Unsafe indexing functions.
    template <typename T>
    T& Matrix<T>::operator[](int index) {
        updateCPUCopy();
        return elements[index];
    }

    template <typename T>
    const T& Matrix<T>::operator[](int index) const {
        return elements[index];
    }

    template <typename T>
    T* Matrix<T>::data() {
        updateCPUCopy();
        return elements.data();
    }

    template <typename T>
    const T* Matrix<T>::data() const {
        return elements.data();
    }

    template <typename T>
    std::vector<T>& Matrix<T>::raw() {
        updateCPUCopy();
        return elements;
    }

    template <typename T>
    const std::vector<T>& Matrix<T>::raw() const {
        return elements;
    }

    template <typename T>
    T* Matrix<T>::dataGPU() {
        updateGPUCopy();
        // We assume that the GPU copy is now more up-to-date than the CPU copy.
        updateGPU = false;
        return GPUPointer;
    }

    template <typename T>
    const T* Matrix<T>::dataGPU() const {
        return GPUPointer;
    }

    template <typename T>
    void Matrix<T>::updateGPUCopy() const {
        if (updateGPU) {
            cudaMemcpy(GPUPointer, elements.data(), size() * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    template <typename T>
    void Matrix<T>::updateGPUCopy() {
        if (updateGPU) {
            cudaMemcpy(GPUPointer, elements.data(), size() * sizeof(T), cudaMemcpyHostToDevice);
            updateGPU = false;
        }
    }

    template <typename T>
    void Matrix<T>::updateGPUCopy(Matrix& other) {
        updateGPUCopy();
        other.updateGPUCopy();
    }

    template <typename T>
    void Matrix<T>::updateCPUCopy() {
        if (!isGPUCopyOld()) {
            cudaMemcpy(elements.data(), GPUPointer, size() * sizeof(T) , cudaMemcpyDeviceToHost);
            updateGPU = true;
        }
    }

    template <typename T>
    void Matrix<T>::updateCPUCopy(Matrix& other) {
        updateCPUCopy();
        other.updateCPUCopy();
    }

    template <typename T>
    bool Matrix<T>::isGPUCopyOld() const {
        return updateGPU;
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
    std::vector<T> Matrix<T>::row(int row) {
        updateCPUCopy();
        std::vector<T> tempRow;
        tempRow.reserve(numColumns());
        int rowIndex = row * numColumns();
        for (int i = 0; i < numColumns(); ++i) {
            tempRow.push_back(elements[rowIndex + i]);
        }
        return tempRow;
    }

    template <typename T>
    std::vector<T> Matrix<T>::column(int col) {
        updateCPUCopy();
        std::vector<T> tempCol;
        tempCol.reserve(numRows());
        for (int i = 0; i < numRows() * numColumns(); i += numColumns()) {
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
                elements.push_back((T) std::stod(tempElements[i]));
            }
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class Matrix<int>;
    template class Matrix<float>;
    template class Matrix<double>;
}
