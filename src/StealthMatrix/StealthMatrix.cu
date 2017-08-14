#include "StealthMatrix.hpp"
#include <iostream>
#include <iomanip>
#include <typeinfo>
// Include matrix functions.
#include "StealthMatrixCUDAFunctions.cu"
#include "StealthMatrixAccessFunctions.cu"
#include "StealthMatrixModificationFunctions.cu"
#include "StealthMatrixComputationFunctions.cu"

namespace StealthMath {
    template <typename T>
    void StealthMatrix<T>::init() {
        isVec = (rows == 1) || (cols == 1);
        this -> matrixSize = rows * cols;
        cudaMallocManaged(&elements, matrixSize * sizeof(T));
    }

    template <typename T>
    void StealthMatrix<T>::init(int rows, int cols) {
        isVec = (rows == 1) || (cols == 1);
        this -> rows = rows;
        this -> cols = cols;
        this -> matrixSize = rows * cols;
        cudaMallocManaged(&elements, matrixSize * sizeof(T));
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(const std::string& filePath) {
        load(filePath);
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(std::ifstream& file) {
        load(file);
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(T elem) {
        init(1, 1);
        elements[0] = elem;
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(int rows, int cols) {
        init(rows, cols);
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(const std::vector<T>& initialElements) {
        // Initialize elements with size (rowsRaw, colsRaw).
        init(1, initialElements.size());
        if (size() != initialElements.size()) {
            throw std::invalid_argument("StealthMatrix initialization dimension mismatch.");
        }
        std::copy(initialElements.data(), initialElements.data() + size(), elements);
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(const std::vector<T>& initialElements, int rows, int cols) {
        // Initialize elements with size (rowsRaw, colsRaw).
        cols = (cols == -1) ? initialElements.size() / rows : cols;
        init(rows, cols);
        if (size() != initialElements.size()) {
            throw std::invalid_argument("StealthMatrix initialization dimension mismatch.");
        }
        std::copy(initialElements.data(), initialElements.data() + size(), elements);
    }

    template <typename T>
    StealthMatrix<T>::StealthMatrix(StealthMatrix&& other) {
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
    StealthMatrix<T>::StealthMatrix(const StealthMatrix<T>& other) {
        init(other.numRows(), other.numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        copyCUDA<<<blocks, threads>>>(data(), other.data(), size());
        cudaDeviceSynchronize();
    }

    template <typename T>
    void StealthMatrix<T>::operator=(StealthMatrix<T> other) {
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
    StealthMatrix<T>::~StealthMatrix() {
        if (elements) {
            cudaFree(elements);
        }
    }

    template <typename T>
    int StealthMatrix<T>::numRows() const {
        return rows;
    }

    template <typename T>
    int StealthMatrix<T>::numColumns() const {
        return cols;
    }

    template <typename T>
    int StealthMatrix<T>::size() const {
        return matrixSize;
    }

    template <typename T>
    bool StealthMatrix<T>::isVector() const {
        return isVec;
    }

    template <typename T>
    bool StealthMatrix<T>::sizeMatches(const StealthMatrix<T>& other) const {
        return numRows() == other.numRows() && numColumns() == other.numColumns();
    }

    template <typename T>
    void StealthMatrix<T>::display(const std::string& title) const {
        std::cout << title << '\n';
        for (int i = 0; i < numRows(); ++i) {
            for (int j = 0; j < numColumns(); ++j) {
                std::cout << elements[i * numColumns() + j] << " ";
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

    template <typename T>
    void StealthMatrix<T>::save(const std::string& filePath) const {
        std::ofstream saveFile(filePath, std::ios::binary);
        if (saveFile.is_open()) {
            save(saveFile);
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void StealthMatrix<T>::save(std::ofstream& outFile) const {
        if (outFile.is_open()) {
            // Write metadata
            int currentRows = rows, currentCols = cols;
            outFile.write(reinterpret_cast<char*>(&currentRows), sizeof currentRows);
            outFile.write(reinterpret_cast<char*>(&currentCols), sizeof currentCols);
            // Write elements
            outFile.write(reinterpret_cast<const char*>(&elements[0]), sizeof(T) * size());
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void StealthMatrix<T>::load(const std::string& filePath) {
        std::ifstream saveFile(filePath, std::ios::binary);
        if (saveFile.is_open()) {
            load(saveFile);
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template <typename T>
    void StealthMatrix<T>::load(std::ifstream& inFile) {
        if (inFile.is_open()) {
            // Get metadata and initialize.
            inFile.read(reinterpret_cast<char*>(&rows), sizeof rows);
            inFile.read(reinterpret_cast<char*>(&cols), sizeof cols);
            init();
            // Get matrix data.
            inFile.read(reinterpret_cast<char*>(&elements[0]), sizeof(T) * size());
        } else {
            throw std::invalid_argument("Could not open file.");
        }
    }

    template class StealthMatrix<int>;
    template class StealthMatrix<char>;
    template class StealthMatrix<unsigned char>;
    template class StealthMatrix<float>;
    template class StealthMatrix<double>;
}
