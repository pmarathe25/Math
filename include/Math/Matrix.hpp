#ifndef MATRIX_H
#define MATRIX_H
#include "Math/Math.hpp"
#include <fstream>
#include <vector>

const int BLOCK_DIM = 32;
const int THREADS_PER_BLOCK = 1024;

namespace math {
    template <typename T>
    class Matrix {
        public:
            void init(int rows, int cols);
            Matrix() {}
            Matrix(T elem);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements);
            Matrix(const std::vector<T>& initialElements, int rows, int cols);
            Matrix(const std::vector<std::vector<T> >& initialElements);
            Matrix(const Matrix& other);
            template <typename O>
            Matrix(const Matrix<O>& other) {
                init(other.numRows(), other.numColumns());
                std::copy(other.data(), other.data() + size(), elements);
            }
            ~Matrix();
            // Indexing functions.
            T& at(int row, int col);
            const T& at(int row, int col) const;
            T& at(int index);
            const T& at(int index) const;
            // Unsafe indexing functions.
            T& operator[](int index);
            const T& operator[](int index) const;
            // Raw data functions.
            T* data();
            const T* data() const;
            // User-facing getter functions.
            int numRows() const;
            int numColumns() const;
            int size() const;
            bool isVector() const;
            // Display
            void display() const;
            // File I/O.
            void write(std::ofstream& outFile) const;
            void read(std::ifstream& inFile);
            // In-place modification
            void reshape(int rows, int cols);
            // Computation functions.
            void randomizeNormal(T mean = 0, T stdDev = 1);
            void randomizeUniform(T lowerBound = 0, T upperBound = 1);
            Matrix transpose() const;
            Matrix rowMean() const;
            Matrix hadamard(Matrix& other);
            Matrix rowWiseDot(const Matrix& other) const;
            Matrix operator*(const Matrix& other) const;
            Matrix operator*(T other) const;
            Matrix operator+(const Matrix& other) const;
            Matrix operator+(T other) const;
            Matrix operator-(const Matrix& other) const;
            Matrix operator-(T other) const;
        private:
            T* elements;
            int rows, cols, matrixSize;
            bool isVec = false;
            // Internal functions.
            Matrix CPUSum(const Matrix& other) const;
            Matrix CPUSum(T other) const;
            Matrix CPUMatrixVectorSum(const Matrix& other) const;
            Matrix CPUMatrixVectorDifference(const Matrix& other) const;
            Matrix CPUDifference(const Matrix& other) const;
            Matrix CPUDifference(T other) const;
            Matrix CPUScalarProduct(T other) const;
            Matrix CPUDotProduct(const Matrix& other) const;
            Matrix matrixArithmetic(const Matrix<T>& other, int mode) const;
            Matrix matrixTiledArithmetic(const Matrix<T>& other, int mode) const;
            Matrix scalarArithmetic(T other, int mode) const;
    };

    template <typename T, typename O>
    Matrix<T> operator*(O other, const Matrix<T>& A) {
        return A * other;
    }

    template <typename T, typename O>
    Matrix<T> operator-(O other, const Matrix<T>& A) {
        return (A * -1) + other;
    }
}

#endif
