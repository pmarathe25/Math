#ifndef MATRIX_H
#define MATRIX_H
#include "Math.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

namespace math {
    template <typename T, T func(T)>
    __global__ void computeApplyFunction(const T* A, int Asize, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            B[index] = func(A[index]);
        }
    }

    template <typename T>
    class Matrix {
        public:
            void init();
            void init(int rows, int cols);
            Matrix() {}
            Matrix(const std::string& filePath);
            Matrix(T elem);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements);
            Matrix(const std::vector<T>& initialElements, int rows, int cols = -1);
            Matrix(Matrix&& other);
            Matrix(const Matrix& other);
            void operator=(Matrix other);
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
            bool sizeMatches(const Matrix& other) const;
            // Display
            void display(const std::string& title = "") const;
            // In-place modification
            Matrix& reshape(int rows, int cols = -1);
            Matrix& set(T setValue);
            // Unary functions.
            Matrix transpose() const;
            Matrix weightedSum(float scaleFactor = 1.0) const;
            // Matrix-Matrix Arithmetic
            Matrix dot(const Matrix& other) const;
            Matrix operator*(const Matrix& other) const;
            Matrix operator+(Matrix other) const;
            void operator+=(const Matrix& other);
            Matrix operator-(Matrix other) const;
            void operator-=(const Matrix& other);
            Matrix hadamard(Matrix other) const;
            // Matrix-Vector Arithmetic
            Matrix addVector(const Matrix& other) const;
            // Matrix-Scalar Arithmetic
            Matrix operator*(T other) const;
            Matrix operator/(T other) const;
            Matrix operator+(T other) const;
            Matrix operator-(T other) const;
            Matrix pow(int exponent);
            // Type conversion.
            template <typename O>
            Matrix<O> asType() const {
                Matrix<O> output(numRows(), numColumns());
                for (int i = 0; i < size(); ++i) {
                    output[i] = (O)(*this)[i];
                }
                return output;
            }
            // In place functions
            template <T func(T)>
            Matrix applyFunction() const {
                Matrix output(numRows(), numColumns());
                dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                computeApplyFunction<T, func><<<blocks, threads>>>(data(), size(), output.data());
                cudaDeviceSynchronize();
                return output;
            }
            // File I/O.
            void save(const std::string& filePath) const;
            void save(std::ofstream& outFile) const;
            void load(const std::string& filePath);
            void load(std::ifstream& inFile);
            // Static functions for Matrix creation.
            static Matrix randomNormal(int rows, int cols, double mean = 0, double stdDev = 1);
            static Matrix randomNormalLike(const Matrix& like, double mean = 0, double stdDev = 1);
            static Matrix randomUniform(int rows, int cols, double lowerBound = 0, double upperBound = 1);
            static Matrix randomUniformLike(const Matrix& like, double lowerBound = 0, double upperBound = 1);
            static Matrix ones(int rows, int cols);
            static Matrix onesLike(const Matrix& like);
            static Matrix zeros(int rows, int cols);
            static Matrix zerosLike(const Matrix& like);
            static Matrix sequential(int rows, int cols, int start = 0);
            static Matrix sequentialLike(const Matrix& like, int start = 0);
        protected:
            T* elements = NULL;
        private:
            static int ceilDivide(int x, int y);
            int rows = 0, cols = 0, matrixSize = 0;
            bool isVec = false;
    };

    template <typename T, typename O>
    Matrix<T> operator*(O other, const Matrix<T>& A) {
        return A * other;
    }

    template <typename T, typename O>
    Matrix<T> operator-(O other, const Matrix<T>& A) {
        return (A * -1) + other;
    }

    template <typename T, typename O>
    Matrix<T> operator+(O other, const Matrix<T>& A) {
        return A + other;
    }
}

typedef math::Matrix<int> Matrix;
typedef math::Matrix<char> Matrix_C;
typedef math::Matrix<unsigned char> Matrix_UC;
typedef math::Matrix<float> Matrix_F;
typedef math::Matrix<double> Matrix_D;

#endif
