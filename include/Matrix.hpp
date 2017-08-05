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
    __global__ void powerCUDA(const T* A, int exponent, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = powf(A[index], exponent);
        }
    }

    template <typename T>
    class Matrix {
        public:
            void init(int rows, int cols);
            Matrix() {}
            Matrix(const std::string& filePath);
            Matrix(T elem);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements);
            Matrix(const std::vector<T>& initialElements, int rows, int cols = -1);
            Matrix(const std::vector<std::vector<T> >& initialElements);
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
            void display() const;
            // In-place modification
            void reshape(int rows, int cols);
            void set(T setValue);
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
            // Static functions for Matrix creation.
            static Matrix randomNormal(int rows, int cols, double mean = 0, double stdDev = 1);
            static Matrix randomNormalLike(const Matrix& like, double mean = 0, double stdDev = 1);
            static Matrix randomUniform(int rows, int cols, double lowerBound = 0, double upperBound = 1);
            static Matrix randomUniformLike(const Matrix& like, double lowerBound = 0, double upperBound = 1);
            static Matrix ones(int rows, int cols);
            static Matrix onesLike(const Matrix& like);
            static Matrix zeros(int rows, int cols);
            static Matrix zerosLike(const Matrix& like);
            static Matrix sequentialMatrix(int rows, int cols);
            // Loading from file.
            static Matrix load(const std::string& filePath);
            static Matrix load(std::ifstream& inFile);
        protected:
            T* elements = NULL;
        private:
            int rows = 0, cols = 0, matrixSize = 0;
            bool isVec = false;
            void copy(const Matrix& other) {
                if (elements) {
                    cudaFree(elements);
                }
                init(other.numRows(), other.numColumns());
                std::copy(other.data(), other.data() + size(), elements);
            }
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

    template <typename T>
    Matrix<T> pow(const Matrix<T>& input, int exponent) {
        Matrix<T> output(input.numRows(), input.numColumns());
        dim3 blocks(ceilDivide(output.size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        powerCUDA<<<blocks, threads>>>(input.data(), exponent, input.size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

}

typedef math::Matrix<int> Matrix;
typedef math::Matrix<float> Matrix_F;
typedef math::Matrix<double> Matrix_D;

#endif
