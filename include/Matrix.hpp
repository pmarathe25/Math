#ifndef MATRIX_H
#define MATRIX_H
#include "Math.hpp"
#include <fstream>
#include <vector>
#include <chrono>
#include <random>

const int BLOCK_DIM = 32;
const int THREADS_PER_BLOCK = 1024;

namespace math {
    template <typename T, T (func)(T)>
    __global__ void computeApplyFunction(T* A, int Asize, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            B[index] = func(A[index]);
        }
    }

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
            // Display
            void display() const;
            // File I/O.
            void write(std::ofstream& outFile) const;
            void read(std::ifstream& inFile);
            // In-place modification
            void reshape(int rows, int cols);
            // Unary functions.
            Matrix transpose() const;
            Matrix rowMean() const;
            // Matrix-Matrix Arithmetic
            Matrix dot(const Matrix& other) const;
            Matrix operator*(const Matrix& other) const;
            Matrix operator+(Matrix other) const;
            Matrix operator-(Matrix other) const;
            Matrix hadamard(Matrix other) const;
            // Matrix-Vector Arithmetic
            Matrix addVector(const Matrix& other) const;
            // Matrix-Scalar Arithmetic
            Matrix operator*(T other) const;
            Matrix operator+(T other) const;
            Matrix operator-(T other) const;
            // In place functions
            template <T (func)(T)>
            Matrix applyFunction() {
                Matrix output(numRows(), numColumns());
                dim3 blocks(std::ceil(size() / (float) THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                computeApplyFunction<T, func><<<blocks, threads>>>(data(), size(), output.data());
                cudaDeviceSynchronize();
                return output;
            }
        protected:
            T* elements = NULL;
        private:
            int rows, cols, matrixSize;
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
    Matrix<T> randomNormal(int rows, int cols, double mean, double stdDev) {
        Matrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> randomNormalLike(const Matrix<T>& like, double mean, double stdDev) {
        Matrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> randomUniform(int rows, int cols, double lowerBound, double upperBound) {
        Matrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> randomUniformLike(const Matrix<T>& like, double lowerBound, double upperBound) {
        Matrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }
}

#endif
