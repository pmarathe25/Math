#ifndef MATRIX_H
#define MATRIX_H
#include "Math.hpp"
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

namespace StealthMath {
    template <typename T, T func(T)>
    __global__ void computeApplyFunction(const T* A, int Asize, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            B[index] = func(A[index]);
        }
    }

    template <typename T>
    class StealthMatrix {
        public:
            void init();
            void init(int rows, int cols);
            StealthMatrix() {}
            StealthMatrix(const std::string& filePath);
            StealthMatrix(std::ifstream& file);
            StealthMatrix(T elem);
            StealthMatrix(int rows, int cols);
            StealthMatrix(const std::vector<T>& initialElements);
            StealthMatrix(const std::vector<T>& initialElements, int rows, int cols = -1);
            StealthMatrix(StealthMatrix&& other);
            StealthMatrix(const StealthMatrix& other);
            void operator=(StealthMatrix other);
            virtual ~StealthMatrix();
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
            bool sizeMatches(const StealthMatrix& other) const;
            // Display
            void display(const std::string& title = "") const;
            // In-place modification
            StealthMatrix& reshape(int rows, int cols = -1);
            StealthMatrix& set(T setValue);
            // Unary functions.
            StealthMatrix transpose() const;
            StealthMatrix weightedRowSum(float scaleFactor = 1.0) const;
            StealthMatrix rowMean() const;
            StealthMatrix argmax(int axis = 1);
            // StealthMatrix-StealthMatrix Arithmetic
            StealthMatrix dot(const StealthMatrix& other) const;
            StealthMatrix operator*(const StealthMatrix& other) const;
            StealthMatrix operator+(StealthMatrix other) const;
            void operator+=(const StealthMatrix& other);
            StealthMatrix operator-(StealthMatrix other) const;
            void operator-=(const StealthMatrix& other);
            StealthMatrix hadamard(StealthMatrix other) const;
            // StealthMatrix-Vector Arithmetic
            StealthMatrix addVector(const StealthMatrix& other) const;
            // StealthMatrix-Scalar Arithmetic
            StealthMatrix operator*(T other) const;
            StealthMatrix operator/(T other) const;
            StealthMatrix operator+(T other) const;
            StealthMatrix operator-(T other) const;
            StealthMatrix pow(int exponent);
            // Type conversion.
            template <typename O>
            StealthMatrix<O> asType() const {
                StealthMatrix<O> output(numRows(), numColumns());
                for (int i = 0; i < size(); ++i) {
                    output[i] = (O)(*this)[i];
                }
                return output;
            }
            // In place functions
            template <T func(T)>
            StealthMatrix applyFunction() const {
                StealthMatrix output(numRows(), numColumns());
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
            // Static functions for StealthMatrix creation.
            static StealthMatrix randomNormal(int rows, int cols, double mean = 0, double stdDev = 1);
            static StealthMatrix randomNormalLike(const StealthMatrix& like, double mean = 0, double stdDev = 1);
            static StealthMatrix randomUniform(int rows, int cols, double lowerBound = 0, double upperBound = 1);
            static StealthMatrix randomUniformLike(const StealthMatrix& like, double lowerBound = 0, double upperBound = 1);
            static StealthMatrix ones(int rows, int cols);
            static StealthMatrix onesLike(const StealthMatrix& like);
            static StealthMatrix zeros(int rows, int cols);
            static StealthMatrix zerosLike(const StealthMatrix& like);
            static StealthMatrix sequential(int rows, int cols, int start = 0);
            static StealthMatrix sequentialLike(const StealthMatrix& like, int start = 0);
        private:
            T* elements = NULL;
            static int ceilDivide(int x, int y);
            int rows = 0, cols = 0, matrixSize = 0;
            bool isVec = false;
    };

    template <typename T, typename O>
    StealthMatrix<T> operator*(O other, const StealthMatrix<T>& A) {
        return A * other;
    }

    template <typename T, typename O>
    StealthMatrix<T> operator-(O other, const StealthMatrix<T>& A) {
        return (A * -1) + other;
    }

    template <typename T, typename O>
    StealthMatrix<T> operator+(O other, const StealthMatrix<T>& A) {
        return A + other;
    }
}

typedef StealthMath::StealthMatrix<int> StealthMatrix;
typedef StealthMath::StealthMatrix<char> StealthMatrix_C;
typedef StealthMath::StealthMatrix<unsigned char> StealthMatrix_UC;
typedef StealthMath::StealthMatrix<float> StealthMatrix_F;
typedef StealthMath::StealthMatrix<double> StealthMatrix_D;

#endif
