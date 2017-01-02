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
            enum randMode {
                UNIFORM = 0,
                NORMAL,
            };
            enum opMode {
                SUM = 0,
                DIFFERENCE,
            };
            // Constructors.
            void init(int rows, int cols);
            Matrix();
            Matrix(T elem);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements);
            Matrix(const std::vector<T>& initialElements, int rows, int cols);
            Matrix(const std::vector<std::vector<T> >& initialElements);
            template <typename O>
            Matrix(const Matrix<O>& other) {
                rows = other.numRows();
                cols = other.numColumns();
                init(rows, cols);
                elements = std::vector<T>(other.raw().begin(), other.raw().end());
            }
            // Indexing functions.
            T& at(int row, int col);
            const T& at(int row, int col) const;
            T& at(int index);
            const T& at(int index) const;
            // Raw data functions.
            T* data();
            const T* data() const;
            std::vector<T>& raw();
            const std::vector<T>& raw() const;
            std::vector<T> getElements() const;
            // Getter functions for the underlying data.
            int numRowsRaw() const;
            int numColumnsRaw() const;
            int sizeRaw() const;
            // User-facing getter functions.
            int numRows() const;
            int numColumns() const;
            int size() const;
            bool isVector() const;
            std::vector<T> row(int row) const;
            std::vector<T> column(int col) const;
            // File I/O.
            void write(std::ofstream& outFile) const;
            void read(std::ifstream& inFile);
            // Computation functions.
            void randomizeNormal(T mean = 0, T stdDev = 1);
            void randomizeUniform(T lowerBound = 0, T upperBound = 1);
            Matrix& transpose();
            Matrix operator*(const Matrix& other) const;
            Matrix operator*(T other) const;
            Matrix operator+(const Matrix& other) const;
            Matrix operator-(const Matrix& other) const;
        private:
            std::vector<T> elements;
            int rowsRaw, colsRaw, rows, cols;
            bool isVec = false;
            // Internal functions.
            void randomize(T param1, T param2, randMode mode);
            Matrix arithmetic(const Matrix<T>& other, opMode mode) const;
    };

    template <typename T>
    void display(const Matrix<T>& toDisplay) {
        for (int i = 0; i < toDisplay.numRows(); ++i) {
            display(toDisplay.row(i));
        }
    }

    template <typename T, typename O>
    Matrix<T> operator*(O other, const Matrix<T>& A) {
        return A * other;
    }

    template <typename T>
    bool operator==(const Matrix<T>& A, const Matrix<T>& B) {
        return (A.numRows() == B.numRows() && A.numColumns() == B.numColumns() && A.getElements() == B.getElements());
    }

    template <typename T, typename O>
    bool operator==(const Matrix<T>& A, const Matrix<O>& B) {
        return false;
    }

}

#endif
