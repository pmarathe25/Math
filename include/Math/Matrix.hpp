#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

namespace math {
    template <typename T>
    class Matrix {
        public:
            // Constructors.
            void init(int rows, int cols);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements, int rows, int cols);
            Matrix(const std::vector<std::vector<T> >& initialElements);
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
            // Getter functions.
            int numRowsRaw() const;
            int numColumnsRaw() const;
            int sizeRaw() const;
            int numRows() const;
            int numColumns() const;
            int size() const;
            std::vector<T> row(int row) const;
            std::vector<T> column(int col) const;
            // Computation functions.
            Matrix transpose() const;
            Matrix operator*(const Matrix& other);
        private:
            std::vector<T> elements;
            int rowsRaw, colsRaw, rows, cols;
    };

    template <typename T>
    void display(const Matrix<T>& toDisplay);

}

#endif
