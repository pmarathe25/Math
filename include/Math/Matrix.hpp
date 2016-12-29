#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

namespace math {
    template <typename T>
    class Matrix {
        public:
            void init(int rows, int cols);
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements, int rows, int cols);
            Matrix(const std::vector<std::vector<T> >& initialElements);
            T& at(int row, int col);
            T& at(int index);
            T* data();
            std::vector<T>& raw();
            const T* const_data() const;
            const std::vector<T>& const_raw() const;
            int numRowsRaw() const;
            int numColumnsRaw() const;
            int numRows() const;
            int numColumns() const;
            int size() const;
            std::vector<T> row(int row) const;
            std::vector<T> column(int col) const;
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
