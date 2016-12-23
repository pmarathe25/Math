#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include "Math/Math.hpp"


namespace math {
    template <typename T>
    class Matrix {
        public:
            Matrix(int rows, int cols);
            Matrix(const std::vector<T>& initialElements, int rows, int cols);
            Matrix(const std::vector<std::vector<T> >& initialElements);
            T& at(int row, int col);
            const std::vector<std::vector<T> >& getElements() const;
            int getNumRows() const;
            int getNumColumns() const;
            const std::vector<T>& getRow(int row) const;
            std::vector<T> getColumn(int col) const;
            Matrix operator*(const Matrix& other);
        private:
            std::vector<std::vector<T> > elements;
    };

    template <typename T>
    void display(const math::Matrix<T>& toDisplay);

}

#endif
