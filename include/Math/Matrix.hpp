#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

namespace math {
    template <class T>
    class Matrix {
        public:
            Matrix(int rows, int cols) {
                elements.reserve(rows * cols);
                this -> rows = rows;
                this -> cols = cols;
            }
            Matrix(int rows, int cols, const std::vector<T>& initialElements) {
                elements = initialElements;
                this -> rows = rows;
                this -> cols = cols;
            }
            Matrix(const std::vector<std::vector<T> >& initialElements) {
                rows = initialElements.size();
                cols = initialElements.at(0).size();
                elements.reserve(rows * cols);
                for (int row = 0; row < initialElements.size(); ++row) {
                    for (int col = 0; col < initialElements.at(0).size(); ++col) {
                        elements.at(row * cols + col) = initialElements.at(row).at(col);
                    }
                }
            }
            T& at(row, col) {
                return elements.at(row * cols + col)
            }
        private:
            std::vector<T> elements;
            int rows, cols;
    };
}

#endif
