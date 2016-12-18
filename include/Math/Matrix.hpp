#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include "Math/Math.hpp"


namespace math {
    template <class T>
    class Matrix {
        public:
            Matrix(int rows, int cols) {
                // Initialize elements with size (rows, cols).
                elements = std::vector<std::vector<T> > (rows, std::vector<T>(cols));
            }
            Matrix(const std::vector<T>& initialElements, int rows, int cols) {
                // Initialize elements with size (rows, cols).
                elements = std::vector<std::vector<T> > (rows, std::vector<T>(cols));
                for (int row = 0; row < rows; ++row) {
                    for (int col = 0; col < cols; ++col) {
                        elements.at(row).at(col) = initialElements.at(row * cols + col);
                    }
                }
            }

            Matrix(const std::vector<std::vector<T> >& initialElements) {
                elements = initialElements;
            }

            T& at(int row, int col) {
                return elements.at(row).at(col);
            }

            const T& getElements() const {
                return elements;
            }

            int getNumRows() const {
                return elements.size();
            }

            int getNumColumns() const {
                return elements.at(0).size();
            }

            const std::vector<T>& getRow(int row) const {
                return elements.at(row);
            }

            std::vector<T> getColumn(int col) const {
                std::vector<T> temp;
                temp.reserve(elements.size());
                for (typename std::vector<std::vector<T> >::const_iterator it = elements.begin(); it != elements.end(); ++it) {
                    temp.push_back((*it).at(col));
                }
                return temp;
            }

            Matrix operator*(const Matrix& other) {
                Matrix product = Matrix(getNumRows(), other.getNumColumns());
                for (int j = 0; j < product.getNumColumns(); ++j) {
                    std::vector<T> otherColumn = other.getColumn(j);
                    for (int i = 0; i < product.getNumRows(); ++i) {
                        product.at(i, j) = innerProduct(getRow(i), otherColumn);
                    }
                }
                return product;
            }

        private:
            std::vector<std::vector<T> > elements;
    };
    template <typename T>
    void display(const math::Matrix<T>& toDisplay) {
        for (int i = 0; i < toDisplay.getNumRows(); ++i) {
            display(toDisplay.getRow(i));
        }
    }

}

#endif
