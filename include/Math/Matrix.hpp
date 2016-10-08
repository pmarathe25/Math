#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <iostream>
#include "Math/Math.hpp"

namespace math {
    template <class T>
    class Matrix {
        public:
            Matrix(int rows, int cols) {
                // Initialize elements with size (rows, cols).
                elements = std::vector<std::vector<T> > (rows, std::vector<T>(cols));
            }
            Matrix(int rows, int cols, const std::vector<T>& initialElements) {
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
            const T& getElements() {
                return elements;
            }
            int getNumRows() {
                return elements.size();
            }
            int getNumColumns() {
                return elements.at(0).size();
            }
            const std::vector<T>& getRow(int row) {
                return elements.at(row);
            }
            std::vector<T> getColumn(int col) {
                std::vector<T> temp;
                temp.reserve(elements.size());
                for (typename std::vector<std::vector<T> >::iterator it = elements.begin(); it != elements.end(); ++it) {
                    temp.push_back((*it).at(col));
                }
                return temp;
            }
            void display(const std::vector<T>& toDisplay = std::vector<T>()) {
                if (toDisplay.empty()) {
                    for (typename std::vector<std::vector<T> >::const_iterator it = elements.begin(); it != elements.end(); ++it) {
                        for (typename std::vector<T>::const_iterator itInner = (*it).begin(); itInner != (*it).end(); ++itInner) {
                            std::cout << *itInner << " ";
                        }
                        std::cout << std::endl;
                    }
                } else {
                    for (typename std::vector<T>::const_iterator itVec = toDisplay.begin(); itVec != toDisplay.end(); ++itVec) {
                        std::cout << *itVec << " ";
                    }
                    std::cout << std::endl;
                }
            }
        private:
            std::vector<std::vector<T> > elements;
    };
}

#endif
