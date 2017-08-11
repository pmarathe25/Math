#ifndef MATRIX_ACCESS_FUNCTIONS
#define MATRIX_ACCESS_FUNCTIONS

namespace math {
    // Indexing Functions.
    template <typename T>
    T& Matrix<T>::at(int row, int col) {
        if (row < numRows() && col < numColumns()) {
            return elements[row * numColumns() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    const T& Matrix<T>::at(int row, int col) const {
        if (row < numRows() && col < numColumns()) {
            return elements[row * numColumns() + col];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    T& Matrix<T>::at(int index) {
        if (index < size()) {
            return elements[index];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    template <typename T>
    const T& Matrix<T>::at(int index) const {
        if (index < size()) {
            return elements[index];
        } else {
            throw std::out_of_range("Index out of range.");
        }
    }

    // Unsafe indexing functions.
    template <typename T>
    T& Matrix<T>::operator[](int index) {
        return elements[index];
    }

    template <typename T>
    const T& Matrix<T>::operator[](int index) const {
        return elements[index];
    }

    template <typename T>
    T* Matrix<T>::data() {
        return elements;
    }

    template <typename T>
    const T* Matrix<T>::data() const {
        return elements;
    }

    template <typename T>
    int Matrix<T>::ceilDivide(int x, int y) {
        return 1 + ((x - 1) / y);
    }
}

#endif
