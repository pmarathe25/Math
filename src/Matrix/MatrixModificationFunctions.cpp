#ifndef MATRIX_MODIFICATION_FUNCTIONS
#define MATRIX_MODIFICATION_FUNCTIONS

namespace math {
    template <typename T>
    void Matrix<T>::reshape(int rows, int cols) {
        if (rows * cols == size()) {
            this -> rows = rows;
            this -> cols = cols;
            this -> isVec = (rows == 1) || (cols == 1);
        } else {
            throw std::invalid_argument("Size mismatch in reshape.");
        }
    }
}

#endif
