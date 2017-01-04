#ifndef MATRIX_CPU_FUNCTIONS
#define MATRIX_CPU_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] + other[i];
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUMatrixVectorSum(const Matrix<T>& other) const {
        Matrix<T> output = Matrix<T>(numRows(), numColumns());
        if (other.numRows() == 1) {
            // std::cout << "HERE" << std::endl;
            for (int row = 0; row < size(); row += numColumns()) {
                for (int col = 0; col < numColumns(); ++col) {
                    output[row + col] = (*this)[row + col] + other[col];
                }
            }
        } else {
            int i = 0;
            for (int row = 0; row < size(); row += numColumns()) {
                for (int col = 0; col < numColumns(); ++col) {
                    output[row + col] = (*this)[row + col] + other[i];
                }
                ++i;
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDifference(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] - other[i];
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUMatrixVectorDifference(const Matrix<T>& other) const {
        Matrix<T> output = Matrix<T>(numRows(), numColumns());
        if (other.numRows() == 1) {
            for (int row = 0; row < size(); row += numColumns()) {
                for (int col = 0; col < numColumns(); ++col) {
                    output[row + col] = (*this)[row + col] - other[col];
                }
            }
        } else {
            for (int row = 0; row < size(); row += numColumns()) {
                for (int col = 0; col < numColumns(); ++col) {
                    output[row + col] = (*this)[row + col] - other[row];
                }
            }
        }
        return output;
    }


    template <typename T>
    Matrix<T> Matrix<T>::CPUScalarProduct(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] * other;
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDotProduct(const Matrix<T>& other) const {
        Matrix<T> output = Matrix<T>(numRows(), 1);
        for (int row = 0; row < numRows(); ++row) {
            for (int col = 0; col < numColumns(); ++col) {
                output[row] += (*this)[col] * other[col];
            }
        }
        return output;
    }

}

#endif
