#ifndef MATRIX_CPU_FUNCTIONS
#define MATRIX_CPU_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        const T* otherData = other.data();
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                outputData[row + col] = thisData[row + col] + otherData[row + col];
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDifference(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        const T* otherData = other.data();
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                outputData[row + col] = thisData[row + col] - otherData[row + col];
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUScalarProduct(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        T* outputData = output.data();
        const T* thisData = data();
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                outputData[row + col] = thisData[row + col] * other;
            }
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUDotProduct(const Matrix<T>& other) const {
        Matrix<T> output = Matrix<T>(numRows(), 1);
        int i = 0;
        for (int row = 0; row < numRows() * numColumnsRaw(); row += numColumnsRaw()) {
            for (int col = 0; col < numColumns(); ++col) {
                output[i] += data()[col] * other.data()[col];
            }
            ++i;
        }
        return output;
    }
}

#endif
