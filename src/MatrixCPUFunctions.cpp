#ifndef MATRIX_CPU_FUNCTIONS
#define MATRIX_CPU_FUNCTIONS

namespace math {
    template <typename T>
    Matrix<T> Matrix<T>::CPURowMean(double scaleFactor) const {
        Matrix output = Matrix(1, numColumns());
        for (int row = 0; row < size(); row += numColumns()) {
            for (int col = 0; col < numColumns(); ++col) {
                output[col] += (*this)[row + col] * scaleFactor;
            }
        }
        return output;
    }

    template <typename T>
    void Matrix<T>::randomizeNormal(T mean, T stdDev) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < size(); ++i) {
            elements[i] = normalDistribution(generator);
        }
    }

    template <typename T>
    void Matrix<T>::randomizeUniform(T lowerBound, T upperBound) {
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < size(); ++i) {
            elements[i] = uniformDistribution(generator);
        }
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] + other[i];
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUSum(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] + other;
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
    Matrix<T> Matrix<T>::CPUDifference(T other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] - other;
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
            int i = 0;
            for (int row = 0; row < size(); row += numColumns()) {
                for (int col = 0; col < numColumns(); ++col) {
                    output[row + col] = (*this)[row + col] - other[i];
                }
            }
            ++i;
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
        for (int i = 0; i < size(); ++i) {
            output[i] += (*this)[i] * other[i];
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUHadamardProduct(const Matrix<T>& other) const {
        Matrix output = Matrix(numRows(), numColumns());
        for (int i = 0; i < size(); ++i) {
            output[i] = (*this)[i] * other[i];
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::CPUKroneckerProduct(const Matrix<T>& other) const {
        Matrix output = Matrix(size(), other.size());
        for (int i = 0; i < size(); ++i) {
            for (int j = 0; j < other.size(); ++j) {
                output[i * other.size() + j] = (*this)[i] * other[j];
            }
        }
        return output;
    }

}

#endif
