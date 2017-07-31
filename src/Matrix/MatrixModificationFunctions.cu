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

    template <typename T>
    void Matrix<T>::set(T setValue) {
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        setCUDA<<<blocks, threads>>>(data(), setValue, size());
        cudaDeviceSynchronize();
    }

    template <typename T>
    Matrix<T> Matrix<T>::randomNormal(int rows, int cols, double mean, double stdDev) {
        Matrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::randomNormalLike(const Matrix& like, double mean, double stdDev) {
        Matrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::randomUniform(int rows, int cols, double lowerBound, double upperBound) {
        Matrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::randomUniformLike(const Matrix& like, double lowerBound, double upperBound) {
        Matrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::ones(int rows, int cols) {
        Matrix<T> output(rows, cols);
        output.set(1);
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::onesLike(const Matrix& like) {
        Matrix<T> output(like.numRows(), like.numColumns());
        output.set(1);
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::zeros(int rows, int cols) {
        Matrix<T> output(rows, cols);
        output.set(0);
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::zerosLike(const Matrix& like) {
        Matrix<T> output(like.numRows(), like.numColumns());
        output.set(0);
        return output;
    }

    template <typename T>
    Matrix<T> Matrix<T>::sequentialMatrix(int rows, int cols) {
        Matrix<T> output(rows, cols);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = i;
        }
        return output;
    }
}

#endif
