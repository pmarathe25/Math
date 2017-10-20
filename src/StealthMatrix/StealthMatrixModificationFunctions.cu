#ifndef MATRIX_MODIFICATION_FUNCTIONS
#define MATRIX_MODIFICATION_FUNCTIONS

namespace StealthMath {
    template <typename T>
    StealthMatrix<T>& StealthMatrix<T>::reshape(int rows, int cols) {
        cols = (cols == -1) ? size() / rows : cols;
        if (rows * cols == size()) {
            this -> rows = rows;
            this -> cols = cols;
            this -> isVec = (rows == 1) || (cols == 1);
            return (*this);
        } else {
            throw std::invalid_argument("Size mismatch in reshape.");
        }
    }

    template <typename T>
    StealthMatrix<T>& StealthMatrix<T>::set(T setValue) {
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        setCUDA<<<blocks, threads>>>(data(), setValue, size());
        cudaDeviceSynchronize();
        return (*this);
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::randomNormal(int rows, int cols, double mean, double stdDev) {
        StealthMatrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::randomNormalLike(const StealthMatrix& like, double mean, double stdDev) {
        StealthMatrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::normal_distribution<double> normalDistribution(mean, stdDev);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = normalDistribution(generator);
        }
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::randomUniform(int rows, int cols, double lowerBound, double upperBound) {
        StealthMatrix<T> output(rows, cols);
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::randomUniformLike(const StealthMatrix& like, double lowerBound, double upperBound) {
        StealthMatrix<T> output(like.numRows(), like.numColumns());
        auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
        std::default_random_engine generator(value.count());
        std::uniform_real_distribution<double> uniformDistribution(lowerBound, upperBound);
        for (int i = 0; i < output.size(); ++i) {
            output[i] = uniformDistribution(generator);
        }
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::ones(int rows, int cols) {
        StealthMatrix<T> output(rows, cols);
        output.set(1);
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::onesLike(const StealthMatrix& like) {
        StealthMatrix<T> output(like.numRows(), like.numColumns());
        output.set(1);
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::zeros(int rows, int cols) {
        StealthMatrix<T> output(rows, cols);
        output.set(0);
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::zerosLike(const StealthMatrix& like) {
        StealthMatrix<T> output(like.numRows(), like.numColumns());
        output.set(0);
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::sequential(int rows, int cols, int start) {
        StealthMatrix<T> output(rows, cols);
        for (int i = start; i < output.size() + start; ++i) {
            output[i] = i;
        }
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::sequentialLike(const StealthMatrix& like, int start) {
        StealthMatrix<T> output(like.numRows(), like.numColumns());
        for (int i = start; i < output.size() + start; ++i) {
            output[i] = i;
        }
        return output;
    }

}

#endif
