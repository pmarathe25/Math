#ifndef MATRIX_COMPUTATION_FUNCTIONS
#define MATRIX_COMPUTATION_FUNCTIONS

namespace StealthMath {
    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::transpose() const {
        // For vectors, we only need to flip the dimensions.
        if (isVector()) {
            StealthMatrix<T> output(*this);
            output.reshape(numColumns(), numRows());
            return output;
        } else {
            StealthMatrix<T> output(numColumns(), numRows());
            dim3 blocks(ceilDivide(numRows(), BLOCK_DIM), ceilDivide(numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            transposeCUDA<<<blocks, threads>>>(data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::weightedSum(int axis, float scaleFactor) const {
        switch (axis) {
            case 0: {
                if (numColumns() == 1) {
                    return (*this);
                } else {
                    StealthMatrix<T> output(numRows(), 1);
                    dim3 blocks(ceilDivide(output.numRows(), THREADS_PER_BLOCK));
                    dim3 threads(THREADS_PER_BLOCK);
                    weightedColSumCUDA<<<blocks, threads>>>(data(), scaleFactor, numRows(), numColumns(), output.data());
                    cudaDeviceSynchronize();
                    return output;
                }
            }
            case 1: {
                if (numRows() == 1) {
                    return (*this);
                } else {
                    StealthMatrix<T> output(1, numColumns());
                    dim3 blocks(ceilDivide(output.numColumns(), THREADS_PER_BLOCK));
                    dim3 threads(THREADS_PER_BLOCK);
                    weightedRowSumCUDA<<<blocks, threads>>>(data(), scaleFactor, numColumns(), size(), output.data());
                    cudaDeviceSynchronize();
                    return output;
                }
            } default: {
                return StealthMatrix<T>();
            }
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::rowMean() const {
        return weightedSum(1, 1 / (float) numRows());
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::columnMean() const {
        return weightedSum(0, 1 / (float) numRows());
    }

    template <typename T>
    StealthMatrix<int> StealthMatrix<T>::argmax(int axis) const {
        switch (axis) {
            case 1: {
                StealthMatrix<int> output(numRows(), 1);
                dim3 blocks(ceilDivide(output.numRows(), THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                argmaxRowCUDA<<<blocks, threads>>>(data(), numRows(), numColumns(), output.data());
                cudaDeviceSynchronize();
                return output;
            } default: {
                return StealthMatrix<int>();
            }

        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::maxMask(int axis) const {
        switch (axis) {
            case 1: {
                StealthMatrix<T> output(numRows(), numColumns());
                dim3 blocks(ceilDivide(output.numRows(), THREADS_PER_BLOCK));
                dim3 threads(THREADS_PER_BLOCK);
                maxMaskRowCUDA<<<blocks, threads>>>(data(), numRows(), numColumns(), output.data());
                cudaDeviceSynchronize();
                return output;
            } default: {
                return StealthMatrix<T>();
            }

        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::dot(const StealthMatrix& other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            StealthMatrix<T> output(numRows(), 1);
            // Launch kernel where numThreads = numRows.
            dim3 blocks(ceilDivide(numRows(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            rowWiseDotProductCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("Incompatible matrices cannot be dotted.");
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator*(const StealthMatrix& other) const {
        if (numColumns() == other.numRows()) {
            StealthMatrix<T> output(numRows(), other.numColumns());
            dim3 blocks(ceilDivide(output.numRows(), BLOCK_DIM), ceilDivide(output.numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            productCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), other.numColumns(), size(), other.size(), output.data());
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("Cannot multiply matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator+(StealthMatrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            sumCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Cannot add matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }

    template <typename T>
    void StealthMatrix<T>::operator+=(const StealthMatrix& other) {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            sumCUDA<<<blocks, threads>>>(other.data(), data(), size());
            cudaDeviceSynchronize();
        } else {
            throw std::invalid_argument("Cannot add matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }
    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator-(StealthMatrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            differenceCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Cannot subtract matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }

    template <typename T>
    void StealthMatrix<T>::operator-=(const StealthMatrix& other) {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            differenceInPlaceCUDA<<<blocks, threads>>>(other.data(), data(), size());
            cudaDeviceSynchronize();
        } else {
            throw std::invalid_argument("Cannot subtract matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }


    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::hadamard(StealthMatrix other) const {
        if (numRows() == other.numRows() && numColumns() == other.numColumns()) {
            dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
            dim3 threads(THREADS_PER_BLOCK);
            hadamardProductCUDA<<<blocks, threads>>>(data(), other.data(), size());
            cudaDeviceSynchronize();
            return other;
        } else {
            throw std::invalid_argument("Cannot find the hadamard product of matrices of dimensions " + std::to_string(numRows()) + "x"
                + std::to_string(numColumns()) + " and " + std::to_string(other.numRows()) + "x" + std::to_string(other.numColumns()));
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::addVector(const StealthMatrix& other) const {
        if (other.isVector() && (other.size() == numRows() || other.size() == numColumns())) {
            StealthMatrix<T> output(numRows(), numColumns());
            dim3 blocks(ceilDivide(output.numRows(), BLOCK_DIM), ceilDivide(output.numColumns(), BLOCK_DIM));
            dim3 threads(BLOCK_DIM, BLOCK_DIM);
            if (other.numRows() == 1) {
                matrixVectorRowSumCUDA<<<blocks, threads>>>(data(), other.data(), numColumns(), numRows(), output.data());
            } else {
                matrixVectorColumnSumCUDA<<<blocks, threads>>>(data(), other.data(), numRows(), numColumns(), output.data());
            }
            cudaDeviceSynchronize();
            return output;
        } else {
            throw std::invalid_argument("addVector only accepts Matrices that are Vectors.");
        }
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator*(T other) const {
        StealthMatrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarProductCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator/(T other) const {
        StealthMatrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarQuotientCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator+(T other) const {
        StealthMatrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarSumCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::operator-(T other) const {
        StealthMatrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        scalarDifferenceCUDA<<<blocks, threads>>>(data(), other, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }

    template <typename T>
    StealthMatrix<T> StealthMatrix<T>::pow(int exponent) {
        StealthMatrix<T> output(numRows(), numColumns());
        dim3 blocks(ceilDivide(output.size(), THREADS_PER_BLOCK));
        dim3 threads(THREADS_PER_BLOCK);
        powerCUDA<<<blocks, threads>>>(data(), exponent, size(), output.data());
        cudaDeviceSynchronize();
        return output;
    }
}

#endif
