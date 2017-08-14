#ifndef MATRIX_CUDA_FUNCTIONS
#define MATRIX_CUDA_FUNCTIONS

namespace StealthMath {
    template <typename T>
    __global__ void copyCUDA(T* A, const T* B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            A[index] = B[index];
        }
    }

    template <typename T>
    __global__ void setCUDA(T* A, T setValue, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            A[index] = setValue;
        }
    }

    template <typename T>
    __global__ void transposeCUDA(const T* original, int numRows, int numCols, T* transposed) {
        int x = blockIdx.y * BLOCK_DIM + threadIdx.x;
        int y = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if (x < numCols && y < numRows) {
            transposed[x * numRows + y] = original[y * numCols + x];
        }
    }

    template <typename T>
    __global__ void weightedRowSumCUDA(const T* A, float scaleFactor, int numCols, int size, T* C) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        T sum = T();
        if (col < numCols) {
            for (int i = col; i < size; i += numCols) {
                sum += A[i];
            }
            C[col] = sum * scaleFactor;
        }
    }

    template <typename T>
    __global__ void argmaxRowCUDA(const T* A, int numCols, int numRows, T* C) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            T max = T();
            T maxIndex = T();
            for (int i = 0; i < numCols; ++i) {
                int index = row * numCols + i;
                if (A[index] > max) {
                    max = A[index];
                    maxIndex = i;
                }
            }
            C[row] = maxIndex;
        }
    }

    template <typename T>
    __global__ void rowWiseDotProductCUDA(const T* A, const T* B, int numRows, int numCols, T* C) {
        // Each thread sCUDA one row.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            T Cvalue = T();
            // numCols is guaranteed to be a multiple of 32.
            for (int i = 0; i < numCols; ++i) {
                Cvalue += A[i] * B[i];
            }
            C[row] = Cvalue;
        }
    }

    template <typename T>
    __global__ void productCUDA(const T* A, const T* B, int numRowsA, int numColsA, int numColsB, int Asize, int Bsize, T* C) {
       __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
       __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
       // Compute the coordinates of matrix C that this thread is responsible for.
       int row = blockIdx.x * blockDim.x + threadIdx.x;
       int col = blockIdx.y * blockDim.y + threadIdx.y;
       bool cValid = row < numRowsA && col < numColsB;
       T Cvalue = T();
       // Iterate over the sub-matrices of A and B.
       int maxIterations = numColsA + BLOCK_DIM - 1;
       for (int i = 0; i < maxIterations; i += BLOCK_DIM) {
           // Compute indices.
           int indexA = row * numColsA + (i + threadIdx.y);
           int indexB = (i + threadIdx.x) * numColsB + col;
           // Load sub-matrix A.
           tileA[threadIdx.x][threadIdx.y] = (indexA < Asize) ? A[indexA] : 0;
           // Load sub-matrix B.
           tileB[threadIdx.x][threadIdx.y] = (indexB < Bsize) ? B[indexB] : 0;
           // Synchronize.
           __syncthreads();
           // Compute dot product only if the point is within the C matrix.
           if (cValid) {
               #pragma unroll
               for (int j = 0; j < BLOCK_DIM; ++j) {
                   Cvalue += tileA[threadIdx.x][j] * tileB[j][threadIdx.y];
               }
           }
           // Synchronize.
           __syncthreads();
       }
       // Write to output.
       if (cValid) {
           C[row * numColsB + col] = Cvalue;
       }
    }

    template <typename T>
    __global__ void sumCUDA(const T* A, T* B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            B[index] += A[index];
        }
    }

    template <typename T>
    __global__ void differenceCUDA(const T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            B[index] = A[index] - B[index];
        }
    }

    template <typename T>
    __global__ void differenceInPlaceCUDA(const T* B, T* A, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] -= B[index];
        }
    }

    template <typename T>
    __global__ void matrixVectorRowSumCUDA(const T* A, const T* B, int numCols, int numRowsA, T* C) {
        __shared__ T tileB[BLOCK_DIM];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        // Load vector - only load each element once.
        if (threadIdx.x == 0) {
            tileB[threadIdx.y] = (col < numCols) ? B[col] : 0;
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        if (row < numRowsA && col < numCols) {
            int index = row * numCols + col;
            C[index] = A[index] + tileB[threadIdx.y];
        }
    }

    template <typename T>
    __global__ void matrixVectorColumnSumCUDA(const T* A, const T* B, int numRows, int numColsA, T* C) {
        __shared__ T tileB[BLOCK_DIM];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        // Load vector - only load each element once.
        if (threadIdx.y == 0) {
            tileB[threadIdx.x] = (row < numRows) ? B[row] : 0;
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        if (row < numRows && col < numColsA) {
            int index = row * numColsA + col ;
            C[index] = A[index] + tileB[threadIdx.x];
        }
    }

    template <typename T>
    __global__ void scalarProductCUDA(const T* A, T B, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = A[index] * B;
        }
    }

    template <typename T>
    __global__ void scalarQuotientCUDA(const T* A, T B, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = A[index] / B;
        }
    }

    template <typename T>
    __global__ void scalarSumCUDA(const T* A, T B, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = A[index] + B;
        }
    }

    template <typename T>
    __global__ void scalarDifferenceCUDA(const T* A, T B, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = A[index] - B;
        }
    }

    template <typename T>
    __global__ void hadamardProductCUDA(const T* A, T* B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            B[index] *= A[index];
        }
    }

    template <typename T>
    __global__ void powerCUDA(const T* A, int exponent, int Asize, T* C) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            C[index] = powf(A[index], exponent);
        }
    }
}

#endif
