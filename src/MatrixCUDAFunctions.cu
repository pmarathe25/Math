#ifndef MATRIX_CUDA_FUNCTIONS
#define MATRIX_CUDA_FUNCTIONS

namespace math {
    template<typename T>
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        int x = blockIdx.y * BLOCK_DIM + threadIdx.x;
        int y = blockIdx.x * BLOCK_DIM + threadIdx.y;
        if (x < numCols && y < numRows) {
            transposed[x * numRows + y] = original[y * numCols + x];
        }
    }

    template <typename T>
    __global__ void computeProduct(T* A, T* B, int numRowsA, int numColsA, int numRowsB, int numColsB, int Asize, int Bsize, T* C) {
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
    __global__ void computeScalarProduct(T* A, T B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            A[index] = A[index] * B;
        }
    }

    template <typename T>
    __global__ void computeDotProduct(T* A, T* B, int numRows, int numCols) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            int rowIndex = row * numCols;
            int index = 0;
            T Cvalue = T();
            // numCols is guaranteed to be a multiple of 32.
            for (int i = 0; i < numCols; i += BLOCK_DIM) {
                index = rowIndex + i;
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    Cvalue += A[index] * B[index];
                    ++index;
                }
            }
            A[row] = Cvalue;
        }
    }

    template <typename T>
    __global__ void computeSum(T* A, T* B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            A[index] = A[index] + B[index];
        }
    }

    template <typename T>
    __global__ void computeMatrixVectorRowSum(T* A, T* B, int numCols, int numRowsA) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
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
            // printf("%d\n", index);
            A[index] = A[index] + tileB[threadIdx.y];
        }
    }

    template <typename T>
    __global__ void computeMatrixVectorColumnSum(T* A, T* B, int numRows, int numColsA) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
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
            A[index] = A[index] + tileB[threadIdx.x];
        }
    }

    template <typename T>
    __global__ void computeMatrixVectorRowDifference(T* A, T* B, int numCols, int numRowsA) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
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
            // printf("%d\n", index);
            A[index] = A[index] - tileB[threadIdx.y];
        }
    }

    template <typename T>
    __global__ void computeMatrixVectorColumnDifference(T* A, T* B, int numRows, int numColsA) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
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
            A[index] = A[index] - tileB[threadIdx.x];
        }
    }

    template <typename T>
    __global__ void computeDifference(T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] - B[index];
        }
    }

    template <typename T>
    __global__ void computeHadamardProduct(T* A, T* B, int Asize) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < Asize) {
            A[index] = A[index] * B[index];
        }
    }


}

#endif
