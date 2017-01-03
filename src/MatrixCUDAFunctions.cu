#ifndef MATRIX_CUDA_FUNCTIONS
#define MATRIX_CUDA_FUNCTIONS

namespace math {
    template <typename T>
    __global__ void computeTranspose(T* original, int numRows, int numCols, T* transposed) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tile[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute row and column of this block.
        int row = blockIdx.x * blockDim.x;
        int col = blockIdx.y * blockDim.y;
        // Load a (transposed) tile into shared memory.
        tile[threadIdx.y][threadIdx.x] = original[(row + threadIdx.x) * numCols + (col + threadIdx.y)];
        // Synchronize.
        __syncthreads();
        // Write the tiles into the output. Switch rows and columns to handle non-square matrices.
        transposed[(col + threadIdx.x) * numRows + (row + threadIdx.y)] = tile[threadIdx.x][threadIdx.y];
    }

    template <typename T>
    __global__ void computeProduct(T* A, T* B, int numColsA, int numColsB, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A.
            tileA[threadIdx.x][threadIdx.y] = A[row * numColsA + (i + threadIdx.y)];
            // Load sub-matrix B.
            tileB[threadIdx.x][threadIdx.y] = B[(i + threadIdx.x) * numColsB + col];
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[threadIdx.x][j] * tileB[j][threadIdx.y];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[row * numColsB + col] = Cvalue;
    }

    template <typename T>
    __global__ void computeVectorProductRight(T* A, T* B, int numColsA, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM][BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A.
            tileA[threadIdx.x][threadIdx.y] = A[row * numColsA + (i + threadIdx.y)];
            // Load sub-matrix B - only load each element once.
            if (threadIdx.y == 0) {
                tileB[threadIdx.x] = B[i + threadIdx.x];
            }
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[threadIdx.x][j] * tileB[j];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[row] = Cvalue;
    }

    template <typename T>
    __global__ void computeVectorProductLeft(T* A, T* B, int numColsA, int numColsB, T* C) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileA[BLOCK_DIM + 1];
        __shared__ T tileB[BLOCK_DIM][BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        T Cvalue = T();
        // Iterate over the sub-matrices of A and B.
        for (int i = 0; i < numColsA; i += BLOCK_DIM) {
            // Load sub-matrix A - only load each element once.
            if (threadIdx.x == 0) {
                tileA[threadIdx.y] = A[i + threadIdx.y];
            }
            // Load sub-matrix B.
            tileB[threadIdx.x][threadIdx.y] = B[(i + threadIdx.x) * numColsB + col];
            // Synchronize.
            __syncthreads();
            // Compute dot product only if the point is within the C matrix.
            #pragma unroll
            for (int j = 0; j < BLOCK_DIM; ++j) {
                Cvalue += tileA[j] * tileB[j][threadIdx.y];
            }
            // Synchronize.
            __syncthreads();
        }
        // Write to output.
        C[col] = Cvalue;
    }

    template <typename T>
    __global__ void computeScalarProduct(T* A, T B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] * B;
    }

    template <typename T>
    __global__ void computeVectorScalarProduct(T* A, T B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] * B;
        }
    }

    template <typename T>
    __global__ void computeDotProduct(T* A, T* B, int numRows, int numColsRaw) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < numRows) {
            int rowIndex = row * numColsRaw;
            int index = 0;
            T Cvalue = T();
            // numColsRaw is guaranteed to be a multiple of 32.
            for (int i = 0; i < numColsRaw; i += BLOCK_DIM) {
                index = rowIndex + i;
                #pragma unroll
                for (int j = 0; j < BLOCK_DIM; ++j) {
                    Cvalue += A[index] * B[index];
                    index++;
                }
            }
            A[row] = Cvalue;
        }
    }

    template <typename T>
    __global__ void computeSum(T* A, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] + B[index];
    }

    template <typename T>
    __global__ void computeVectorSum(T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] + B[index];
        }
    }

    template <typename T>
    __global__ void computeMatrixVectorRowSum(T* A, T* B, int numColsB) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        // Load vector - only load each element once.
        if (threadIdx.x == 0) {
            tileB[threadIdx.y] = B[col];
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * numColsB + col;
        A[index] = A[index] + tileB[threadIdx.y];
    }

    template <typename T>
    __global__ void computeMatrixVectorColumnSum(T* A, T* B, int numColsB) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        // Load vector - only load each element once.
        if (threadIdx.y == 0) {
            tileB[threadIdx.x] = B[row];
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        int index = row * numColsB + blockIdx.y * blockDim.y + threadIdx.y;
        A[index] = A[index] + tileB[threadIdx.x];
    }

    template <typename T>
    __global__ void computeMatrixVectorRowDifference(T* A, T* B, int numColsB) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        // Load vector - only load each element once.
        if (threadIdx.x == 0) {
            tileB[threadIdx.y] = B[col];
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * numColsB + col;
        A[index] = A[index] - tileB[threadIdx.y];
    }

    template <typename T>
    __global__ void computeMatrixVectorColumnDifference(T* A, T* B, int numColsB) {
        // Avoid bank conflicts by allocating a single dummy element.
        __shared__ T tileB[BLOCK_DIM + 1];
        // Compute the coordinates of matrix C that this thread is responsible for.
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        // Load vector - only load each element once.
        if (threadIdx.y == 0) {
            tileB[threadIdx.x] = B[row];
        }
        // Synchronize.
        __syncthreads();
        // Write to output.
        int index = row * numColsB + blockIdx.y * blockDim.y + threadIdx.y;
        A[index] = A[index] - tileB[threadIdx.x];
    }

    template <typename T>
    __global__ void computeDifference(T* A, T* B) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        A[index] = A[index] - B[index];
    }

    template <typename T>
    __global__ void computeVectorDifference(T* A, T* B, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            A[index] = A[index] - B[index];
        }
    }
}

#endif
