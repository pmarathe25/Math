#include "Math/Math.hpp"
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>
#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 8192

namespace math {
    int fibonacci(int n) {
        if (n < 2) {
            return n;
        } else {
            // Only need to keep track of the previous 2 numbers in the sequence.
            int prev[2] = {0, 1};
            int result = 0;
            for (int i = 1; i < n; ++i) {
                result = prev[0] + prev[1];
                prev[0] = prev[1];
                prev[1] = result;
            }
            return result;
        }
    }

    double factorial(double operand) {
        int temp;
        // Set to 1 if the operand is less.
        temp = (operand < 1) ? 1 : operand;
        for (int i = 1; i < (int) operand; ++i) {
            temp *= i;
        }
        return temp;
    }

    double divide(double operand1, double operand2) {
        return (operand1 / operand2);
    }

    double multiply(double operand1, double operand2) {
        return (operand1 * operand2);
    }

    double add(double operand1, double operand2) {
        return (operand1 + operand2);
    }

    double subtract(double operand1, double operand2) {
        return (operand1 - operand2);
    }

    template<typename T>
    __global__ void computeInnerProduct(T* a, T* b, int size, T* result) {
        // Memory shared across the thread block.
        __shared__ T temp[THREADS_PER_BLOCK];
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < size) {
            temp[threadIdx.x] = a[index] * b[index];
        } else {
            temp[threadIdx.x] = T();
        }
        // Synchronize.
        __syncthreads();
        //  Final computation.
        T tempResult = T();
        if (threadIdx.x == 0) {
            for (int i = 0; i < THREADS_PER_BLOCK; ++i) {
                tempResult += temp[i];
            }
            // Add this block's result to the final.
            atomicAdd(result, tempResult);
        }
    }
    // Specialization declarations.
    template __global__ void computeInnerProduct<float>(float* a, float* b, int size, float* result);
    template __global__ void computeInnerProduct<int>(int* a, int* b, int size, int* result);

    template<typename T>
    T innerProduct(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument( "Vectors are not of the same lengths." );
        }
        T product = T();
        //  Initialize device copies.
        T *dev_a, *dev_b, *dev_result;
        //  Allocate memory for device copies.
        cudaMalloc((void**)&dev_a, a.size() * sizeof(T));
        cudaMalloc((void**)&dev_b, b.size() * sizeof(T));
        cudaMalloc((void**)&dev_result, sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
        // Initialize output to 0.
        cudaMemcpy(dev_result, &product, sizeof(T), cudaMemcpyHostToDevice);
        // Launch kernel.
        computeInnerProduct<T><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_a, dev_b, a.size(), dev_result);
        // Get result.
        cudaMemcpy(&product, dev_result, sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_result);
        // Return.
        return product;
    }
    // Specialization declarations.
    template float innerProduct(const std::vector<float>& a, const std::vector<float>& b);
    template int innerProduct(const std::vector<int>& a, const std::vector<int>& b);
}
