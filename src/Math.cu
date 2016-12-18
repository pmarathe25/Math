#include "Math/Math.hpp"
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <cuda.h>
#define N 512

namespace math {
    int fibonacci(int n) {
        if (n < 2) {
            return n;
        } else {
            // Only need to keep track of the previous 2 numbers in the sequence.
            int prev[2] = {0, 1};
            int result = 0;
            for (int i = 1; i < n; i++) {
                result = prev[0] + prev[1];
                prev[0] = prev[1];
                prev[1] = result;
            }
            return result;
        }
    }

    double fibonacci(double n) {
        return fibonacci((int) n);
    }

    double factorial(double operand) {
        int temp;
        // Set to 1 if the operand is less.
        temp = (operand < 1) ? 1 : operand;
        for (int i = 1; i < (int) operand; i++) {
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
    __global__ void computeInnerProduct(T* a, T* b, int* size, T* result) {
        // Memory shared across the thread block.
        __shared__ double temp[N];
        temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
        // Synchronize
        __syncthreads();
        //  Final computation
        *result = T();
        if (threadIdx.x == 0) {
            for (int i = 0; i < *size; ++i) {
                *result += temp[i];
            }
        }
    }
    // Specialization declarations.
    template __global__ void computeInnerProduct<double>(double* a, double* b, int* size, double* result);
    template __global__ void computeInnerProduct<float>(float* a, float* b, int* size, float* result);
    template __global__ void computeInnerProduct<int>(int* a, int* b, int* size, int* result);

    template<typename T>
    T innerProduct(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument( "Vectors are not of the same lengths." );
        }
        int vecSize = a.size();
        T product = T();
        //  Initialize device copies.
        T *dev_a, *dev_b, *dev_result;
        int* dev_size;
        //  Allocate memory for device copies.
        cudaMalloc((void**)&dev_a, vecSize * sizeof(T));
        cudaMalloc((void**)&dev_b, vecSize * sizeof(T));
        cudaMalloc((void**)&dev_size, sizeof(int));
        cudaMalloc((void**)&dev_result, sizeof(T));
        // Copy inputs to device.
        cudaMemcpy(dev_a, a.data(), vecSize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b.data(), vecSize * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_size, &vecSize, sizeof(int), cudaMemcpyHostToDevice);
        // Launch kernel.
        computeInnerProduct<T><<<1, vecSize>>>(dev_a, dev_b, dev_size, dev_result);
        // Get result.
        cudaMemcpy(&product, dev_result, sizeof(T) , cudaMemcpyDeviceToHost);
        // Free memory.
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_size);
        cudaFree(dev_result);
        // Return.
        return product;
    }
    // Specialization declarations.
    template double innerProduct(const std::vector<double>& a, const std::vector<double>& b);
    template float innerProduct(const std::vector<float>& a, const std::vector<float>& b);
    template int innerProduct(const std::vector<int>& a, const std::vector<int>& b);
}
