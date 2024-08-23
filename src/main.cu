#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR() do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "File: %s, Line: %d, CUDA error: %s\n", __FILE__, __LINE__ , cudaGetErrorString(error)); \
        exit(-1); \
    } \
} while(0)

__global__ void add_kernel(float *a, float *b, float *c, uint n) {
    uint idx = threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Select the GPU you want to use (0 for the first GPU, 1 for the second, etc.)
    // int deviceId = 0; // Change this to 1 if you want to use the second GPU
    // cudaError_t error = cudaSetDevice(deviceId);
    // if (error != cudaSuccess) {
    //     fprintf(stderr, "cudaSetDevice failed! Error: %s\n", cudaGetErrorString(error));
    //     return 1;
    // }

    // Allocate host memory and initialize data
    float *a_h = (float *)malloc(10 * sizeof(float));
    a_h[0] = 1.0;
    a_h[1] = 2.0;
    a_h[2] = 3.0;
    a_h[3] = 4.0;
    a_h[4] = 5.0;
    a_h[5] = 6.0;
    a_h[6] = 7.0;
    a_h[7] = 8.0;
    a_h[8] = 9.0;
    a_h[9] = 10.0;

    float *b_h = (float *)malloc(10 * sizeof(float));
    b_h[0] = 10.0;
    b_h[1] = 11.0;
    b_h[2] = 12.0;
    b_h[3] = 13.0;
    b_h[4] = 14.0;
    b_h[5] = 15.0;
    b_h[6] = 16.0;
    b_h[7] = 17.0;
    b_h[8] = 18.0;
    b_h[9] = 19.0;

    // Allocate device memory and copy data from host to device
    float *a_d, *b_d, *c_d;

    cudaMalloc((void**)&a_d, 10 * sizeof(float)); CHECK_CUDA_ERROR();
    cudaMalloc((void**)&b_d, 10 * sizeof(float)); CHECK_CUDA_ERROR();
    cudaMalloc((void**)&c_d, 10 * sizeof(float)); CHECK_CUDA_ERROR();

    cudaMemcpy(a_d, a_h, 10 * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();
    cudaMemcpy(b_d, b_h, 10 * sizeof(float), cudaMemcpyHostToDevice); CHECK_CUDA_ERROR();

    add_kernel<<<1, 10>>>(a_d, b_d, c_d, 10);
    
    CHECK_CUDA_ERROR();

    float *c_h = (float *)malloc(10 * sizeof(float));
    cudaMemcpy(c_h, c_d, 10 * sizeof(float), cudaMemcpyDeviceToHost); CHECK_CUDA_ERROR();

    for (int i = 0; i < 10; i++) {
        printf("%f\n", c_h[i]);
    }

    // Cleanup device memory
    cudaFree(a_d); CHECK_CUDA_ERROR();
    cudaFree(b_d); CHECK_CUDA_ERROR();
    cudaFree(c_d); CHECK_CUDA_ERROR();

    // Cleanup host memory
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}