#include <iostream>
#include <cuda_runtime.h>

#define N 1024  // Size of the input signal
#define K 10    // Size of the kernel
#define BLOCK_SIZE 256  // Define block size

__global__ void cross_correlation_parallel(int* input, int* kernel, int* output, int N, int K) {
    // Allocate shared memory for the block
    __shared__ int temp[BLOCK_SIZE + K - 1];  // No padding, just enough space for input + kernel size

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load input data into shared memory
    if (gindex < N) {
        temp[threadIdx.x] = input[gindex];
    } else {
        temp[threadIdx.x] = 0;  // Pad with zero if out-of-bounds
    }

    __syncthreads();  // Synchronize threads to make sure all data is loaded

    // Compute the cross-correlation for the current index
    if (gindex < N - K + 1) {  // Only compute for valid output indices
        int result = 0;
        for (int i = 0; i < K; i++) {
            result += temp[threadIdx.x + i] * kernel[i];
        }
        output[gindex] = result;
    }
}

int main() {
    int *h_input, *h_output, *h_kernel;
    int *d_input, *d_output, *d_kernel;

    // Allocate host memory
    h_input = (int*)malloc(N * sizeof(int));
    h_output = (int*)malloc((N - K + 1) * sizeof(int));
    h_kernel = (int*)malloc(K * sizeof(int));

    // Initialize input signal and kernel (example with values 1)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1;
    }
    for (int i = 0; i < K; ++i) {
        h_kernel[i] = 1;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_input, N * sizeof(int));
    cudaMalloc((void**)&d_output, (N - K + 1) * sizeof(int));
    cudaMalloc((void**)&d_kernel, K * sizeof(int));

    // Copy input signal and kernel to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with N and K as arguments
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_correlation_parallel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_kernel, d_output, N, K);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, (N - K + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Output: ";
    for (int i = 0; i < (N - K + 1); ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}
