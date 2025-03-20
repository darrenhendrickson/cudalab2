#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256         // Define the block size
#define KERNEL_SIZE 10         // Size of the filter kernel
#define OUTPUT_SIZE (N - KERNEL_SIZE + 1)

__global__ void cross_correlation_1d(int *in, int *out, int *kernel, int N) {
    // Allocate shared memory: block portion plus extra elements for the kernel window.
    __shared__ int temp[BLOCK_SIZE + KERNEL_SIZE - 1];

    // Calculate the output (global) index for this thread.
    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the corresponding global input index.
    int input_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load the primary input element into shared memory.
    if (input_index < N) {
        temp[threadIdx.x] = in[input_index];
    } else {
        temp[threadIdx.x] = 0;  // Out-of-bounds elements set to zero.
    }

    // Load the extra (KERNEL_SIZE - 1) elements required for the convolution window.
    int extra_index = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
    if (threadIdx.x < (KERNEL_SIZE - 1)) {
        if (extra_index < N)
            temp[blockDim.x + threadIdx.x] = in[extra_index];
        else
            temp[blockDim.x + threadIdx.x] = 0;
    }

    __syncthreads();

    // Compute cross-correlation only for valid output indices.
    if (out_index < OUTPUT_SIZE) {
        int result = 0;
        // Each output element is computed as:
        // out[out_index] = temp[threadIdx.x]*kernel[0] + temp[threadIdx.x+1]*kernel[1] + ... + temp[threadIdx.x+KERNEL_SIZE-1]*kernel[KERNEL_SIZE-1]
        for (int k = 0; k < KERNEL_SIZE; k++) {
            result += temp[threadIdx.x + k] * kernel[k];
        }
        out[out_index] = result;
    }
}
#define N 1024  

int main() {
    int *h_in, *h_out, *h_kernel;
    int *d_in, *d_out, *d_kernel;

    // Allocate host memory using C++ new
    h_in = new int[N];
    h_out = new int[OUTPUT_SIZE];
    h_kernel = new int[KERNEL_SIZE];

    // Initialize input array with the fixed pattern: 0,1,2,...,9,0,1,2,...
    for (int i = 0; i < N; i++) {
        h_in[i] = i % 10;
    }

    // Initialize kernel with fixed values: 1, 2, 3, ..., 10
    for (int i = 0; i < KERNEL_SIZE; i++) {
        h_kernel[i] = i + 1;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_in, N * sizeof(int));
    cudaMalloc((void **)&d_out, OUTPUT_SIZE * sizeof(int));
    cudaMalloc((void **)&d_kernel, KERNEL_SIZE * sizeof(int));

    // Copy input and kernel data from host to device
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel:
    // Each thread computes one element of the output.
    int numBlocks = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_correlation_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_kernel, N);

    // Copy the computed output from device back to host
    cudaMemcpy(h_out, d_out, OUTPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output (printing 20 values per line for readability)
    std::cout << "CUDA Output (" << OUTPUT_SIZE << " elements):" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << h_out[i] << " ";
        if ((i + 1) % 20 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);

    // Free host memory
    delete[] h_in;
    delete[] h_out;
    delete[] h_kernel;

    return 0;
}
