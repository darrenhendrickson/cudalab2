#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KERNEL_SIZE 10
#define OUTPUT_SIZE (N - KERNEL_SIZE + 1)

__global__ void cross_correlation_1d(int *in, int *out, int *kernel, int N) {
    __shared__ int temp[BLOCK_SIZE + KERNEL_SIZE - 1];

    int out_index = blockIdx.x * blockDim.x + threadIdx.x;
    int input_index = out_index;

    // Load main input data into shared memory
    if (input_index < N) {
        temp[threadIdx.x] = in[input_index];
    } else {
        temp[threadIdx.x] = 0;
    }

    // Load extra (KERNEL_SIZE - 1) values into shared memory
    int extra_index = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
    if (threadIdx.x < (KERNEL_SIZE - 1)) {
        if (extra_index < N) {
            temp[blockDim.x + threadIdx.x] = in[extra_index];
        } else {
            temp[blockDim.x + threadIdx.x] = 0;
        }
    }

    __syncthreads();

    // Perform cross-correlation
    if (out_index < OUTPUT_SIZE) {
        int result = 0;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            result += temp[threadIdx.x + k] * kernel[k];
        }
        out[out_index] = result;
    }
}
#define N 1024
int main() {
    int *h_in = new int[N];
    int *h_kernel = new int[KERNEL_SIZE];
    int *h_out = new int[OUTPUT_SIZE];

    int *d_in, *d_out, *d_kernel;

    // Initialize input and kernel
    for (int i = 0; i < N; i++) h_in[i] = i % 10;
    for (int i = 0; i < KERNEL_SIZE; i++) h_kernel[i] = i + 1;

    // Allocate GPU memory
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, OUTPUT_SIZE * sizeof(int));
    cudaMalloc(&d_kernel, KERNEL_SIZE * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Configure grid
    int numBlocks = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // --- CUDA Timing ---
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Optional: repeat kernel for more measurable time
    const int runs = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) {
        cross_correlation_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_kernel, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Average CUDA kernel time over " << runs << " runs: "
              << (milliseconds / runs) << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_out, d_out, OUTPUT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output (optional)
    std::cout << "CUDA Output (" << OUTPUT_SIZE << " elements):" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << h_out[i] << " ";
        if ((i + 1) % 20 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    delete[] h_in;
    delete[] h_kernel;
    delete[] h_out;

    return 0;
}
