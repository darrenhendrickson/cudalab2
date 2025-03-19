//use gauto to push to git
//example code given
#include <iostream>

#include <cuda_runtime.h>

 

#define BLOCK_SIZE 256  // Define the block size

#define RADIUS 2        // Define the radius of the kernel

#define KERNEL_SIZE (2 * RADIUS + 1)  // Kernel size

#define N 10  // Size of the input array

 

__global__ void cross_correlation_1d(int *in, int *out, int *kernel, int N) {

    // Allocate shared memory for the block, including halo regions

    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];

 

    // Calculate global index in the input array

    int gindex = threadIdx.x + blockIdx.x * blockDim.x;

 

    // Calculate local index in shared memory

    int lindex = threadIdx.x + RADIUS;

 

    // Load input elements into shared memory

    if (gindex < N) {

        temp[lindex] = in[gindex];

    } else {

        temp[lindex] = 0;  // Handle out-of-bounds by padding with zero

    }

 

    // Load halo elements into shared memory

    if (threadIdx.x < RADIUS) {

        // Left halo

        if (gindex - RADIUS >= 0) {

            temp[lindex - RADIUS] = in[gindex - RADIUS];

        } else {

            temp[lindex - RADIUS] = 0;  // Pad with zero if out-of-bounds

        }

 

        // Right halo

        if (gindex + BLOCK_SIZE < N) {

            temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];

        } else {

            temp[lindex + BLOCK_SIZE] = 0;  // Pad with zero if out-of-bounds

        }

    }

 

    // Synchronize to ensure all threads have loaded their data

    __syncthreads();

 

    // Apply the cross-correlation

    if (gindex < N) {  // Only compute for valid output indices

        int result = 0;

        for (int offset = 0; offset < KERNEL_SIZE; offset++) {

            result += temp[lindex + offset] * kernel[offset];

        }

        out[gindex] = result;  // Store the result at the corresponding index

    }

}

 

int main() {

    int *h_in, *h_out, *h_kernel;

    int *d_in, *d_out, *d_kernel;

 

    // Allocate host memory

    h_in = (int *)malloc(N * sizeof(int));

    h_out = (int *)malloc(N * sizeof(int));

    h_kernel = (int *)malloc(KERNEL_SIZE * sizeof(int));

 

    // Initialize input array

    for (int i = 0; i < N; i++) {

        h_in[i] = 1;  // Example: Initialize with 1

    }

 

    // Initialize kernel (e.g., [1, 2, 3, 2, 1] for RADIUS = 2)

    for (int i = 0; i < KERNEL_SIZE; i++) {

        h_kernel[i] = i + 1;  // Example: Linear kernel

    }

 

    // Allocate device memory

    cudaMalloc((void **)&d_in, N * sizeof(int));

    cudaMalloc((void **)&d_out, N * sizeof(int));

    cudaMalloc((void **)&d_kernel, KERNEL_SIZE * sizeof(int));

 

    // Copy input data and kernel to device

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * sizeof(int), cudaMemcpyHostToDevice);

 

    // Launch kernel

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cross_correlation_1d<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_kernel, N);

 

    // Copy result back to host

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

 

    // Print the result

    std::cout << "Output: ";

    for (int i = 0; i < N; i++) {

        std::cout << h_out[i] << " ";

    }

    std::cout << std::endl;

 

    // Free memory

    free(h_in);

    free(h_out);

    free(h_kernel);

    cudaFree(d_in);

    cudaFree(d_out);

    cudaFree(d_kernel);

 

    return 0;

}