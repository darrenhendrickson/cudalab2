#include <iostream>

#define N 1024         // Size of the input array
#define KERNEL_SIZE 10 // Size of the filter kernel
#define OUTPUT_SIZE (N - KERNEL_SIZE + 1)

int main() {
    // Allocate host memory using C++ new
    int* h_in = new int[N];
    int* h_kernel = new int[KERNEL_SIZE];
    int* h_out = new int[OUTPUT_SIZE];

    // Initialize input array with a fixed pattern: 0,1,2,...,9,0,1,2,...
    for (int i = 0; i < N; i++) {
        h_in[i] = i % 10;
    }

    // Initialize kernel with fixed values: 1, 2, 3, ..., 10
    for (int i = 0; i < KERNEL_SIZE; i++) {
        h_kernel[i] = i + 1;
    }

    // Perform the sequential 1D cross-correlation.
    // Each output element is computed as:
    // h_out[i] = h_in[i]*h_kernel[0] + h_in[i+1]*h_kernel[1] + ... + h_in[i+KERNEL_SIZE-1]*h_kernel[KERNEL_SIZE-1]
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int result = 0;
        for (int j = 0; j < KERNEL_SIZE; j++) {
            result += h_in[i + j] * h_kernel[j];
        }
        h_out[i] = result;
    }

    // Print the output (printing 20 values per line for readability)
    std::cout << "Sequential Output (" << OUTPUT_SIZE << " elements):" << std::endl;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        std::cout << h_out[i] << " ";
        if ((i + 1) % 20 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_in;
    delete[] h_kernel;
    delete[] h_out;

    return 0;
}
