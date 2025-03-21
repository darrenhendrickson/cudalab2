#include <iostream>
#include <chrono>

#define N 1024         // Size of the input signal
#define KERNEL_SIZE 10 // Size of the filter kernel
#define OUTPUT_SIZE (N - KERNEL_SIZE + 1)

int main() {
    // Allocate memory
    int* h_in = new int[N];
    int* h_kernel = new int[KERNEL_SIZE];
    int* h_out = new int[OUTPUT_SIZE];

    // Initialize input: repeating pattern 0, 1, 2, ..., 9
    for (int i = 0; i < N; i++) {
        h_in[i] = i % 10;
    }

    // Initialize kernel: values 1 through 10
    for (int i = 0; i < KERNEL_SIZE; i++) {
        h_kernel[i] = i + 1;
    }

    // ---- Timing Starts ----
    auto start = std::chrono::high_resolution_clock::now();

    // Perform 1D cross-correlation
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        int result = 0;
        for (int j = 0; j < KERNEL_SIZE; j++) {
            result += h_in[i + j] * h_kernel[j];
        }
        h_out[i] = result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Sequential execution time: " << elapsed.count() << " ms" << std::endl;
    // ---- Timing Ends ----

    // Print the output (20 values per line for readability)
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
