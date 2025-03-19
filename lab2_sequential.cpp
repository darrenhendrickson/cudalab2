//gauto
#include <iostream>
#include <vector>

#define N 1024  // Size of the input signal
#define K 10    // Size of the kernel

void cross_correlation_sequential(const std::vector<int>& input, const std::vector<int>& kernel, std::vector<int>& output) {
    for (int i = 0; i < N - K + 1; ++i) {
        int result = 0;
        for (int j = 0; j < K; ++j) {
            result += input[i + j] * kernel[j];
        }
        output[i] = result;
    }
}

int main() {
    // Generate an input signal and a kernel
    std::vector<int> input(N, 1);  // Example: Initialize with 1
    std::vector<int> kernel(K, 1); // Example: Initialize with 1
    std::vector<int> output(N - K + 1, 0);

    // Compute the cross-correlation sequentially
    cross_correlation_sequential(input, kernel, output);

    // Print the result
    std::cout << "Output: ";
    for (int i = 0; i < output.size(); ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
