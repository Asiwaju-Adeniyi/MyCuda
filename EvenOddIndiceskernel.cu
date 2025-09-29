#include <iostream>
#include <cuda_runtime.h>

__global__ void colorEvenOdd(float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (tid % 2 == 0) {
            output[tid] = 1;
        } else {
            output[tid] = 0;
        }
    }
}

int main() {
    int N = 16;

    float* h_output = new float[N];
    float* d_output;
    cudaMalloc(&d_output, N * sizeof(float));

    int numThreadsPerBlock = 35;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    colorEvenOdd<<<numBlocks, numThreadsPerBlock>>>(d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Here is the pattern:\n";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << std::endl;
    }

    delete[] h_output;
    cudaFree(d_output);

    return 0;
}
