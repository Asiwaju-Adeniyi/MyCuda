#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    auto startH2D = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    auto stopH2D = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> h2d_time = stopH2D - startH2D;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto startKernel = std::chrono::high_resolution_clock::now();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    auto stopKernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_time_chrono = stopKernel - startKernel;

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float kernel_time_cuda = 0;
    cudaEventElapsedTime(&kernel_time_cuda, startEvent, stopEvent);

    auto startD2H = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    auto stopD2H = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> d2h_time = stopD2H - startD2H;

    std::cout << "Host-to-Device transfer time: " << h2d_time.count() << " ms\n";
    std::cout << "Kernel time (std::chrono): " << kernel_time_chrono.count() << " ms\n";
    std::cout << "Kernel time (CUDA events): " << kernel_time_cuda << " ms\n";
    std::cout << "Device-to-Host transfer time: " << d2h_time.count() << " ms\n";

    std::cout << "Sample output: " << h_C[0] << ", " << h_C[1] << ", " << h_C[2] << "\n";

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

