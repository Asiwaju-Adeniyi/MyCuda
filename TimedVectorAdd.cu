#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

__global__ void vecAdd(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    c[tid] = a[tid] + b[tid];
}

int main() {
    const int N = 10;

   
    int a[N], b[N], c[N];

    for (int q = 0; q < N; q++) {
        a[q] = q;
        b[q] = q * 3;
    }

    size_t size = N * sizeof(int);

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);

    vecAdd<<<1, N>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    float gpuDuration = 0.0f;
    cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

    std::cout << "GPU kernel time: " << gpuDuration << " ms\n";
    std::cout << "Vector Addition Result:\n";

    for (int i = 0; i < N; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << "\n";

    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}
