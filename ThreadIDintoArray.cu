#include <cstdio> 
#include <iostream> 

__global__ void PrintThreadIDs(int* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = tid;
}


int main() {
   int numThreadsPerBlock = 8;
   int numBlock = 2;
   int totalThreads = numThreadsPerBlock * numBlock;

   int* h_output = new int[totalThreads];
   int* d_output;
   cudaMalloc(&d_output, totalThreads * sizeof(int));

   PrintThreadIDs<<<numBlock, numThreadsPerBlock>>>(d_output);

   cudaMemcpy(h_output, d_output, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);

   std::cout << "Thread IDs:\n";
   for(int i = 0; i < totalThreads; i++) {
    std::cout << h_output[i] << " ";
   }

   std::cout << std::endl;

   delete[] h_output;
   cudaFree(d_output);

   return 0;




}
