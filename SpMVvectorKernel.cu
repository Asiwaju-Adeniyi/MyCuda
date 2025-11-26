#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename data_type> 

__global__ void csr_spmv_vector_kernel(
    unsigned int n_rows,
    unsigned int *row_ptr,
    const unsigned int *colIdx,
    const data_type *data,
    const data_type *x,
    data_type *y
) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;

    const unsigned int row = warp_id;

data_type sum = 0;

if (row < n_rows) {
    const unsigned int row_start = row_ptr[row];
    const unsigned int row_end = row_ptr[row + 1];

    for (int k = 0; k < row_end; k++)
        sum += data[k] * x[colIdx[k]];

        
}

sum = warp_reduce (sum);

if (lane == 0 && row < n_rows)
y[row] = sum;

}
