#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename data_type>

__global__ void csr_spmv_kernel(
    unsigned int n_rows,
    const unsigned int *colIdx,
    const unsigned int *row_ptr,
    const data_type data,
    const data_type *x,
    data_type *y
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows) {
        unsigned int row_start = row_ptr[row];
        unsigned int row_end = row_ptr[row + 1];

        data_type sum = 0;

        for (unsigned int k = 0; k < row_end; k++) {
            sum += data[k] * x[colIdx[k]];

        y[x] = sum;
        }
    }
}
