#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

template <class T>
__device__ T warp_reduce (T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    return val;
}



template <typename data_type>
__global__ void csr_spmv_kernel(
    unsigned int n_rows,
    const unsigned int *row_ptr,   // where each row starts
    const unsigned int *colIdx,    // column indices for each non-zero
    const data_type *data,         // non-zero values
    const data_type *x,            // input vector
    data_type *y                   // output vector
) {
    // global thread id
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // which warp (team) among all threads
    const unsigned int warp_id = tid / 32;   // <-- use global tid
    const unsigned int lane = threadIdx.x % 32; // lane inside the block warp (0..31)
    const unsigned int row = warp_id;        // assign one warp -> one matrix row

    if (row >= n_rows) return;  // warp has no row to work on

    // find where this row's non-zeros are in data/colIdx
    const unsigned int row_start = row_ptr[row];
    const unsigned int row_end   = row_ptr[row + 1];

    // each lane handles a strided subset of the row
    data_type sum = 0;

    // start at row_start + lane, step by warpSize (32)
    for (unsigned int idx = row_start + lane; idx < row_end; idx += 32) {
        unsigned int col = colIdx[idx];
        data_type val = data[idx];
        sum += val * x[col];
    }

    // reduce the 32 lane sums down to a single sum in the warp
    sum = warp_reduce(sum);   // after this, lane 0 contains the total for the row

    // only lane 0 writes the final result
    if (lane == 0) {
        y[row] = sum;
    }
}
