#ifndef ADD_H
#define ADD_H

#include <stdio.h>
#include <cuda_runtime.h>
typedef unsigned int uint;

void cudaCallAddVectorKernel(
    const uint block_count,
    const uint per_block_thread_count,
    const float* a,
    const float* b,
    float* c,
    const uint size
);
#endif