#include "add.cuh"


__global__ 
void cudaAddVectorKernel(const float* a, const float* b, float* c, const uint size){
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_index < size){ // ensure current thread is valid
        c[thread_index] = a[thread_index] + b[thread_index];
    }

    // while(thread_index < size){ // ensure current thread is valid
    //     c[thread_index] = a[thread_index] + b[thread_index];
    //     thread_index += gridDim.x * blockDim.x; // cross over elements of a grid?
    // }
}


void cudaCallAddVectorKernel(
    const uint block_count,
    const uint per_block_thread_count,
    const float* a,
    const float* b,
    float* c,
    const uint size
){
    cudaAddVectorKernel<<<block_count, per_block_thread_count>>>(a, b, c, size);
}