#include <cmath>
#include <algorithm>
#include <cassert>
#include "add.cuh"

int main()
{
    const uint max_block_count = 65535;
    const uint per_block_thread_count = 1024;

    const uint array_size = 10000000;
    float* a = new float[array_size];
    float* b = new float[array_size];
    float* c = new float[array_size];
    for(uint i=0; i<array_size; ++i){
        a[i] = i;
        b[i] = array_size - i;
    }


    // allocate GPU memory
    float* dev_a;
    float* dev_b;
    float* dev_c;
    cudaMalloc((void**)&dev_a, array_size * sizeof(float));
    cudaMalloc((void**)&dev_b, array_size * sizeof(float));
    cudaMalloc((void**)&dev_c, array_size * sizeof(float));
    // copy memory from cpu to GPU
    cudaMemcpy(dev_a, a, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, array_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // cuda parallel computing
    uint block_count = std::min(max_block_count, (uint)ceil(array_size/(float)per_block_thread_count));
    cudaCallAddVectorKernel(max_block_count, per_block_thread_count, dev_a, dev_b, dev_c, array_size);

    // copy back
    cudaMemcpy(c, dev_c, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // test, assert
    for(uint i=0; i<array_size; ++i){
        assert(c[i]==array_size);
    }
    printf("done!\n");

    delete[] a;
    delete[] b;
    delete[] c;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}