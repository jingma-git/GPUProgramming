#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include "add.cuh"
using namespace std;

float cpu_add(float* a, float* b, float* c, uint sz) {
	float time_milli;
	cudaEvent_t start_cpu, stop_cpu;
	cudaEventCreate(&start_cpu);
	cudaEventCreate(&stop_cpu);
	cudaEventRecord(start_cpu);
	for (uint i = 0; i < sz; ++i) {
		c[i] = a[i] + b[i];
	}
	cudaEventRecord(stop_cpu);
	cudaEventSynchronize(stop_cpu);
	cudaEventElapsedTime(&time_milli, start_cpu, stop_cpu);
	return time_milli;
}

int main()
{
    const uint max_block_count = 65535;
    const uint per_block_thread_count = 1024;

    const uint array_size = 1e8;
    float* a = new float[array_size];
    float* b = new float[array_size];
    float* c = new float[array_size];
    for(uint i=0; i<array_size; ++i){
        a[i] = i;
        b[i] = array_size - i;
    }
	float cpu_time = cpu_add(a, b, c, array_size);

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

	cudaEvent_t start_gpu, stop_gpu;
	float gpu_time = -1;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu);
    cudaCallAddVectorKernel(max_block_count, per_block_thread_count, dev_a, dev_b, dev_c, array_size);

    // copy back
    cudaMemcpy(c, dev_c, array_size * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
	cout << "CPU time: " << cpu_time << " milliseconds" << endl;
	cout << "GPU time: " << gpu_time << " milliseconds" << endl;
	cout << endl << "Speedup factor: " <<
		cpu_time / gpu_time << endl << endl;

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