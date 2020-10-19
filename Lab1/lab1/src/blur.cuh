
#ifndef BLUR_DEVICE_CUH
#define BLUR_DEVICE_CUH
#include <cuda_runtime.h>

// __device__ int min_int(int a, int b);
// __device__ int max_int(int a, int b);
// __global__ void gaussian_blur(const unsigned char *const inputChannel,
//                               unsigned char *outputChannel,
//                               int numRows, int numCols,
//                               const float *filter, const int filterWidth);

// This function will be called from the host code to invoke the kernel
// function. Any memory address/pointer locations passed to this function
// must be host addresses. This function will be in charge of allocating
// GPU memory, invoking the kernel, and cleaning up afterwards.
float cuda_call_blur_kernel(const unsigned char *h_image,
                            const float *blur_v,
                            unsigned char *h_image_blur,
                            const unsigned int height,
                            const unsigned int row,
                            const unsigned int filter_size);

#endif
