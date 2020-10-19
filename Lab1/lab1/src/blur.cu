
#include "blur.cuh"

#include <cstdio>
#include <cuda_runtime.h>

// devicel functions:
__device__ 
int min_int(int a, int b)
{
  if (a <= b)
    return a;
  else
    return b;
}

__device__ 
int max_int(int a, int b)
{
  if (a >= b)
    return a;
  else
    return b;
}

__global__ 
void gaussian_blur_shared(const unsigned char * inputChannel,
                              unsigned char * outputChannel,
                              int imgHeight, int imgWidth,
                              const float * filter, const int filterWidth)
{
  extern __shared__ float loadIn[];
  int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;

  if(col < imgWidth && row < imgHeight){
    loadIn[threadIdx.y*blockDim.x + threadIdx.x] = static_cast<float>(inputChannel[row * imgWidth + col]);
    __syncthreads();

    if(threadIdx.x< blockDim.x && threadIdx.y<blockDim.y){
      float result = 0.0;
      for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; ++filter_r)
      {
        int i = filter_r + filterWidth / 2;
        for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; ++filter_c)
        {
          int j = filter_c + filterWidth / 2;
          int r = min_int(max_int(threadIdx.y+i, 0), blockDim.y-1);
          int c = min_int(max_int(threadIdx.x+j, 0), blockDim.x-1);
          float filter_value = filter[i*filterWidth+j];
          // printf("[block %d, %d], filter_r=%d, filter_c=%d, image_r=%d, image_c=%d, image_v=%d, filter_v=%.6f\n", 
          //   blockIdx.x, blockIdx.y, filter_r, filter_c, threadIdx.y+i, threadIdx.x+j, loadIn[(threadIdx.y+i)*blockDim.x+j+threadIdx.x], filter_value);
          result+= loadIn[r*blockDim.x+c] * filter_value;
          // result+= loadIn[(threadIdx.y+i)*blockDim.x+j+threadIdx.x] / 81;
        }
        //printf("thread=%d filter_r=%d, result=%.6f\n", threadIdx.y*blockDim.x+threadIdx.x, i, result);
      }

      // // mean blur
      // for(int i=0; i<filterWidth; i++){
      //   for(int j=0; j<filterWidth; j++){
      //     int r = min_int(max_int(threadIdx.y+i, 0), blockDim.y-1);
      //     int c = min_int(max_int(threadIdx.x+j, 0), blockDim.x-1);
      //     result+= loadIn[r*blockDim.x+c] / 81;
      //   }
      // }

      outputChannel[row*imgWidth + col] = result;
      // outputChannel[row*imgWidth + col] = loadIn[threadIdx.y*blockDim.x+threadIdx.x];
  }
  }
 
}

__global__ 
void gaussian_blur(const unsigned char * inputChannel,
                              unsigned char * outputChannel,
                              int numRows, int numCols,
                              const float * filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                       blockIdx.y * blockDim.y + threadIdx.y);
  
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  float result = 0.f;
  //For every value in the filter around the pixel (c, r)
  for (int filter_r = -filterWidth / 2; filter_r <= filterWidth / 2; ++filter_r)
  {
    for (int filter_c = -filterWidth / 2; filter_c <= filterWidth / 2; ++filter_c)
    {
      //Find image position for this filter position
      int image_r = min_int(max_int(thread_2D_pos.y + filter_r, 0), static_cast<int>(numRows - 1));
      int image_c = min_int(max_int(thread_2D_pos.x + filter_c, 0), static_cast<int>(numCols - 1));

      float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
      float filter_value = filter[(filter_r + filterWidth / 2) * filterWidth + filter_c + filterWidth / 2];
      // printf("[block %d, %d], filter_r=%d, filter_c=%d, image_r=%d, image_c=%d, image_v=%d, filter_v=%.6f\n", 
      // blockIdx.x, blockIdx.y, filter_r, filter_c, image_r, image_c, image_value, filter_value);
      result += image_value * filter_value;
    }
  }
  __syncthreads();
  outputChannel[thread_1D_pos] = result;
}

float cuda_call_blur_kernel(const unsigned char *h_image,
                            const float *h_filter,
                            unsigned char *h_image_blur,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int filter_size)
{
  // Use the CUDA machinery for recording time
  cudaEvent_t start_gpu, stop_gpu;
  float time_milli = -1;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);
  cudaEventRecord(start_gpu);
  unsigned int n_pixels = height * width;

  // Allocate GPU memory for the raw input image & filter
  unsigned char *d_image;
  cudaMalloc((void **)&d_image, n_pixels * sizeof(unsigned char));
  cudaMemcpy(d_image, h_image, n_pixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

  float *d_filter;
  cudaMalloc((void **)&d_filter, filter_size * filter_size * sizeof(float));
  cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate GPU memory for output blurred image
  unsigned char *d_image_blur;
  cudaMalloc((void **)&d_image_blur, n_pixels * sizeof(unsigned char));
  cudaMemset(d_image_blur, 0, n_pixels * sizeof(unsigned char));

  // Call the kernel function.
  const dim3 blockDim(32, 32); // 1024 threads
  const dim3 gridDim(width/blockDim.x + 1, height/blockDim.y+1);
  gaussian_blur_shared<<<gridDim, blockDim, blockDim.x*blockDim.y*sizeof(float)>>>(d_image,  d_image_blur, height, width, d_filter, filter_size);
  // gaussian_blur<<<gridDim, blockDim>>>(d_image,  d_image_blur, height, width, d_filter, filter_size);

  // Check for errors on kernel call
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
    fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
  else
    fprintf(stderr, "No kernel error detected\n");

  // Copy from the GPU to host memory
  cudaMemcpy(h_image_blur, d_image_blur, n_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

 
  cudaFree(d_image);
  cudaFree(d_filter);
  cudaFree(d_image_blur);
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);

  return time_milli;
}
