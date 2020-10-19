#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "blur.cuh"

using namespace std;

const float PI = 3.14159265358979;

// http://stackoverflow.com/questions/14038589/
// what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpu_errchk(ans)                        \
    {                                          \
        gpu_assert((ans), __FILE__, __LINE__); \
    }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void check_args(int argc, char **argv)
{
#ifdef IMG_ON
    if (argc != 2)
    {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: <input file>\n";
        exit(EXIT_FAILURE);
    }
#else
    if (argc != 3)
    {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: <height> <width>\n";
        exit(EXIT_FAILURE);
    }
#endif
}

void gaussian_blur_cpu(const unsigned char *h_image,
                       const float *filter_v,
                       unsigned char *h_image_blur, int height, int width,
                       int filter_size)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float result = 0.f;
            //For every value in the filter around the pixel (c, r)
            for (int filter_r = -filter_size / 2; filter_r <= filter_size / 2; ++filter_r)
            {
                for (int filter_c = -filter_size / 2; filter_c <= filter_size / 2; ++filter_c)
                {
                    int image_r = min(max(y + filter_r, 0), height - 1);
                    int image_c = min(max(x + filter_c, 0), width - 1);
                    float image_value = static_cast<float>(h_image[image_r * width + image_c]);
                    float filter_value = filter_v[(filter_r + filter_size / 2) * filter_size + filter_c + filter_size / 2];
                    result += image_value * filter_value;
                }
            }
            h_image_blur[y * width + x] = result;
        }
    }
}

int run_gauss_test(int argc, char **argv)
{
    check_args(argc, argv);

    // Generate gaussian kernel
    // https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
    float sigma = 5.0;
    int kernel_size = 9;
    float *filter_v = (float *)malloc(sizeof(float) * kernel_size * kernel_size);
    float filter_sum = 0.;
    for (int r = -kernel_size / 2; r <= kernel_size / 2; ++r)
    {
        for (int c = -kernel_size / 2; c <= kernel_size / 2; ++c)
        {
            float val = expf(-(float)(c * c + r * r) / (2.f * sigma * sigma));
            unsigned int row = (r + kernel_size / 2);
            unsigned int col = (c + kernel_size / 2);
            filter_v[row * kernel_size + col] = val;
            filter_sum += val;
        }
    }
    printf("Guassian Filter\n");
    float inv_filter_sum = 1. / filter_sum;
    for (int r = -kernel_size / 2; r <= kernel_size / 2; ++r)
    {
        for (int c = -kernel_size / 2; c <= kernel_size / 2; ++c)
        {
            unsigned int row = (r + kernel_size / 2);
            unsigned int col = (c + kernel_size / 2);
            filter_v[row * kernel_size + col] *= inv_filter_sum;
            printf("%.6f ", filter_v[row * kernel_size + col]);
        }
        printf("\n");
    }

#ifdef IMG_ON
    cv::Mat img = cv::imread(std::string(argv[1]), cv::IMREAD_GRAYSCALE);
    int height = img.rows, width = img.cols;
    int n_pixels = height * width;
#else
    int height = atoi(argv[1]), width = atoi(argv[2]);
    int n_pixels = height * width;
#endif

    unsigned char *h_image = (unsigned char *)malloc(n_pixels * sizeof(unsigned char));
    unsigned char *image_blur_cpu = (unsigned char *)malloc(n_pixels * sizeof(unsigned char));
    unsigned char *image_blur_gpu = (unsigned char *)malloc(n_pixels * sizeof(unsigned char));
    memset(image_blur_cpu, 0, n_pixels * sizeof(unsigned char));
    memset(image_blur_gpu, 0, n_pixels * sizeof(unsigned char));

#ifdef IMG_ON
    // Read image from file
    memcpy(h_image, img.ptr<unsigned char>(), n_pixels * sizeof(unsigned char));
    cv::Mat orig_img = cv::Mat(height, width, CV_8U, h_image);
    cv::imwrite("resources/input.jpg", orig_img);
#else
    // Generate random data
    for (int i = 0; i < n_pixels; i++)
        h_image[i] = ((unsigned char)rand() % 255);
#endif

    // CPU Blurring
    printf("CPU blurring...\n");
    // Use the CUDA machinery for recording time
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventRecord(start_cpu);
    gaussian_blur_cpu(h_image, filter_v,
                      image_blur_cpu, height, width,
                      kernel_size);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float cpu_time_milliseconds;
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);
    cout << "CPU time: " << cpu_time_milliseconds << " milliseconds" << endl;

    // GPU blurring
    printf("GPU blurring...\n");
    float gpu_time_milliseconds = 0.0;
    gpu_time_milliseconds = cuda_call_blur_kernel(h_image, filter_v,
                                                  image_blur_gpu, height, width,
                                                  kernel_size);

    cout << "GPU time: " << gpu_time_milliseconds << " milliseconds" << endl;
    cout << endl
         << "Speedup factor: " << cpu_time_milliseconds / gpu_time_milliseconds << endl
         << endl;

    // Compare results
    float error = 0.;
    for (int i = 0; i < n_pixels; i++)
    {
        error = image_blur_cpu[i] - image_blur_gpu[i];
    }
    printf("Error between CPU and GPU blurred image: %.6f\n", error / (height * width));

#ifdef IMG_ON
    cv::Mat mat_blur_cpu = cv::Mat(height, width, CV_8U, image_blur_cpu);
    cv::imwrite("resources/blur_cpu.jpg", mat_blur_cpu);

    cv::Mat mat_blur_gpu = cv::Mat(height, width, CV_8U, image_blur_gpu);
    cv::imwrite("resources/blur_gpu.jpg", mat_blur_gpu);
#endif

    // Free memory on host
    free(h_image);
    free(image_blur_cpu);
    free(image_blur_gpu);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    return run_gauss_test(argc, argv);
}
