### Introduction
Implement guassian image filter (image blur). Three Functions:
1. guassian_blur_cpu
2. guassian_blur_gpu
3. guassian_blur_gpu_shared_memory
### Programming language and hardware details
1. Programming language: C/C++
2. Operating system: Ubuntu 20.04
3. CPU: AMD® Ryzen 7 3700x 8-core processor × 16 | Cpu MHz: 2195.820 | Cache size: 512 KB
4. GPU: GeForce GTX TITAN X/PCIe/SSE2 | Maximum threads in each block: 1024 | Total memory: 12288MB
5. Third-party library: OpenCV 4.3.0
6. Compile the program: CMake
7. Run the program:

### Application details and design decisions
1. Which application are you parallesing: convolution operation
2. Why you choose this application: each convolution operation is indepent from each other. Each pixel value in blurred image is the weighted average (by guassian kernel) of a kernel_size region in the input image. Thus, this program is very suitable for parallization.
3. What's interesting?

### Discussion
(1) image size = 512
blocks x=17, y=17
threads per block x=32, y=32

Comparing...
CPU time: 0.002432 milliseconds
GPU time: 0.884256 milliseconds

Speedup factor: 0.00275033

(2) image size = 4028
Comparing...
CPU time: 0.002624 milliseconds
GPU time: 51.4521 milliseconds
Speedup factor: 5.09989e-05

(3) image size = 16112
Comparing...
CPU time: 0.00304 milliseconds
GPU time: 699.638 milliseconds

Speedup factor: 4.3451e-06

### References
https://www.slideshare.net/DarshanParsana/gaussian-image-blurring-in-cuda-c

