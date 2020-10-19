### Introduction
Implement guassian image filter (image blur).
1. guassian_blur_cpu: refer to src/blur.cpp Line57-Line82
2. guassian_blur_gpu: refer to src/blur.cu Line26-Line58
### Programming language and hardware details
1. Programming language: C/C++
2. Operating system: Ubuntu 20.04
3. CPU: AMD® Ryzen 7 3700x 8-core processor × 16 | Cpu MHz: 2195.820 | Cache size: 512 KB
4. GPU: GeForce GTX TITAN X/PCIe/SSE2 | Maximum threads in each block: 1024 | Total memory: 12288MB
5. Third-party library: OpenCV 4.3.0
### Compile and run the program
The executable program is in 'build' directory (build/blur), I only provide the version that supports 'input image'. To compile the program, you need to install CMake(https://cmake.org/cmake/help/latest/guide/tutorial/index.html). 
After the CMake is installed in your system, go to 'GaussianBlur' directory, open the terminal, type in the following command
#### Run the program by input image
```
mkdir build
cd build
cmake ..
make
cd ..
./build/blur resources/lena.jpg
```
#### Run the program by generating random number, please refer to 'Discussion-Quantitative Results' section
```
mkdir build
cd build
cmake -D IMG_ON=OFF ..
make
cd ..
./build/blur resources/lena.jpg
```
### Application details and design decisions
1. Which application are you parallesing: convolution operation
2. Why you choose this application: each convolution operation is indepent from each other. Each pixel value in blurred image is the weighted average (by guassian kernel) of a kernel_size region in the input image. Thus, this program is very suitable for parallization.

### Discussion
#### Qualitative Result
![output image](./resources/blur_gpu.jpg)
#### Quantitative Result ( milliseconds )
|Image Size|CPU     |GPU       |Speed Up|
|----------|--------|----------|--------|
|128 x 128 |14.0174 |0.247712  |56.5877 |
|256 x 256 |55.4940 |0.397216  |139.707 |
|512 x 512 |227.952 |1.04365   |218.419 |
|1024 x 1024 |905.56 |3.51798   |257.409 |


### References
https://www.slideshare.net/DarshanParsana/gaussian-image-blurring-in-cuda-c

