# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build

# Include any dependencies generated for this target.
include CMakeFiles/blur.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/blur.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/blur.dir/flags.make

CMakeFiles/blur.dir/src/blur_generated_blur.cu.o: CMakeFiles/blur.dir/src/blur_generated_blur.cu.o.depend
CMakeFiles/blur.dir/src/blur_generated_blur.cu.o: CMakeFiles/blur.dir/src/blur_generated_blur.cu.o.cmake
CMakeFiles/blur.dir/src/blur_generated_blur.cu.o: ../src/blur.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/blur.dir/src/blur_generated_blur.cu.o"
	cd /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src && /usr/bin/cmake -E make_directory /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src/.
	cd /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src/./blur_generated_blur.cu.o -D generated_cubin_file:STRING=/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src/./blur_generated_blur.cu.o.cubin.txt -P /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src/blur_generated_blur.cu.o.cmake

CMakeFiles/blur.dir/src/blur.cpp.o: CMakeFiles/blur.dir/flags.make
CMakeFiles/blur.dir/src/blur.cpp.o: ../src/blur.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/blur.dir/src/blur.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/blur.dir/src/blur.cpp.o -c /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/src/blur.cpp

CMakeFiles/blur.dir/src/blur.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/blur.dir/src/blur.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/src/blur.cpp > CMakeFiles/blur.dir/src/blur.cpp.i

CMakeFiles/blur.dir/src/blur.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/blur.dir/src/blur.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/src/blur.cpp -o CMakeFiles/blur.dir/src/blur.cpp.s

# Object files for target blur
blur_OBJECTS = \
"CMakeFiles/blur.dir/src/blur.cpp.o"

# External object files for target blur
blur_EXTERNAL_OBJECTS = \
"/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/src/blur_generated_blur.cu.o"

blur: CMakeFiles/blur.dir/src/blur.cpp.o
blur: CMakeFiles/blur.dir/src/blur_generated_blur.cu.o
blur: CMakeFiles/blur.dir/build.make
blur: /usr/local/cuda-10.1/lib64/libcudart_static.a
blur: /usr/lib/x86_64-linux-gnu/librt.so
blur: /usr/local/cuda-10.1/lib64/libcudart_static.a
blur: /usr/lib/x86_64-linux-gnu/librt.so
blur: /usr/local/lib/libopencv_dnn.so.4.3.0
blur: /usr/local/lib/libopencv_gapi.so.4.3.0
blur: /usr/local/lib/libopencv_highgui.so.4.3.0
blur: /usr/local/lib/libopencv_ml.so.4.3.0
blur: /usr/local/lib/libopencv_objdetect.so.4.3.0
blur: /usr/local/lib/libopencv_photo.so.4.3.0
blur: /usr/local/lib/libopencv_stitching.so.4.3.0
blur: /usr/local/lib/libopencv_video.so.4.3.0
blur: /usr/local/lib/libopencv_videoio.so.4.3.0
blur: /usr/local/lib/libopencv_imgcodecs.so.4.3.0
blur: /usr/local/lib/libopencv_calib3d.so.4.3.0
blur: /usr/local/lib/libopencv_features2d.so.4.3.0
blur: /usr/local/lib/libopencv_flann.so.4.3.0
blur: /usr/local/lib/libopencv_imgproc.so.4.3.0
blur: /usr/local/lib/libopencv_core.so.4.3.0
blur: CMakeFiles/blur.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable blur"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/blur.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/blur.dir/build: blur

.PHONY : CMakeFiles/blur.dir/build

CMakeFiles/blur.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/blur.dir/cmake_clean.cmake
.PHONY : CMakeFiles/blur.dir/clean

CMakeFiles/blur.dir/depend: CMakeFiles/blur.dir/src/blur_generated_blur.cu.o
	cd /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build /home/server/MaJing/paper_reading/Courses/CalTechCS179/GussianBlur/build/CMakeFiles/blur.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/blur.dir/depend

