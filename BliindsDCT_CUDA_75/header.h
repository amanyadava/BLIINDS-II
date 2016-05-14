// header.h
#define PI 3.14159265358979323846
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
//=============================================================================
// Include OpenCV to read in image
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//=============================================================================
// Include CUDA runtime & NSIGHT annotations
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
//#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
//#include "nvToolsExt.h" // Core NVTX API
//#include "nvToolsExtCuda.h"
//=============================================================================
#include <cufft.h>
#include <thrust/execution_policy.h>
#include <thrust\host_vector.h>
#include <thrust/reduce.h>
#include <thrust\sort.h>
//=============================================================================
// CPU Timer
#include <windows.h>
//=============================================================================
// Basic Stuff:
#include <fstream>
#include <iostream>
#include <stdio.h>	// not sure what for
#include <cstdlib>	// not sure what for
#include <cstring>
#include <math.h>

//=============================================================================
// Declare Functions:
void kernel_wrapper(const cv::Mat &Mat_in);
void kernel_wrapper(const cv::Mat &Mat_in, cufftHandle &p);
void device_rst();