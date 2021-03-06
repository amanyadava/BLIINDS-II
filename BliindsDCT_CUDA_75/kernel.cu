#include "header.h"
#include "bliinds_info.h"

__constant__ float dc_r_info[9971];

__global__ void pad(float const* src, const int size, float * new_img) {
	//should have threads = blocks = 512 + overlap + blk_size + overlap
	//and both are one dimensional
	int x = threadIdx.x;
	int y = blockIdx.x;
	if (x<1 || x>size || y<1 || y>size)
		new_img[x + y * (size + 2 * 1 + 3)] = 0;
	else
		new_img[x + y * (size + 2 * 1 + 3)] = src[x - 1 + (y - 1) * size];
}

__global__ void rearrangeForCuFFT(float const * new_img, const int size, cufftComplex * rearr_img) {
	// threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	int const pos = (y / (size / 3 + 1)) * 3 * (size + 5) + (y % (size / 3 + 1)) * 3; // position in new_img of first element to be copied
	//*****Important*** change this to row-wise
	rearr_img[y * 50 + x / 5 * 10 + x % 5].x = new_img[pos + x % 5 + x / 5 * (size + 5)];
	rearr_img[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].x = new_img[pos + x % 5 + x / 5 * (size + 5)];
	rearr_img[y * 50 + x / 5 * 10 + x % 5].y = 0;
	rearr_img[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].y = 0;
}

__global__ void setZero(float * array) {
	array[threadIdx.x + blockIdx.x * blockDim.x] = 0.0f;
}

__global__ void rearrangeForDCT(float const * new_img, const int size, float * rearr_img) {
	// threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	int const pos = (y / (size / 3 + 1)) * 3 * (size + 5) + (y % (size / 3 + 1)) * 3; // position in new_img of first block element to be copied
	//*****Important*** change this to row-wise
	rearr_img[y * 25 + x] = new_img[pos + x % 5 + x / 5 * (size + 5)];
	/*if (x == 0 && y == 0)
		printf("I can print\n");*/
}

// Higher warps
__global__ void rearrangeForDCTv2(float const * new_img, const int size, float * rearr_img) {
	// threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 4;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = y + sblkIdx;
	int const pos = (gblkIdx / (size / 3 + 1)) * 3 * (size + 5) + (gblkIdx % (size / 3 + 1)) * 3; // position in new_img of first block element to be copied
	//*****Important*** change this to row-wise
	if (x<25)
		rearr_img[gblkIdx * 32 + x] = new_img[pos + x % 5 + x / 5 * (size + 5)];
}

__global__ void transposeForCuFFT(cufftComplex const*read, cufftComplex *write) {
	// threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	float temp1;// temp2;
	temp1 = (read[y * 50 + x % 5 * 10 + x / 5].x * cos(PI * (x / 5) / 10.0) + read[y * 50 + x % 5 * 10 + x / 5].y * sin(PI * (x / 5) / 10.0)) / 2.0 * (x / 5 == 0 ? sqrt(0.2) : sqrt(0.4));
	//temp2 = (read[y * 50 + x % 5 * 10 + x / 5].y * cos(PI * (x / 5) / 10.0) - read[y * 50 + x % 5 * 10 + x / 5].x * sin(PI * (x / 5) / 10.0)) / 2.0 * (x % 5 == 0 ? sqrt(0.2) : sqrt(0.4));
	//__syncthreads();
	write[y * 50 + x / 5 * 10 + x % 5].x = temp1;//fabsf(temp1) > 0.00001 ? temp1 : 0;
	write[y * 50 + x / 5 * 10 + x % 5].y = 0;
	write[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].x = temp1; //fabsf(temp1) > 0.00001 ? temp1 : 0;
	write[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].y = 0;
}

__global__ void dct55(const float * rearr_img, const double* dctmtx, float* dctImg) {
	//threads = 25, blocks = (512/3 +1)^2

	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ double img[25];
	__shared__ double dct[25];
	img[x] = rearr_img[y * 25 + x];
	dct[x] = dctmtx[x];
	double temp = 0.0;
	__syncthreads();
	/*if (x == 0) {
		if (y == 450) {
			for (int i = 0; i < 25; i++)
				printf("%0.20f\n", img[i]);
			printf("\n");
		}
	}*/
	for (int i = 0; i < 5; i++) {
		temp += dct[5 * (x / 5) + i] * (img[5 * i + x % 5]);
	}
	__syncthreads();
	img[x] = temp;
	__syncthreads();
	temp = 0.0;
	for (int i = 0; i < 5; i++) {
		temp += img[5 * (x / 5) + i] * dct[5 * (x % 5) + i];
	}
	dctImg[y * 25 + x/*5*(x%5) + x/5*/] = temp; //fabsf(temp) > 0.0000001 ? temp : 0;
	//__syncthreads();
	/*if (x == 0) {
		if (y == 7155) {
			for (int i = 0; i < 25; i++)
				printf("%0.20f, %0.20f\n", rearr_img[y*25 + i], dctImg[y * 25 + i]);
			printf("\n");
		}
	}*/
}

__global__ void dct55v2(float* rearr_img, const double* dctmtx, float* dctImg) {
	int const x = threadIdx.x%32;
	int const y = blockIdx.x*8;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = (y + sblkIdx) * 32;
	__shared__ float img[32*8];
	__shared__ double dct[32];
	img[threadIdx.x] = rearr_img[gblkIdx + x];
	dct[x] = dctmtx[x];
	double temp = 0.0f;
	for (int i = 0; i < 5; i++) {
		temp += dct[5 * (x / 5) + i] * img[sblkIdx * 32 + 5 * i + x % 5];
	}
	if (x<25)
		img[threadIdx.x] = temp;
	temp = 0.0f;
	for (int i = 0; i < 5; i++) {
		temp += img[sblkIdx * 32 + 5 * (x / 5) + i] * dct[5 * (x % 5) + i];
	}
	if (x>0 && x<25)
		dctImg[gblkIdx + x/*5*(x%5) + x/5*/] = temp;
	/*if (gblkIdx == 0 && x < 32) {
		printf("%f\n", dctImg[gblkIdx + x]);
	}*/
}

// Merge rearrange and DCT into one kernel. Hoping to avoid register spilling
__global__ void rearrangeAndDCT55(float const* new_img, const int size, const double* dctmtx, float* dctImg) {
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 8;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = y + sblkIdx;
	int const pos = (gblkIdx / (size / 3 + 1)) * 3 * (size + 5) + (gblkIdx % (size / 3 + 1)) * 3; // position in new_img of first block element to be copied
	__shared__ float img[32 * 8];
	__shared__ double dct[32];
	//*****Important*** change this to row-wise
	//img[threadIdx.x] = 0;
	//if (x<25)

	//int const gblkIdx = (y + sblkIdx) * 32;
	img[threadIdx.x] = new_img[pos + x % 5 + x / 5 * (size + 5)];
	dct[x] = dctmtx[x];
	double temp = 0.0f;
	for (int i = 0; i < 5; i++) {
		temp += dct[5 * (x / 5) + i] * img[sblkIdx * 32 + 5 * i + x % 5];
	}
	if (x<25)
		img[threadIdx.x] = temp;
	temp = 0.0f;
	for (int i = 0; i < 5; i++) {
		temp += img[sblkIdx * 32 + 5 * (x / 5) + i] * dct[5 * (x % 5) + i];
	}
	if (x>0 && x<25)
		dctImg[gblkIdx * 32 + x/*5*(x%5) + x/5*/] = temp;
}

__global__ void copyDCT(cufftComplex const*dct_img, float *dctImg) {
	//threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	//dctImg[y * 25 + x] = dct_img[y * 50 + x / 5 * 10 + x % 5].x; 
	dctImg[y * 25 + x] = dct_img[y * 50 + x / 5 * 10 + x % 5].x; // (fabsf(dct_img[y * 50 + x / 5 * 10 + x % 5].x) > 0.0001 ? dct_img[y * 50 + x / 5 * 10 + x % 5].x : 0);
}

__global__ void rearrangeTest(cufftComplex * d_rearr_in){
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	d_rearr_in[y * 50 + x / 5 * 10 + x % 5].x = (x / 5)*(x % 5);
	d_rearr_in[y * 50 + x / 5 * 10 + x % 5].y = 0;
	d_rearr_in[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].x = (x / 5)*(x % 5);
	d_rearr_in[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].y = 0;
}
__global__ void copyTest(cufftComplex const* d_rearr_in, float *dctImg) {
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	dctImg[y * 20 + x] = (abs(d_rearr_in[y * 20 + x].x) > 0.00001 ? d_rearr_in[y * 20 + x].x : 0);
}

__global__ void rho_dct(float const* d_dctImg, float * coeff_freq_var) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[25];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();

	if (x == 0) {
		float mean_abs = 0, std_gauss = 0;
		for (int i = 1; i < 25; i++) {			
			mean_abs += abs(dctBlock[i]);
			/*if (y == 450) {
				printf("%0.20f, %0.20f\n", mean_abs, abs(dctBlock[i]));
			}*/
		}
		
		mean_abs = mean_abs / 24.0;
		/*if (mean_abs < 0.0001) {
			coeff_freq_var[y] = 0;
			return;
		}*/
		for (int i = 1; i < 25; i++) {
			float temp = fabs(dctBlock[i]) - mean_abs;
			std_gauss += temp * temp;
			/*if (y == 450) {
				printf("%0.20f, %0.20f\n", std_gauss, temp*temp);
			}*/
		}
		std_gauss = sqrt(std_gauss / 23.0);
		coeff_freq_var[y] = std_gauss / (mean_abs + 0.0000001);
		/*if (y == 450) {
			printf("std_gauss: %0.20f, \tmean_abs: %0.20f, \tcoeff: %0.20f\n", std_gauss, mean_abs, coeff_freq_var[y]);
		}*/
		
	}
		
}

__device__ inline void MyAtomicAdd(float *address, float value)
{
	int oldval, newval, readback;

	oldval = __float_as_int(*address);
	newval = __float_as_int(__int_as_float(oldval) + value);
	while ((readback = atomicCAS((int *)address, oldval, newval)) != oldval)
	{
		oldval = readback;
		newval = __float_as_int(__int_as_float(oldval) + value);
	}
}

// Higher number of warps
__global__ void rho_dct2(float const* d_dctImg, float * coeff_freq_var) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 16;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = (y + sblkIdx) * 32;

	__shared__ float dctBlock[32 * 16];
	dctBlock[threadIdx.x] = fabs(d_dctImg[gblkIdx + x]);
	dctBlock[sblkIdx * 32] = 0;
	//__syncthreads();

	//if (x == 0) {
	float mean_abs = 0, std_gauss = 0;
	#pragma unroll
	for (int i = 1; i < 25; i++) {
		mean_abs += dctBlock[sblkIdx * 32 + i];
	}
	mean_abs /= 24.0f;
	dctBlock[threadIdx.x] -= mean_abs;
#pragma unroll
	for (int i = 1; i < 25; i++) {
		//float temp = dctBlock[sblkIdx * 32 + i] - mean_abs;
		std_gauss += dctBlock[sblkIdx * 32 + i] * dctBlock[sblkIdx * 32 + i];
	}
	std_gauss = sqrt(std_gauss / 23.0f);
	coeff_freq_var[gblkIdx / 32] = std_gauss / (mean_abs + 0.0000001f);
	//}
}

__global__ void gama_dct(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[25];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();
	if (x == 0) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
	 	mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		const float rho = var_gauss / (mean_abs + 0.0000001);

		float gamma_gauss = 11.0;
		for (int j = 0; j < 9970; j++) {
			if (rho>r_vector[j + 1] && rho <= r_vector[j]) {
				gamma_gauss = g_vector[j];
			}
		}
		d_gama[y] = gamma_gauss;
	}
}

__global__ void gama_dct2(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 1024
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[9996];
	// copy first 25 elements as dctImg and the remaining as r_vec so that r_vec completely fits in shared memory.
	dctBlock[x] = x<25?d_dctImg[y * 25 + x]:r_vector[x-25];
	__syncthreads();
	if (x < 32) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
		mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		dctBlock[0] = var_gauss / (mean_abs + 0.0000001);
	}
	else {
		//int n = 999 + 10*(x - 32);// 1024 - 25 + x - 32
		int n = 999 + x - 32;
		/*for (int i = n; i < n + 10; i++) {
			if (i < 9971) {
				dctBlock[i + 25] = r_vector[i];
			}
		}*/		
		while (n < 9971) {
			dctBlock[n + 25] = r_vector[n];
			n += 992;
		}
	}
	__syncthreads();

	/*if (x == 0) {
		float gamma_gauss = 11.0, rho = dctBlock[0];
		for (int j = 25; j < 9995; j++) {
			if (rho>dctBlock[j + 1] && rho <= dctBlock[j]) {
				gamma_gauss = g_vector[j-25];
			}
		}
		d_gama[y] = gamma_gauss;
	}
	*/

	float rho = dctBlock[0];
	dctBlock[1] = 11.0; // being used as gamma_gauss over here
	for (int j = 10 * x; j <= 10 * x + 12; j++) {
		if (j < 9970) {
			int idx = 25 + j;
			if (rho>dctBlock[idx + 1] && rho <= dctBlock[idx]) {
				dctBlock[1] = g_vector[j];
			}
		}		
	}
	__syncthreads();
	d_gama[y] = dctBlock[1];
}

// gama_dct3 tries to use all 1024 threads for populating the shared memory with r_vector.
// Resulted in a 3 ms increase in time over gama_dct2. (29ms over 26 ms of gama_dct2)
__global__ void gama_dct3(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 1024
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[9996];
	// copy first 25 elements as dctImg and the remaining as r_vec so that r_vec completely fits in shared memory.
	dctBlock[x] = x<25 ? d_dctImg[y * 25 + x] : r_vector[x - 25];
	int n = 999 + x;
	while (n < 9971) {
		dctBlock[n + 25] = r_vector[n];
		n += 1024;
	}
	__syncthreads();
	if (x < 32) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
		mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		dctBlock[0] = var_gauss / (mean_abs + 0.0000001);
	}
	__syncthreads();

	float rho = dctBlock[0];
	dctBlock[1] = 11.0; // being used as gamma_gauss over here
	for (int j = 10 * x; j <= 10 * x + 12; j++) {
		if (j < 9970) {
			int idx = 25 + j;
			if (rho>dctBlock[idx + 1] && rho <= dctBlock[idx]) {
				dctBlock[1] = g_vector[j];
			}
		}
	}
	__syncthreads();
	d_gama[y] = dctBlock[1];
}

// gama_dct4 makes use of r_vector in the constant memory.
// Takes 159 ms for all the three runs! :(
__global__ void gama_dct4(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[25];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();
	if (x == 0) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
		mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		const float rho = var_gauss / (mean_abs + 0.0000001);

		float gamma_gauss = 11.0;
		for (int j = 0; j < 9970; j++) {
			if (rho>dc_r_info[j + 1] && rho <= dc_r_info[j]) {
			//if (rho>r_vector[j + 1] && rho <= r_vector[j]) {
				gamma_gauss = g_vector[j];
			}
		}
		d_gama[y] = gamma_gauss;
	}
}

// gama_dct5 copies r_vector into shared memory from const memory instead of global memory
// takes 108 ms, combined for all the three invocations :(
__global__ void gama_dct5(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 1024
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[9996];
	// copy first 25 elements as dctImg and the remaining as r_vec so that r_vec completely fits in shared memory.
	dctBlock[x] = x<25 ? d_dctImg[y * 25 + x] : dc_r_info[x - 25];
	__syncthreads();
	if (x < 32) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
		mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		dctBlock[0] = var_gauss / (mean_abs + 0.0000001);
	}
	else {
		int n = 999 + x - 32;
		while (n < 9971) {
			dctBlock[n + 25] = dc_r_info[n];
			n += 992;
		}
	}
	__syncthreads();

	float rho = dctBlock[0];
	dctBlock[1] = 11.0; // being used as gamma_gauss over here
	for (int j = 10 * x; j <= 10 * x + 12; j++) {
		if (j < 9970) {
			int idx = 25 + j;
			if (rho>dctBlock[idx + 1] && rho <= dctBlock[idx]) {
				dctBlock[1] = g_vector[j];
			}
		}
	}
	__syncthreads();
	d_gama[y] = dctBlock[1];
}

// gama_dct6 returns the rho values to d_gama
__global__ void gama_dct6(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	__shared__ float dctBlock[25];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();
	if (x == 0) {
		float mean_gauss = 0;
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[i];
		}
		mean_gauss = mean_gauss / 24.0;
		float var_gauss = 0;
		float mean_abs = 0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += temp * temp;
			mean_abs += temp;
		}
		var_gauss = var_gauss / 23.0;
		mean_abs = mean_abs / 24.0;
		mean_abs *= mean_abs;
		const float rho = var_gauss / (mean_abs + 0.0000001);

		d_gama[y] = rho;
	}
}

// gama_dct6 with higher warps
__global__ void gama_dct62(float const * d_dctImg, float const * g_vector, float const * r_vector, float * d_gama) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 16;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = (y + sblkIdx) * 32;
	__shared__ float dctBlock[32*16];
	dctBlock[threadIdx.x] = d_dctImg[gblkIdx + x];
	//__syncthreads();
	//if (x == 0) {
		float mean_gauss = 0;
#pragma unroll
		for (int i = 1; i < 25; i++) {
			mean_gauss += dctBlock[sblkIdx * 32 + i];
		}
		mean_gauss = mean_gauss / 24.0f;
		float var_gauss = 0;
		float mean_abs = 0;
		dctBlock[sblkIdx * 32 + x] = fabsf(dctBlock[sblkIdx * 32 + x] - mean_gauss);
#pragma unroll
		for (int i = 1; i < 25; i++) {
			//float temp = fabsf(dctBlock[i] - mean_gauss);
			var_gauss += dctBlock[sblkIdx * 32 + i] * dctBlock[sblkIdx * 32 + i];
			mean_abs += dctBlock[sblkIdx * 32 + i];
		}
		var_gauss = var_gauss / 23.0f;
		mean_abs = mean_abs / 24.0f;
		mean_abs *= mean_abs;
		const float rho = var_gauss / (mean_abs + 0.0000001f);

		d_gama[gblkIdx / 32] = rho;
	//}
}

__global__ void gama_dct6_3(float * d_rho, float const * g_vector, float const * r_vector, float * d_gama, int max) {
	int const pos = threadIdx.x + blockIdx.x * blockDim.x;
	//if (pos < max) {
		float const rho = d_rho[pos];
		int left(0), right(9970), mid(4985);
		float gamma_gauss = 11;
		while (right > left) {
			mid = (left + right) / 2;
			float r_vec_mid_1 = r_vector[mid + 1];
			if (rho > r_vec_mid_1 && rho <= r_vector[mid]) {
				gamma_gauss = g_vector[mid];
				break;
			}
			else if (rho <= r_vec_mid_1) {
				left = mid + 1;
				continue;
			}
			else {
				right = mid;
			}
		}
		d_gama[pos] = gamma_gauss;
	//}	
	/*float gamma_gauss = 11.0;
		for (int j = 0; j < 9970; j++) {
			if (rho>r_vector[j + 1] && rho <= r_vector[j]) {
				gamma_gauss = g_vector[j];
			}
		}
		d_gama[y] = gamma_gauss;
		*/
}

__global__ void oriented_dct_rho(float const * d_dctImg, float * ori_rho, int orient) {
	//plan grids = (512/3 + 1)^2, threads = 8
	__shared__ float dctBlock[8];
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	if (orient == 1) {
		if (x == 0) {
			dctBlock[0] = fabs(d_dctImg[blockIdx.x * 25 + 1]);
			dctBlock[1] = fabs(d_dctImg[blockIdx.x * 25 + 2]);
			dctBlock[2] = fabs(d_dctImg[blockIdx.x * 25 + 7]);
			dctBlock[3] = fabs(d_dctImg[blockIdx.x * 25 + 3]);
			dctBlock[4] = fabs(d_dctImg[blockIdx.x * 25 + 8]);
			dctBlock[5] = fabs(d_dctImg[blockIdx.x * 25 + 4]);
			dctBlock[6] = fabs(d_dctImg[blockIdx.x * 25 + 9]);
			dctBlock[7] = fabs(d_dctImg[blockIdx.x * 25 + 14]);
		}
	}
	else if (orient == 2) {
		if (x == 0) {
			dctBlock[0] = fabsf(d_dctImg[blockIdx.x * 25 + 6]);
			dctBlock[1] = fabsf(d_dctImg[blockIdx.x * 25 + 12]);
			dctBlock[2] = fabsf(d_dctImg[blockIdx.x * 25 + 17]);
			dctBlock[3] = fabsf(d_dctImg[blockIdx.x * 25 + 13]);
			dctBlock[4] = fabsf(d_dctImg[blockIdx.x * 25 + 18]);
			dctBlock[5] = fabsf(d_dctImg[blockIdx.x * 25 + 23]);
			dctBlock[6] = fabsf(d_dctImg[blockIdx.x * 25 + 19]);
			dctBlock[7] = fabsf(d_dctImg[blockIdx.x * 25 + 24]);
		}
	}
	else if (orient == 3) {
		if (x == 0) {
			dctBlock[0] = fabsf(d_dctImg[blockIdx.x * 25 + 5]);
			dctBlock[1] = fabsf(d_dctImg[blockIdx.x * 25 + 10]);
			dctBlock[2] = fabsf(d_dctImg[blockIdx.x * 25 + 15]);
			dctBlock[3] = fabsf(d_dctImg[blockIdx.x * 25 + 20]);
			dctBlock[4] = fabsf(d_dctImg[blockIdx.x * 25 + 11]);
			dctBlock[5] = fabsf(d_dctImg[blockIdx.x * 25 + 16]);
			dctBlock[6] = fabsf(d_dctImg[blockIdx.x * 25 + 21]);
			dctBlock[7] = fabsf(d_dctImg[blockIdx.x * 25 + 22]);
		}
	}
	/*for (int i = 0; i < 8; i++) {
		if (dctBlock[i] < 0.0001)
			dctBlock[i] = 0;
	}*/
	double mean = 0.0, std_gauss = 0.0;
	if (x == 0) {
		for (int i = 0; i < 8; i++) {
			mean += dctBlock[i];
			/*if (y == 1) {
				printf("%f\n", dctBlock[i]);
			}*/
		}
		/*if (y == 1) {
			printf("\n");
		}*/
		mean /= 8.0;
		/*if (fabsf(mean) < 0.0001) {
			ori_rho[y] = 0;
			return;
		}*/
		for (int i = 0; i < 8; i++) {
			double temp = dctBlock[i] - mean;
			std_gauss += temp * temp;
		}
		std_gauss = sqrt(std_gauss / 7.0);
		ori_rho[y] = std_gauss / (mean + 0.0000001);
		/*if (y == 7155) {
			printf("mean = %0.20f, std_gauss = %0.20f\nori[i] = %0.20f\n", mean, std_gauss, std_gauss / (mean + 0.00000001));
		}*/
	}
}

// Increase the number of warps to 4, threads = 128
__global__ void oriented_dct_rho2(float const * d_dctImg, float * ori_rho, int orient) {
	__shared__ float dctBlock[32*8];
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 8;
	int const sblkIdx = (threadIdx.x / 32) * 32;
	int const gblkIdx = (y + threadIdx.x / 32) * 32;
	int const ori[3] = {
		2042757120,	// 0b01111001110000100000000000000000
		34369920,	// 0b00000010000011000111000110000000
		70356480	// 0b00000100001100011000111000000000
	};
	if (x < 8) {
		if (orient == 1) {
			int inter_idx = (x + 1) / 5 + (x + 1) / 8;
			dctBlock[sblkIdx + x] = fabs(d_dctImg[gblkIdx + x + 1 + 5 * inter_idx - (x + 1) / 5 * 3 - (x + 1) / 8]);
		}
		else if (orient == 2) {
			int row = (x + 1) - x / 2 - x / 5 + x / 6 - x / 7;
			dctBlock[sblkIdx + x] = fabsf(d_dctImg[gblkIdx + row * 5 + x + 1 - x / 3 * 2]);
		}
		else if (orient == 3) {
			int const col = (x + 1) / 5 + (x + 1) / 8;
			dctBlock[sblkIdx + x] = fabsf(d_dctImg[gblkIdx + (x + 1) * 5 - 14 * col + (x + 1) / 8 * 10]);
		}
		float mean = dctBlock[sblkIdx + 0] + dctBlock[sblkIdx + 1] + dctBlock[sblkIdx + 2] + dctBlock[sblkIdx + 3] + \
			dctBlock[sblkIdx + 4] + dctBlock[sblkIdx + 5] + dctBlock[sblkIdx + 6] + dctBlock[sblkIdx + 7];
		mean /= 8;
		dctBlock[sblkIdx + x] -= mean;
		float std_gauss = dctBlock[sblkIdx + 0] * dctBlock[sblkIdx + 0] + dctBlock[sblkIdx + 1] * dctBlock[sblkIdx + 1] + dctBlock[sblkIdx + 2] * dctBlock[sblkIdx + 2] + \
			dctBlock[sblkIdx + 3] * dctBlock[sblkIdx + 3] + dctBlock[sblkIdx + 4] * dctBlock[sblkIdx + 4] + dctBlock[sblkIdx + 5] * dctBlock[sblkIdx + 5] + \
			dctBlock[sblkIdx + 6] * dctBlock[sblkIdx + 6] + dctBlock[sblkIdx + 7] * dctBlock[sblkIdx + 7];
		std_gauss = sqrtf(std_gauss / 7);
		ori_rho[gblkIdx / 32] = std_gauss / (mean + 0.0000001f);
	}
}

// Increase the number of warps to 4, threads = 128
__global__ void oriented_dct_rho3(float const * d_dctImg, float * ori_rho, int orient) {
	__shared__ float dctBlock[32 * 8];
	//dctBlock[threadIdx.x] = 0;
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 8;
	int const sblkIdx = (threadIdx.x / 32);
	int const gblkIdx = (y + threadIdx.x / 32) * 25;
	int  const ori = (orient - 1) * 32;

	/*__shared__*/ bool const orient_mat[96] = {
		//orient 1
		0, 1, 1, 1, 1,
		0, 0, 1, 1, 1,
		0, 0, 0, 0, 1,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, // extras for alignment to 32
		0, 0,

		//orient 2
		0, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 1, 0,
		0, 0, 1, 1, 1,
		0, 0, 0, 1, 1,
		0, 0, 0, 0, 0, // extras for alignment to 32
		0, 0,

		//orient 3
		0, 0, 0, 0, 0,
		1, 0, 0, 0, 0,
		1, 1, 0, 0, 0,
		1, 1, 0, 0, 0,
		1, 1, 1, 0, 0,
		0, 0, 0, 0, 0, // extras for alignment to 32
		0, 0,
	};
	dctBlock[threadIdx.x] = fabsf(d_dctImg[gblkIdx + x]);
	if (orient_mat[ori + x] == 0) {
		dctBlock[sblkIdx * 32 + x] = 0;
	}
	float mean = 0, std_gauss = 0;
	for (int i = 1; i < 25; i++) {
		mean += dctBlock[sblkIdx * 32 + i];
	}
	mean /= 8.0f;
	dctBlock[threadIdx.x] -= mean;
	for (int i = 1; i < 25; i++) {
		if (orient_mat[ori + i]) {
			std_gauss += dctBlock[sblkIdx * 32 + i] * dctBlock[sblkIdx * 32 + i];
		}
	}
	std_gauss = sqrtf(std_gauss / 7.0f);
	//if (x < 8) {
	/*if (orient == 1) {
	int inter_idx = (x + 1) / 5 + (x + 1) / 8;
	dctBlock[sblkIdx + x] = fabs(d_dctImg[gblkIdx + x + 1 + 5 * inter_idx - (x + 1) / 5 * 3 - (x + 1) / 8]);
	}
	else if (orient == 2) {
	int row = (x + 1) - x / 2 - x / 5 + x / 6 - x / 7;
	dctBlock[sblkIdx + x] = fabsf(d_dctImg[gblkIdx + row * 5 + x + 1 - x / 3 * 2]);
	}
	else if (orient == 3) {
	int const col = (x + 1) / 5 + (x + 1) / 8;
	dctBlock[sblkIdx + x] = fabsf(d_dctImg[gblkIdx + (x + 1) * 5 - 14 * col + (x + 1) / 8 * 10]);
	}
	double mean = dctBlock[sblkIdx + 0] + dctBlock[sblkIdx + 1] + dctBlock[sblkIdx + 2] + dctBlock[sblkIdx + 3] + \
	dctBlock[sblkIdx + 4] + dctBlock[sblkIdx + 5] + dctBlock[sblkIdx + 6] + dctBlock[sblkIdx + 7];
	mean /= 8;
	dctBlock[sblkIdx + x] -= mean;
	double std_gauss = dctBlock[sblkIdx + 0] * dctBlock[sblkIdx + 0] + dctBlock[sblkIdx + 1] * dctBlock[sblkIdx + 1] + dctBlock[sblkIdx + 2] * dctBlock[sblkIdx + 2] + \
	dctBlock[sblkIdx + 3] * dctBlock[sblkIdx + 3] + dctBlock[sblkIdx + 4] * dctBlock[sblkIdx + 4] + dctBlock[sblkIdx + 5] * dctBlock[sblkIdx + 5] + \
	dctBlock[sblkIdx + 6] * dctBlock[sblkIdx + 6] + dctBlock[sblkIdx + 7] * dctBlock[sblkIdx + 7];
	std_gauss = sqrtf(std_gauss / 7);*/
	ori_rho[gblkIdx / 25] = std_gauss / (mean + 0.0000001f);
	//}
}

__global__ void oriented_dct_final(const float * ori1_rho, const float * ori2_rho, const float * ori3_rho, float * ori_rho) {
	//plan grids = (512/3 + 1)^2, threads = 1
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	
	float num[3];
	num[1] = ori1_rho[y];
	num[2] = ori2_rho[y];
	num[0] = ori3_rho[y];
	double mean = 0, variance = 0;
	for (int i = 0; i < 3; i++) {
		mean += num[i];
	}
	mean /= 3.0;
	//const double variance = ((num[1] - mean) * (num[1] - mean) + (num[2] - mean) * (num[2] - mean) + (num[0] - mean) * (num[0] - mean)) / 2;
	for (int i = 0; i < 3; i++) {
		double temp = num[i] - mean;
		variance += temp * temp;
	}
	variance /= 2.0;

	ori_rho[y] = variance;
	/*if (y == 7155) {
		printf("ori1 = %0.20f\nori2 = %0.20f\nori3 = %0.20f\nori = %0.20f\n", ori1_rho[y], ori2_rho[y], ori3_rho[y], ori_rho[y]);
	}*/
}

// For more warps
__global__ void oriented_dct_final2(const float * ori1_rho, const float * ori2_rho, const float * ori3_rho, float * ori_rho) {
	//plan grids = (512/3 + 1)^2, threads = 1
	int const x = threadIdx.x;
	int const y = blockIdx.x * 512;

	float num[3];
	num[0] = ori1_rho[y + x];
	num[1] = ori2_rho[y + x];
	num[2] = ori3_rho[y + x];
	float mean = (num[0] + num[1] + num[2])/3.0;
	float variance = ((num[0] - mean)*(num[0] - mean) + (num[1] - mean)*(num[1] - mean) + (num[2] - mean)*(num[2] - mean)) / 2.0;

	ori_rho[y + x] = variance;
}

__global__ void subband_energy(const float * d_dctImg, float * freq_bands) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;

	__shared__ float dctBlock[25];
	__shared__ double inter[3];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();
	if (x == 0) {
		//const float num1 = dctBlock[1], num2 = dctBlock[2], num3 = dctBlock[5],
		//	num4 = dctBlock[6], num5 = dctBlock[10];
		const double mean = ((double)dctBlock[1] + dctBlock[2] + dctBlock[5] + dctBlock[6] + dctBlock[10]) / 5.0;
		inter[0] = ((dctBlock[1] - mean) * (dctBlock[1] - mean) + (dctBlock[2] - mean) * (dctBlock[2] - mean) +
			(dctBlock[5] - mean) * (dctBlock[5] - mean) + (dctBlock[6] - mean) * (dctBlock[6] - mean) + (dctBlock[10] - mean) * (dctBlock[10] - mean)) / 4.0;
	}
	if (x == 1) {
		const float num1 = dctBlock[15], num2 = dctBlock[20], num3 = dctBlock[11],
			num4 = dctBlock[16], num5 = dctBlock[21], num6 = dctBlock[7], num7 = dctBlock[12], num8 = dctBlock[17], num9 = dctBlock[3],
			num10 = dctBlock[8], num11 = dctBlock[13], num12 = dctBlock[4], num13 = dctBlock[9];
		const double mean = ((double)num1 + num2 + num3 + num4 + num5 + num6 + num7 + num8 + num9 + num10 + num11 + num12 + num13) / 13.0;
		inter[1] = ((num1 - mean) * (num1 - mean) + (num2 - mean) * (num2 - mean) +
			(num3 - mean) * (num3 - mean) + (num4 - mean) * (num4 - mean) + (num5 - mean) * (num5 - mean) +
			(num6 - mean) * (num6 - mean) + (num7 - mean) * (num7 - mean) +
			(num8 - mean) * (num8 - mean) + (num9 - mean) * (num9 - mean) + (num10 - mean) * (num10 - mean) +
			(num11 - mean) * (num11 - mean) + (num12 - mean) * (num12 - mean) + (num13 - mean) * (num13 - mean)) / 12.0;
	}
	if (x == 2) {
		//const float num1 = dctBlock[14], num2 = dctBlock[18], num3 = dctBlock[22],
		//	num4 = dctBlock[19], num5 = dctBlock[23], num6 = dctBlock[24];
		const double mean = ((double)dctBlock[14] + dctBlock[18] + dctBlock[22] + dctBlock[19] + dctBlock[23] + dctBlock[24]) / 6.0;
		inter[2] = ((dctBlock[14] - mean) * (dctBlock[14] - mean) + (dctBlock[18] - mean) * (dctBlock[18] - mean) +
			(dctBlock[22] - mean) * (dctBlock[22] - mean) + (dctBlock[19] - mean) * (dctBlock[19] - mean) + 
			(dctBlock[23] - mean) * (dctBlock[23] - mean) + (dctBlock[24] - mean) * (dctBlock[24] - mean)) / 5.0;
	}
	__syncthreads();
	if (x == 0) {
		//const double var_band1 = dctBlock[25], var_band2 = dctBlock[26], var_band3 = dctBlock[27];
		const double r1 = fabsf(inter[2] - (inter[0] + inter[1]) / 2.0) / (inter[2] + (inter[0] + inter[1]) / 2.0 + 0.00000001);
		const double r2 = fabsf(inter[1] - inter[0]) / (inter[2] + inter[0] + 0.00000001);
		//const float r1 = fabsf(var_band3 - (var_band1 + var_band2) / 2.0) / (var_band3 + (var_band1 + var_band2) / 2.0 + 0.00000001);
		//const float r2 = fabsf(var_band2 - var_band1) / (var_band3 + var_band1 + 0.00000001);
		/*if (var_band3 + var_band1 < 0.0001) {
			freq_bands[y] = 0;
			return;
		}*/
		freq_bands[y] = (r1 + r2) / 2.0;
	}
}

// Higher number of warps
__global__ void subband_energy2(const float * d_dctImg, float * freq_bands) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 4;
	int const sblkIdx = threadIdx.x / 32;
	int const gblkIdx = (y + sblkIdx) * 32;

	__shared__ float dctBlock[32*4];
	__shared__ float inter[3*4];
	dctBlock[threadIdx.x] = d_dctImg[gblkIdx + x];
	//__syncthreads();
	//if (x == 0) {
		const float mean1 = (dctBlock[sblkIdx * 32 + 1] + dctBlock[sblkIdx * 32 + 2] + dctBlock[sblkIdx * 32 + 5] + \
			dctBlock[sblkIdx * 32 + 6] + dctBlock[sblkIdx * 32 + 10]) / 5.0f;
		/*dctBlock[sblkIdx * 32 + x] -= mean;
		inter[sblkIdx * 3 + 0] = ((dctBlock[sblkIdx * 32 + 1]) * (dctBlock[sblkIdx * 32 + 1]) + (dctBlock[sblkIdx * 32 + 2]) * (dctBlock[sblkIdx * 32 + 2]) +
			(dctBlock[sblkIdx * 32 + 5]) * (dctBlock[sblkIdx * 32 + 5]) + (dctBlock[sblkIdx * 32 + 6]) * (dctBlock[sblkIdx * 32 + 6]) + \
			(dctBlock[sblkIdx * 32 + 10]) * (dctBlock[sblkIdx * 32 + 10])) / 4.0;
		dctBlock[sblkIdx * 32 + x] += mean;*/
		inter[sblkIdx * 3 + 0] = ((dctBlock[sblkIdx * 32 + 1] - mean1) * (dctBlock[sblkIdx * 32 + 1] - mean1) + (dctBlock[sblkIdx * 32 + 2] - mean1) * (dctBlock[sblkIdx * 32 + 2] - mean1) +
			(dctBlock[sblkIdx * 32 + 5] - mean1) * (dctBlock[sblkIdx * 32 + 5] - mean1) + (dctBlock[sblkIdx * 32 + 6] - mean1) * (dctBlock[sblkIdx * 32 + 6] - mean1) + \
			(dctBlock[sblkIdx * 32 + 10] - mean1) * (dctBlock[sblkIdx * 32 + 10] - mean1)) / 4.0f;
	//}
	//if (x == 1) {
		/*const float num1 = dctBlock[sblkIdx * 32 + 15], num2 = dctBlock[sblkIdx * 32 + 20], num3 = dctBlock[sblkIdx * 32 + 11], \
			num4 = dctBlock[sblkIdx * 32 + 16], num5 = dctBlock[sblkIdx * 32 + 21], num6 = dctBlock[sblkIdx * 32 + 7], num7 = dctBlock[sblkIdx * 32 + 12], \
			num8 = dctBlock[sblkIdx * 32 + 17], num9 = dctBlock[sblkIdx * 32 + 3], num10 = dctBlock[sblkIdx * 32 + 8], num11 = dctBlock[sblkIdx * 32 + 13], \
			num12 = dctBlock[sblkIdx * 32 + 4], num13 = dctBlock[sblkIdx * 32 + 9];*/
		/*const double*/ const float mean2 = (dctBlock[sblkIdx * 32 + 15] + dctBlock[sblkIdx * 32 + 20] + dctBlock[sblkIdx * 32 + 11] + \
			dctBlock[sblkIdx * 32 + 16] + dctBlock[sblkIdx * 32 + 21] + dctBlock[sblkIdx * 32 + 7] + dctBlock[sblkIdx * 32 + 12] + \
			dctBlock[sblkIdx * 32 + 17] + dctBlock[sblkIdx * 32 + 3] + dctBlock[sblkIdx * 32 + 8] + dctBlock[sblkIdx * 32 + 13] + \
			dctBlock[sblkIdx * 32 + 4] + dctBlock[sblkIdx * 32 + 9]) / 13.0f;
		/*dctBlock[sblkIdx * 32 + x] -= mean;
		inter[sblkIdx * 3 + 1] = ((dctBlock[sblkIdx * 32 + 15]) * (dctBlock[sblkIdx * 32 + 15]) + (dctBlock[sblkIdx * 32 + 20]) * (dctBlock[sblkIdx * 32 + 20]) +
			(dctBlock[sblkIdx * 32 + 11]) * (dctBlock[sblkIdx * 32 + 11]) + (dctBlock[sblkIdx * 32 + 16]) * (dctBlock[sblkIdx * 32 + 16]) + \
			(dctBlock[sblkIdx * 32 + 21]) * (dctBlock[sblkIdx * 32 + 21]) +	(dctBlock[sblkIdx * 32 + 7]) * (dctBlock[sblkIdx * 32 + 7]) + \
			(dctBlock[sblkIdx * 32 + 12]) * (dctBlock[sblkIdx * 32 + 12]) + (dctBlock[sblkIdx * 32 + 17]) * (dctBlock[sblkIdx * 32 + 17]) + \
			(dctBlock[sblkIdx * 32 + 3]) * (dctBlock[sblkIdx * 32 + 3]) + (dctBlock[sblkIdx * 32 + 8]) * (dctBlock[sblkIdx * 32 + 8]) + \
			(dctBlock[sblkIdx * 32 + 13]) * (dctBlock[sblkIdx * 32 + 13]) + (dctBlock[sblkIdx * 32 + 4]) * (dctBlock[sblkIdx * 32 + 4]) + \
			(dctBlock[sblkIdx * 32 + 9]) * (dctBlock[sblkIdx * 32 + 9])) / 12.0;
		dctBlock[sblkIdx * 32 + x] += mean;*/
		inter[sblkIdx * 3 + 1] = ((dctBlock[sblkIdx * 32 + 15] - mean2) * (dctBlock[sblkIdx * 32 + 15] - mean2) + (dctBlock[sblkIdx * 32 + 20] - mean2) * \
			(dctBlock[sblkIdx * 32 + 20] - mean2) + (dctBlock[sblkIdx * 32 + 11] - mean2) * (dctBlock[sblkIdx * 32 + 11] - mean2) + (dctBlock[sblkIdx * 32 + 16] - mean2) * \
			(dctBlock[sblkIdx * 32 + 16] - mean2) + (dctBlock[sblkIdx * 32 + 21] - mean2) * (dctBlock[sblkIdx * 32 + 21] - mean2) + (dctBlock[sblkIdx * 32 + 7] - mean2) * \
			(dctBlock[sblkIdx * 32 + 7] - mean2) + (dctBlock[sblkIdx * 32 + 12] - mean2) * (dctBlock[sblkIdx * 32 + 12] - mean2) + (dctBlock[sblkIdx * 32 + 17] - mean2) * \
			(dctBlock[sblkIdx * 32 + 17] - mean2) + (dctBlock[sblkIdx * 32 + 3] - mean2) * (dctBlock[sblkIdx * 32 + 3] - mean2) + (dctBlock[sblkIdx * 32 + 8] - mean2) * \
			(dctBlock[sblkIdx * 32 + 8] - mean2) + (dctBlock[sblkIdx * 32 + 13] - mean2) * (dctBlock[sblkIdx * 32 + 13] - mean2) + (dctBlock[sblkIdx * 32 + 4] - mean2) * \
			(dctBlock[sblkIdx * 32 + 4] - mean2) + (dctBlock[sblkIdx * 32 + 9] - mean2) * (dctBlock[sblkIdx * 32 + 9] - mean2)) / 12.0f;
	//}
	//if (x == 2) {
		const float mean3 = (dctBlock[sblkIdx * 32 + 14] + dctBlock[sblkIdx * 32 + 18] + dctBlock[sblkIdx * 32 + 22] + dctBlock[sblkIdx * 32 + 19] + \
			dctBlock[sblkIdx * 32 + 23] + dctBlock[sblkIdx * 32 + 24]) / 6.0f;
		/*dctBlock[sblkIdx * 32 + x] -= mean;
		inter[sblkIdx * 3 + 2] = ((dctBlock[sblkIdx * 32 + 14]) * (dctBlock[sblkIdx * 32 + 14]) + (dctBlock[sblkIdx * 32 + 18]) * (dctBlock[sblkIdx * 32 + 18]) +
			(dctBlock[sblkIdx * 32 + 22]) * (dctBlock[sblkIdx * 32 + 22]) + (dctBlock[sblkIdx * 32 + 19]) * (dctBlock[sblkIdx * 32 + 19]) +
			(dctBlock[sblkIdx * 32 + 23]) * (dctBlock[sblkIdx * 32 + 23]) + (dctBlock[sblkIdx * 32 + 24]) * (dctBlock[sblkIdx * 32 + 24])) / 5.0;
		//dctBlock[sblkIdx * 32 + x] += mean;
		const double mean = (dctBlock[sblkIdx * 32 + 14] + dctBlock[sblkIdx * 32 + 18] + dctBlock[sblkIdx * 32 + 22] + dctBlock[sblkIdx * 32 + 19] + \
			dctBlock[sblkIdx * 32 + 23] + dctBlock[sblkIdx * 32 + 24]) / 6.0;*/
		inter[sblkIdx * 3 + 2] = ((dctBlock[sblkIdx * 32 + 14] - mean3) * (dctBlock[sblkIdx * 32 + 14] - mean3) + (dctBlock[sblkIdx * 32 + 18] - mean3) * \
			(dctBlock[sblkIdx * 32 + 18] - mean3) + (dctBlock[sblkIdx * 32 + 22] - mean3) * (dctBlock[sblkIdx * 32 + 22] - mean3) + (dctBlock[sblkIdx * 32 + 19] - mean3) * \
			(dctBlock[sblkIdx * 32 + 19] - mean3) + (dctBlock[sblkIdx * 32 + 23] - mean3) * (dctBlock[sblkIdx * 32 + 23] - mean3) + (dctBlock[sblkIdx * 32 + 24] - mean3) * \
			(dctBlock[sblkIdx * 32 + 24] - mean3)) / 5.0f;
	//}
	//__syncthreads();
	//if (x == 0) {
		const float r1 = fabs(inter[sblkIdx * 3 + 2] - (inter[sblkIdx * 3 + 0] + inter[sblkIdx * 3 + 1]) / 2.0f) / \
			(inter[sblkIdx * 3 + 2] + (inter[sblkIdx * 3 + 0] + inter[sblkIdx * 3 + 1]) / 2.0f + 0.00000001f);
		const float r2 = fabs(inter[sblkIdx * 3 + 1] - inter[sblkIdx * 3 + 0]) / (inter[sblkIdx * 3 + 2] + inter[sblkIdx * 3 + 0] + 0.00000001f);
		freq_bands[gblkIdx / 32] = (r1 + r2) / 2.0f;
	//}
	/*if (gblkIdx + x == 200) {
		printf("inter[0] = %f\ninter[1] = %f\ninter[2] = %f\n", inter[sblkIdx * 3], inter[sblkIdx + 1], inter[sblkIdx + 2]);
	}*/
}

// Specialized warps for each subband does not do well :(
__global__ void subband_energy3(const float * d_dctImg, float * freq_bands) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x % 32;
	int const y = blockIdx.x * 4;
	int const sblkIdx = threadIdx.x / 96;
	int const gblkIdx = (y + sblkIdx) * 25;

	__shared__ float dctBlock[32 * 4];
	__shared__ float inter[3 * 4];
	dctBlock[sblkIdx * 32 + x] = d_dctImg[gblkIdx + x];
	//__syncthreads();
	float mean;
	if (threadIdx.x / 32 % 3 == 0) {
		mean = (dctBlock[sblkIdx * 32 + 1] + dctBlock[sblkIdx * 32 + 2] + dctBlock[sblkIdx * 32 + 5] + \
			dctBlock[sblkIdx * 32 + 6] + dctBlock[sblkIdx * 32 + 10]) / 5.0f;
		inter[sblkIdx * 3 + 0] = ((dctBlock[sblkIdx * 32 + 1] - mean) * (dctBlock[sblkIdx * 32 + 1] - mean) + (dctBlock[sblkIdx * 32 + 2] - mean) * (dctBlock[sblkIdx * 32 + 2] - mean) +
			(dctBlock[sblkIdx * 32 + 5] - mean) * (dctBlock[sblkIdx * 32 + 5] - mean) + (dctBlock[sblkIdx * 32 + 6] - mean) * (dctBlock[sblkIdx * 32 + 6] - mean) + \
			(dctBlock[sblkIdx * 32 + 10] - mean) * (dctBlock[sblkIdx * 32 + 10] - mean)) / 4.0f;
	}
	if (threadIdx.x / 32 % 3 == 1) {
		/*const double*/ mean = (dctBlock[sblkIdx * 32 + 15] + dctBlock[sblkIdx * 32 + 20] + dctBlock[sblkIdx * 32 + 11] + \
			dctBlock[sblkIdx * 32 + 16] + dctBlock[sblkIdx * 32 + 21] + dctBlock[sblkIdx * 32 + 7] + dctBlock[sblkIdx * 32 + 12] + \
			dctBlock[sblkIdx * 32 + 17] + dctBlock[sblkIdx * 32 + 3] + dctBlock[sblkIdx * 32 + 8] + dctBlock[sblkIdx * 32 + 13] + \
			dctBlock[sblkIdx * 32 + 4] + dctBlock[sblkIdx * 32 + 9]) / 13.0f;
		inter[sblkIdx * 3 + 1] = ((dctBlock[sblkIdx * 32 + 15] - mean) * (dctBlock[sblkIdx * 32 + 15] - mean) + (dctBlock[sblkIdx * 32 + 20] - mean) * (dctBlock[sblkIdx * 32 + 20] - mean) +
		(dctBlock[sblkIdx * 32 + 11] - mean) * (dctBlock[sblkIdx * 32 + 11] - mean) + (dctBlock[sblkIdx * 32 + 16] - mean) * (dctBlock[sblkIdx * 32 + 16] - mean) + (dctBlock[sblkIdx * 32 + 21] - mean) * (dctBlock[sblkIdx * 32 + 21] - mean) +
		(dctBlock[sblkIdx * 32 + 7] - mean) * (dctBlock[sblkIdx * 32 + 7] - mean) + (dctBlock[sblkIdx * 32 + 12] - mean) * (dctBlock[sblkIdx * 32 + 12] - mean) +
		(dctBlock[sblkIdx * 32 + 17] - mean) * (dctBlock[sblkIdx * 32 + 17] - mean) + (dctBlock[sblkIdx * 32 + 3] - mean) * (dctBlock[sblkIdx * 32 + 3] - mean) + \
		(dctBlock[sblkIdx * 32 + 8] - mean) * (dctBlock[sblkIdx * 32 + 8] - mean) +
		(dctBlock[sblkIdx * 32 + 13] - mean) * (dctBlock[sblkIdx * 32 + 13] - mean) + (dctBlock[sblkIdx * 32 + 4] - mean) * (dctBlock[sblkIdx * 32 + 4] - mean) +
		(dctBlock[sblkIdx * 32 + 9] - mean) * (dctBlock[sblkIdx * 32 + 9] - mean)) / 12.0f;
	}
	if (threadIdx.x / 32 % 3 == 2) {
		mean = (dctBlock[sblkIdx * 32 + 14] + dctBlock[sblkIdx * 32 + 18] + dctBlock[sblkIdx * 32 + 22] + dctBlock[sblkIdx * 32 + 19] + \
			dctBlock[sblkIdx * 32 + 23] + dctBlock[sblkIdx * 32 + 24]) / 6.0f;
		inter[sblkIdx * 3 + 2] = ((dctBlock[sblkIdx * 32 + 14] - mean) * (dctBlock[sblkIdx * 32 + 14] - mean) + (dctBlock[sblkIdx * 32 + 18] - mean) * (dctBlock[sblkIdx * 32 + 18] - mean) +
			(dctBlock[sblkIdx * 32 + 22] - mean) * (dctBlock[sblkIdx * 32 + 22] - mean) + (dctBlock[sblkIdx * 32 + 19] - mean) * (dctBlock[sblkIdx * 32 + 19] - mean) +
			(dctBlock[sblkIdx * 32 + 23] - mean) * (dctBlock[sblkIdx * 32 + 23] - mean) + (dctBlock[sblkIdx * 32 + 24] - mean) * (dctBlock[sblkIdx * 32 + 24] - mean)) / 5.0f;
	}
	__syncthreads();
	if (threadIdx.x / 32 % 3 == 1) {
		const float r1 = fabsf(inter[sblkIdx * 3 + 2] - (inter[sblkIdx * 3 + 0] + inter[sblkIdx * 3 + 1]) / 2.0f) / \
			(inter[sblkIdx * 3 + 2] + (inter[sblkIdx * 3 + 0] + inter[sblkIdx * 3 + 1]) / 2.0f + 0.00000001f);
		const float r2 = fabsf(inter[sblkIdx * 3 + 1] - inter[sblkIdx * 3 + 0]) / (inter[sblkIdx * 3 + 2] + inter[sblkIdx * 3 + 0] + 0.00000001f);
		freq_bands[gblkIdx / 25] = (r1 + r2) / 2.0f;
	}
}

__global__ void mean_100(float * d_input, float * d_mean_array, int num_elements) {
	// d_input is 171*171 array
	// for now let us have it structured as shared[171]
	//int  thread2;
	//double temp;
	__shared__ float sum[171];
	//_shared__ float sum[171];

	int x = threadIdx.x;
	int y = blockIdx.x * blockDim.x;
	//int blockDi2 = (blockDim.x / 2);

	sum[x] = d_input[y + x];
	__syncthreads();

	for (int s = 128; s > 0; s >>= 1) {
		if (x < s) {
			int temp = x + s;
			if (temp<blockDim.x) sum[x] += sum[x + s];
		}
		__syncthreads();
	}
	if (x == 0) {
		d_mean_array[blockIdx.x] = sum[0] / num_elements;
	}
}

__global__ void convolveRow(const float * d_input, const int size, float * d_output) {
	// Takes original image as input of dimensions size x size
	const int x = threadIdx.x;
	const int y = blockIdx.x;
	const int offset = -1;
	const int num_taps = 3;
	const double h[3] = { 0.106506978919200, 0.786986042161605, 0.106506978919200 };
	double val = 0.0;
	/*
	int x_comp = x + offset;
	val += h[0] * ((x_comp<0 || x_comp >= size) ? 0 : d_input[y * size + x_comp]);
	x_comp++;
	val += h[1] * ((x_comp<0 || x_comp >= size) ? 0 : d_input[y * size + x_comp]);
	x_comp++;
	val += h[2] * ((x_comp<0 || x_comp >= size) ? 0 : d_input[y * size + x_comp]);
	*/
	for (int tap_idx = 0; tap_idx < num_taps; tap_idx++) {
		int x_comp = x + offset + tap_idx;
		val += h[tap_idx] * ((x_comp<0 || x_comp>=size) ? 0 : d_input[y * size + x_comp]);
		//val += ((x_comp<0 || x_comp >= size) ? 0 : (h[tap_idx] * d_input[y * size + x_comp]));
	}
	d_output[y * size + x] = val;
	__syncthreads();
}

__global__ void convolveCol(const float * d_input, const int size, float * d_output) {
	const int x = threadIdx.x;
	const int y = blockIdx.x;
	const int offset = -1;
	const int num_taps = 3;
	const double h[3] = { 0.106506978919200, 0.786986042161605, 0.106506978919200 };
	double val = 0.0;
	/*
	int y_comp = y + offset;
	val += h[0] * ((y_comp<0 || y_comp >= size) ? 0 : d_input[y_comp * size + x]);
	y_comp++;
	val += h[1] * ((y_comp<0 || y_comp >= size) ? 0 : d_input[y_comp * size + x]);
	y_comp++;
	val += h[2] * ((y_comp<0 || y_comp >= size) ? 0 : d_input[y_comp * size + x]);
	*/
	for (int tap_idx = 0; tap_idx < num_taps; tap_idx++) {
		int y_comp = y + offset + tap_idx;
		val += h[tap_idx] * ((y_comp<0 || y_comp >= size) ? 0 : d_input[y_comp * size + x]);
	//val += ((y_comp<0 || y_comp >= size) ? 0 : (h[tap_idx] * d_output[y_comp * size + x]));
	}
	__syncthreads();
	d_output[y * size + x] = val;
}

__global__ void downsample_by2(const float * d_image, int const size, float * d_image_by2) {
	const int pos = blockIdx.x * size / 2 + threadIdx.x;
	d_image_by2[pos] = d_image[1 + size + 2 * pos + blockIdx.x * size];
}
	

void linearize_and_cast_from_Mat_to_float(const cv::Mat &Mat_in, float *float_array)
{
	// Linearize image matrix => Convert from 2D unsigned char Mat into a 1D array of floats
	for (int row = 0; row < 512; row++)
		for (int col = 0; col < 512; col++)
			float_array[row * 512 + col] = static_cast<float>(Mat_in.at<unsigned char>(row, col));
}

void device_rst()
{
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void kernel_wrapper(const cv::Mat &Mat_in)
{
	/*
	// cuFFT settings for DCT
	cufftHandle p;
	int rank = 1;
	int lengthOfDFT = 10;
	int howmany = 5 * (512 / 3 + 1) * (512 / 3 + 1);
	int odist;
	int idist = odist = lengthOfDFT;
	int ostride;
	int istride = ostride = 1; // array is contiguous in memory
	cufftPlanMany(&p, rank, &lengthOfDFT, NULL, istride, idist, NULL, ostride,
		odist, CUFFT_C2C, howmany);
	*/
	// The section of code below checks for zero copy feature. This way a kernel can directly use host memory.
	/*cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (!prop.canMapHostMemory)
		exit(0);
	else
		std::cout << "canMapHostMemory\n";
	*/

	// Allocate HOST memory
	float* h_in;												cudaError_t status = cudaMallocHost(&h_in, 512 * 512 * sizeof(float));
	if (status != cudaSuccess) {
		std::cout << "Error allocating pinned host memory for imput image";
	}
	
	float* features;											status = cudaMallocHost(&features, 25 * sizeof(float));
	if (status != cudaSuccess) {
		std::cout << "Error allocating pinned host memory for features";
	}

	LARGE_INTEGER frequency;        // ticks per second
	LARGE_INTEGER t1, t2;           // ticks
	double elapsedTime;

	// get ticks per second
	QueryPerformanceFrequency(&frequency);
	// start timer
	QueryPerformanceCounter(&t1);

	// Linearize image and cast from 8UC1 Mat to float array
	linearize_and_cast_from_Mat_to_float(Mat_in, h_in);

	//const float* h_in = Mat_in.ptr<float>(0);

	//Device memory allocations
	float *d_in;												cudaMalloc((void **)&d_in, 512 * 512 * sizeof(float));

	cudaProfilerStart();
	// Copy data from HOST -> DEVICE
	cudaMemcpy(d_in, h_in, 512 * 512 * sizeof(float), cudaMemcpyHostToDevice);
	float * d_g_info;											cudaMalloc((void **)&d_g_info, 9971 * sizeof(float));
	float * d_r_info;											cudaMalloc((void **)&d_r_info, 9971 * sizeof(float));
	float *d_in_pad;											cudaMalloc((void **)&d_in_pad, 517 * 517 * sizeof(float));
	/*cufftComplex *d_rearr_in;									cudaMalloc((void **)&d_rearr_in, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	cufftComplex *d_dct_inter;									cudaMalloc((void **)&d_dct_inter, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	cufftComplex *d_dct_in;										cudaMalloc((void **)&d_dct_in, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	*/
	float * d_dctImg;											cudaMalloc((void **)&d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 32 * sizeof(float));
	float * d_coeff_freq_var_L1;								cudaMalloc((void **)&d_coeff_freq_var_L1, (512 / 3 + 1) * (512 / 3 + 1) * sizeof(float));
	float * d_ori1_rho_L1;										cudaMalloc((void **)&d_ori1_rho_L1, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_ori2_rho_L1;										cudaMalloc((void **)&d_ori2_rho_L1, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_ori3_rho_L1;										cudaMalloc((void **)&d_ori3_rho_L1, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_ori_rho_L1;										cudaMalloc((void **)&d_ori_rho_L1, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_freq_bands;										cudaMalloc((void **)&d_freq_bands, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_gama_L1;											cudaMalloc((void **)&d_gama_L1, (512 / 3 + 1) * (512 / 3 + 1)  * sizeof(float));
	float * d_mean_array;										cudaMalloc((void **)&d_mean_array, (512 / 3 + 1) * sizeof(float));

	cudaStream_t stream1;
	cudaStream_t stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	//cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	//cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	// copy gama vector
	cudaMemcpyAsync(d_g_info, bliinds_info::g_vector, 9971 * sizeof(float), cudaMemcpyHostToDevice, stream1);

	// copy rho vector
	cudaMemcpyAsync(d_r_info, bliinds_info::r_vector, 9971 * sizeof(float), cudaMemcpyHostToDevice, stream2);

	cudaMemcpyToSymbol(dc_r_info, bliinds_info::r_vector, 9971 * sizeof(float));

	// pad input image for DCT in blocks
	pad << <517, 517 >> >(d_in, 512, d_in_pad);

	// Total number of DCT blocks at current scale
	int square = (512 / 3 + 1) * (512 / 3 + 1);
	//CuFFT at 512x512
	/*rearrangeForCuFFT << <square, 25, 0, 0 >> >(d_in_pad, 512, d_rearr_in);
	
	cufftExecC2C(p, d_rearr_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25, 0, 0 >> >(d_dct_in, d_dct_in);
	cufftExecC2C(p, d_dct_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25, 0, 0 >> >(d_dct_in, d_dct_in);

	copyDCT << <square, 25 >> >(d_dct_in, d_dctImg);
	cudaDeviceSynchronize();
	*/
	//float * h_dctImg = (float*)malloc((512 / 3 + 1) * (512 / 3 + 1) * sizeof(float));
	//cudaMemcpy(h_dctImg, d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float), cudaMemcpyDeviceToHost);
	
	float* d_rearr_man;											cudaMalloc((void **)&d_rearr_man, 32 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(float));
	cudaMemset(d_rearr_man, 0, 32 * square * sizeof(float));
	cudaMemset(d_dctImg, 0, 32 * square * sizeof(float));
	double* d_dctmtx;											cudaMalloc((void **)&d_dctmtx, 32 * sizeof(double));
	cudaMemset(d_dctmtx, 0, 32 * sizeof(double));
	cudaMemcpy(d_dctmtx, dct2_55::dctmtx_5, 25 * sizeof(double), cudaMemcpyHostToDevice);
	//float* h_dctImg = (float *)malloc(32 * square*sizeof(float));
	
	/*for (int i = 517; i < 517+32; i++) {
		std::cout << h_dctImg[i] << "\t";
		if ((i + 1) % 5 == 0)
			std::cout << std::endl;
	}*/
	//rearrangeForDCTv2 << <square / 4 + 1, 128 >> >(d_in_pad, 512, d_rearr_man);
	/*cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR1: %s\n", cudaGetErrorString(error));
		exit(-1);
	}*/
	rearrangeAndDCT55 << <square / 8 + 1, 256 >> >(d_in_pad, 512, d_dctmtx, d_dctImg);
	/*cudaDeviceSynchronize();
	error = cudaGetLastError();
	//cudaMemcpy(h_dctImg, d_dctImg, 32 * square * sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "ERROR2: %s\n", cudaGetErrorString(error));
		exit(-1);
	}*/

//#ifdef EBM
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//#endif
	rho_dct2 << <square / 16 + 1, 512 >> >(d_dctImg, d_coeff_freq_var_L1);
	//thrust::device_ptr<float> keys(d_coeff_freq_var_L1);
	thrust::sort(thrust::device, d_coeff_freq_var_L1, d_coeff_freq_var_L1 + square);
	//thrust::host_vector<float> h_coeff_freq_L1(d_coeff_freq_var_L1, d_coeff_freq_var_L1 + square);
	int mean10_size = ceil((square) / 10.0);
	features[0] = thrust::reduce(thrust::device, d_coeff_freq_var_L1, d_coeff_freq_var_L1 + square) / square;
	features[1] = thrust::reduce(thrust::device, d_coeff_freq_var_L1 + square - mean10_size, d_coeff_freq_var_L1 + square) / mean10_size;

	gama_dct62 << <square / 16 + 1, 512 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L1);
	thrust::sort(thrust::device, d_gama_L1, d_gama_L1 + square);
	gama_dct6_3 << <square / 128 + 1, 128 >> >(d_gama_L1, d_g_info, d_r_info, d_gama_L1, square);
	features[2] = thrust::reduce(thrust::device, d_gama_L1, d_gama_L1 + square) / square;
	features[3] = thrust::reduce(thrust::device, d_gama_L1 + square - mean10_size, d_gama_L1 + square) / mean10_size;

	/*cudaMemcpy(h_dctImg, d_coeff_freq_var_L1, square * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile3("harbourJPGcoeff_freq_varL1GPU.txt");
	for (int j = 0; j < square; j++) {
		//for (int i = 0; i < 5; i++) {
		outfile3 << h_dctImg[j];
		//if ((i + 1) % 5 == 0){
		//}
		//}
		outfile3 << std::endl;
	}
	outfile3.close();
	*/

	/*std::cout << "square1 = " << square << std::endl;
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L1, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L1, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L1, 3);*/
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori1_rho_L1, 1);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori2_rho_L1, 2);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori3_rho_L1, 3);
	oriented_dct_final2 << <square / 512 + 1, 512, 0 >> >(d_ori1_rho_L1, d_ori2_rho_L1, d_ori3_rho_L1, d_ori_rho_L1);
	thrust::sort(thrust::device, d_ori_rho_L1, d_ori_rho_L1 + square);
	features[6] = thrust::reduce(thrust::device, d_ori_rho_L1, d_ori_rho_L1 + square) / square;
	features[7] = thrust::reduce(thrust::device, d_ori_rho_L1 + square - mean10_size, d_ori_rho_L1 + square) / mean10_size;
	//std::cout << "or_rho_dct done\n";

	subband_energy2 << <square / 4 + 1, 128 >> >(d_dctImg, d_freq_bands);
	thrust::sort(thrust::device, d_freq_bands, d_freq_bands + square);
	features[4] = thrust::reduce(thrust::device, d_freq_bands, d_freq_bands + square) / square;
	features[5] = thrust::reduce(thrust::device, d_freq_bands + square - mean10_size, d_freq_bands + square) / mean10_size;
	//std::cout << "subband done\n";

	//cudaMemcpy(h_dctImg, d_gama_L1, (512 / 3 + 1) * (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_ori1_rho_L1);
	cudaFree(d_ori2_rho_L1);
	cudaFree(d_ori3_rho_L1);
	cudaFree(d_gama_L1);
	cudaFree(d_coeff_freq_var_L1);
	cudaFree(d_freq_bands);
	cudaFree(d_ori_rho_L1);

	//----------------------Start Phase 2----------------------------------------------------------

	cudaDeviceSynchronize();
	square = (256 / 3 + 1) * (256 / 3 + 1);
	float *d_in_conv_inter_L2;									cudaMalloc((void **)&d_in_conv_inter_L2, 512 * 512 * sizeof(float));
	float *d_in_convolve_L2;									cudaMalloc((void **)&d_in_convolve_L2, 512 * 512 * sizeof(float));
	float *d_in_L2;												cudaMalloc((void **)&d_in_L2, 256 * 256 * sizeof(float));
	float *d_in_pad_L2;											cudaMalloc((void **)&d_in_pad_L2, 261 * 261 * sizeof(float));
	float *d_coeff_freq_var_L2;									cudaMalloc((void **)&d_coeff_freq_var_L2, square * sizeof(float));
	cufftComplex *d_rearr_in_L2;								cudaMalloc((void **)&d_rearr_in_L2, 50 * square * sizeof(cufftComplex));
	float * d_ori1_rho_L2;										cudaMalloc((void **)&d_ori1_rho_L2, square * sizeof(float));
	float * d_ori2_rho_L2;										cudaMalloc((void **)&d_ori2_rho_L2, square * sizeof(float));
	float * d_ori3_rho_L2;										cudaMalloc((void **)&d_ori3_rho_L2, square * sizeof(float));
	float * d_ori_rho_L2;										cudaMalloc((void **)&d_ori_rho_L2, square * sizeof(float));
	float * d_freq_bands_L2;									cudaMalloc((void **)&d_freq_bands_L2, square * sizeof(float));
	float * d_gama_L2;											cudaMalloc((void **)&d_gama_L2, square * sizeof(float));
	
	convolveRow << <512, 512 >> >(d_in, 512, d_in_conv_inter_L2);
	convolveCol << <512, 512 >> >(d_in_conv_inter_L2, 512, d_in_convolve_L2);
	cudaDeviceSynchronize();
	downsample_by2 << <256, 256 >> >(d_in_convolve_L2, 512, d_in_L2);
	pad << <261, 261 >> >(d_in_L2, 256, d_in_pad_L2);

	/*float * h_dctImg = (float*)malloc(square * sizeof(float));


	cudaMemcpy(h_dctImg, d_in_convolve_L2, square * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile3("convolve_L2GPU.txt");
	for (int j = 0; j < square; j++) {
		//for (int i = 0; i < 5; i++) {
		outfile3 << h_dctImg[j];
		//if ((i + 1) % 5 == 0){
		//}
		//}
		outfile3 << std::endl;
	}
	outfile3.close();
	
	cudaMemcpy(h_dctImg, d_in_L2, 256 * 256 * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile2("d_in_L2GPU.txt");
	for (int j = 0; j < 256 * 256; j++) {
		//for (int i = 0; i < 5; i++) {
		outfile2 << h_dctImg[j];
		//if ((i + 1) % 5 == 0){
		//}
		//}
		outfile2 << std::endl;
	}
	outfile2.close();
	*/
	/*howmany = 5 * square;
	cufftPlanMany(&p, rank, &lengthOfDFT, NULL, istride, idist, NULL, ostride,
		odist, CUFFT_C2C, howmany);

	cudaDeviceSynchronize();
	
	rearrangeForCuFFT << <square, 25 >> >(d_in_pad_L2, 256, d_rearr_in_L2);

	cufftExecC2C(p, d_rearr_in_L2, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25 >> >(d_dct_in, d_dct_in);
	cufftExecC2C(p, d_dct_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25 >> >(d_dct_in, d_dct_in);

	copyDCT << <square, 25 >> >(d_dct_in, d_dctImg);
	cudaDeviceSynchronize();
	*/
	//std::cout << "phase 1 done \n";
	//rearrangeForDCTv2 << <square / 4 + 1, 128 >> >(d_in_pad_L2, 256, d_rearr_man);
	rearrangeAndDCT55 << <square / 8 + 1, 256 >> >(d_in_pad_L2, 256, d_dctmtx, d_dctImg);

	//h_dctImg = (float*)malloc(25 * square * sizeof(float));
	//std::cout << "second dct\n";
	rho_dct2 << <square / 16 + 1, 512 >> >(d_dctImg, d_coeff_freq_var_L2);
	thrust::sort(thrust::device, d_coeff_freq_var_L2, d_coeff_freq_var_L2 + square);
	mean10_size = ceil((square) / 10.0);
	features[9] = thrust::reduce(thrust::device, d_coeff_freq_var_L2 + square - mean10_size, d_coeff_freq_var_L2 + square) / mean10_size;
	features[8] = thrust::reduce(thrust::device, d_coeff_freq_var_L2, d_coeff_freq_var_L2 + square) / square;

	gama_dct62 << <square / 16 + 1, 512 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L2);
	//gama_dct5 << <square, 1024 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L2);
	thrust::sort(thrust::device, d_gama_L2, d_gama_L2 + square);
	gama_dct6_3 << <square / 128 + 1, 128 >> >(d_gama_L2, d_g_info, d_r_info, d_gama_L2, square);
	features[11] = thrust::reduce(thrust::device, d_gama_L2 + square - mean10_size, d_gama_L2 + square) / mean10_size;
	features[10] = thrust::reduce(thrust::device, d_gama_L2, d_gama_L2 + square) / square;

	subband_energy2 << <square / 4 + 1, 128 >> >(d_dctImg, d_freq_bands_L2);
	thrust::sort(thrust::device, d_freq_bands_L2, d_freq_bands_L2 + square);
	features[13] = thrust::reduce(thrust::device, d_freq_bands_L2 + square - mean10_size, d_freq_bands_L2 + square) / mean10_size;
	features[12] = thrust::reduce(thrust::device, d_freq_bands_L2, d_freq_bands_L2 + square) / square;

	/*std::cout << "square2 = " << square << std::endl;
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L2, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L2, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L2, 3);*/
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori1_rho_L2, 1);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori2_rho_L2, 2);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori3_rho_L2, 3);
	oriented_dct_final2 << <square / 512 + 1, 512 >> >(d_ori1_rho_L2, d_ori2_rho_L2, d_ori3_rho_L2, d_ori_rho_L2);
	thrust::sort(thrust::device, d_ori_rho_L2, d_ori_rho_L2 + square);
	features[15] = thrust::reduce(thrust::device, d_ori_rho_L2 + square - mean10_size, d_ori_rho_L2 + square) / mean10_size;
	features[14] = thrust::reduce(thrust::device, d_ori_rho_L2, d_ori_rho_L2 + square) / square;
	
	/*float * h_dctImg = (float*)malloc(square * 25 * sizeof(float));
	cudaMemcpy(h_dctImg, d_dctImg, square * 25 * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile3("d_dctImg_L2_babyJPG.txt");
	for (int j = 0; j < square; j++) {
		for (int i = 0; i < 25; i++) {
			outfile3 << h_dctImg[j * 25 + i] << ",";
			if ((i + 1) % 5 == 0)
				outfile3 << std::endl;
		}
		outfile3 << std::endl;
	}
	outfile3.close();*/

	/*cudaMemcpy(h_dctImg, d_ori2_rho_L2, square * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile4("d_ori2_L2_babyJPG.txt");
	for (int j = 0; j < square; j++) {
		outfile4 << h_dctImg[j] << std::endl;
	}
	outfile4.close();

	cudaMemcpy(h_dctImg, d_ori3_rho_L2, square * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile5("d_ori3_L2_babyJPG.txt");
	for (int j = 0; j < square; j++) {
		outfile5 << h_dctImg[j] << std::endl;
	}
	outfile5.close();*/

	cudaFree(d_ori1_rho_L2);
	cudaFree(d_ori2_rho_L2);
	cudaFree(d_ori3_rho_L2);
	cudaFree(d_gama_L2);
	cudaFree(d_coeff_freq_var_L2);
	cudaFree(d_freq_bands_L2);
	cudaFree(d_ori_rho_L2);
	/*
	cudaFree(d_in_conv_inter_L2);
	cudaFree(d_in_pad_L2);
	cudaFree(d_in_conv_inter_L2);
	cudaFree(d_rearr_in_L2);
	*/

	//----------------------Start Phase 3----------------------------------------------------------

	cudaDeviceSynchronize();
	square = (128 / 3 + 1) * (128 / 3 + 1);
	float *d_in_conv_inter_L3;									cudaMalloc((void **)&d_in_conv_inter_L3, 256 * 256 * sizeof(float));
	float *d_in_convolve_L3;									cudaMalloc((void **)&d_in_convolve_L3, 256 * 256 * sizeof(float));
	float *d_in_L3;												cudaMalloc((void **)&d_in_L3, 128 * 128 * sizeof(float));
	float *d_in_pad_L3;											cudaMalloc((void **)&d_in_pad_L3, 133 * 133 * sizeof(float));
	float *d_coeff_freq_var_L3;									cudaMalloc((void **)&d_coeff_freq_var_L3, square * sizeof(float));
	cufftComplex *d_rearr_in_L3;								cudaMalloc((void **)&d_rearr_in_L3, 50 * square * sizeof(cufftComplex));
	float * d_ori1_rho_L3;										cudaMalloc((void **)&d_ori1_rho_L3, square * sizeof(float));
	float * d_ori2_rho_L3;										cudaMalloc((void **)&d_ori2_rho_L3, square * sizeof(float));
	float * d_ori3_rho_L3;										cudaMalloc((void **)&d_ori3_rho_L3, square * sizeof(float));
	float * d_ori_rho_L3;										cudaMalloc((void **)&d_ori_rho_L3, square * sizeof(float));
	float * d_freq_bands_L3;									cudaMalloc((void **)&d_freq_bands_L3, square * sizeof(float));
	float * d_gama_L3;											cudaMalloc((void **)&d_gama_L3, square * sizeof(float));

	convolveRow << <256, 256 >> >(d_in_L2, 256, d_in_conv_inter_L3);
	convolveCol << <256, 256 >> >(d_in_conv_inter_L3, 256, d_in_convolve_L3);
	cudaDeviceSynchronize();
	downsample_by2 << <128, 128 >> >(d_in_convolve_L3, 256, d_in_L3);
	pad << <133, 133 >> >(d_in_L3, 128, d_in_pad_L3);

	/*howmany = 5 * square;
	cufftPlanMany(&p, rank, &lengthOfDFT, NULL, istride, idist, NULL, ostride,
		odist, CUFFT_C2C, howmany);

	cudaDeviceSynchronize();
	rearrangeForCuFFT << <square, 25 >> >(d_in_pad_L3, 128, d_rearr_in_L3);

	cufftExecC2C(p, d_rearr_in_L3, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25 >> >(d_dct_in, d_dct_in);
	cufftExecC2C(p, d_dct_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25 >> >(d_dct_in, d_dct_in);

	copyDCT << <square, 25 >> >(d_dct_in, d_dctImg);
	cudaDeviceSynchronize();
	*/
	//rearrangeForDCTv2 << <square / 4 + 1, 128 >> >(d_in_pad_L3, 128, d_rearr_man);
	rearrangeAndDCT55 << <square / 8 + 1, 256 >> >(d_in_pad_L3, 128, d_dctmtx, d_dctImg);
	cudaFree(d_dctmtx);

	rho_dct2 << <square / 16 + 1, 512 >> >(d_dctImg, d_coeff_freq_var_L3);
	thrust::sort(thrust::device, d_coeff_freq_var_L3, d_coeff_freq_var_L3 + square);
	mean10_size = ceil((square) / 10.0);
	features[17] = thrust::reduce(thrust::device, d_coeff_freq_var_L3 + square - mean10_size, d_coeff_freq_var_L3 + square) / mean10_size;
	features[16] = thrust::reduce(thrust::device, d_coeff_freq_var_L3, d_coeff_freq_var_L3 + square) / square;

	gama_dct62 << <square / 16 + 1, 512 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L3);
	//gama_dct5 << <square, 1024 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L3);
	thrust::sort(thrust::device, d_gama_L3, d_gama_L3 + square);
	gama_dct6_3 << <square / 128 + 1, 128 >> >(d_gama_L3, d_g_info, d_r_info, d_gama_L3, square);
	features[19] = thrust::reduce(thrust::device, d_gama_L3 + square - mean10_size, d_gama_L3 + square) / mean10_size;
	features[18] = thrust::reduce(thrust::device, d_gama_L3, d_gama_L3 + square) / square;

	// square = 1849
	subband_energy2 << <square / 4 + 1, 128 >> >(d_dctImg, d_freq_bands_L3);
	thrust::sort(thrust::device, d_freq_bands_L3, d_freq_bands_L3 + square);
	features[21] = thrust::reduce(thrust::device, d_freq_bands_L3 + square - mean10_size, d_freq_bands_L3 + square) / mean10_size;
	features[20] = thrust::reduce(thrust::device, d_freq_bands_L3, d_freq_bands_L3 + square) / square;

	/*std::cout << "square3 = " << square << std::endl;
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L3, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L3, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L3, 3);*/
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori1_rho_L3, 1);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori2_rho_L3, 2);
	oriented_dct_rho2 << <square / 8 + 1, 256 >> >(d_dctImg, d_ori3_rho_L3, 3);
	oriented_dct_final2 << <square / 512 + 1, 512 >> >(d_ori1_rho_L3, d_ori2_rho_L3, d_ori3_rho_L3, d_ori_rho_L3);
	thrust::sort(thrust::device, d_ori_rho_L3, d_ori_rho_L3 + square);
	features[23] = thrust::reduce(thrust::device, d_ori_rho_L3 + square - mean10_size, d_ori_rho_L3 + square) / mean10_size;
	features[22] = thrust::reduce(thrust::device, d_ori_rho_L3, d_ori_rho_L3 + square) / square;

	//Print features
	/*
	std::cout << "coeff_freq_var_l1:" << features[0] << ", " << features[1] << std::endl;
	std::cout << "gama_dct_l1:" << features[2] << ", " << features[3] << std::endl;
	std::cout << "freq_bands:" << features[4] << ", " << features[5] << std::endl;
	std::cout << "ori_rho_l1:" << features[6] << ", " << features[7] << std::endl;
	std::cout << "coeff_freq_var_l2: " << features[8] << ", " << features[9] << std::endl;
	std::cout << "gama_l2: " << features[10] << ", " << features[11] << std::endl;
	std::cout << "freq_bands_l2: " << features[12] << ", " << features[13] << std::endl;
	std::cout << "ori_rho_l2: " << features[14] << ", " << features[15] << std::endl;
	std::cout << "coeff_freq_var_l3: " << features[16] << ", " << features[17] << std::endl;
	std::cout << "gama_l3: " << features[18] << ", " << features[19] << std::endl;
	std::cout << "freq_bands_l3: " << features[20] << ", " << features[21] << std::endl;
	std::cout << "ori_rho_l3: " << features[22] << ", " << features[23] << std::endl;
	
	printf("coeff_freq_var_l1: %0.15f, %0.15f\n", features[0], features[1]);
	printf("gama_dct_l1: %0.15f, %0.15f\n", features[2], features[3]);
	printf("freq_bands: %0.15f, %0.15f\n", features[4], features[5]);
	printf("ori_rho_l1: %0.15f, %0.15f\n", features[6], features[7]);
	printf("coeff_freq_var_l2: %0.15f, %0.15f\n", features[8], features[9]);
	printf("gama_l2: %0.15f, %0.15f\n", features[10], features[11]);
	printf("freq_bands_l2: %0.15f, %0.15f\n", features[12], features[13]);
	printf("ori_rho_l2: %0.15f, %0.15f\n", features[14], features[15]);
	printf("coeff_freq_var_l3: %0.15f, %0.15f\n", features[16], features[17]);
	printf("gama_l3: %0.15f, %0.15f\n", features[18], features[19]);
	printf("freq_bands_l3: %0.15f, %0.15f\n", features[20], features[21]);
	printf("ori_rho_l3: %0.15f, %0.15f\n", features[22], features[23]);
	*/
	cudaFree(d_ori1_rho_L3);
	cudaFree(d_ori2_rho_L3);
	cudaFree(d_ori3_rho_L3);
	cudaFree(d_gama_L3);
	cudaFree(d_coeff_freq_var_L3);
	cudaFree(d_freq_bands_L3);
	cudaFree(d_ori_rho_L3);
	cudaFree(d_in_conv_inter_L3);
	cudaFree(d_in_convolve_L3);
	cudaFree(d_in_L3);
	cudaFree(d_in_pad_L3);
	cudaFree(d_rearr_in_L3);
	
	cudaFree(d_in);

	/*// stop timer
	QueryPerformanceCounter(&t2);

	// compute and print the elapsed time in millisec
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	std::cout << elapsedTime << " ms.\n";
	*/
	
	float * sub_temp = (float *)malloc(25 * sizeof(float));
	for (int i = 0; i < 24; i++){
		sub_temp[i] = features[i] - bliinds_info::mu_vector[i];
	}
	float * mult_temp = (float *)malloc(25 * sizeof(float));
	float product, max = 0, max_k, b = 1.0168, gama = 0.4200;

	for (int k = 2; k <= 200; k++) {
		sub_temp[24] = k / 2.0 - bliinds_info::mu_vector[24];
		for (int i = 0; i < 25; i++) {
			mult_temp[i] = 0;
			for (int j = 0; j < 25; j++) {
				mult_temp[i] += sub_temp[j] * bliinds_info::sigma_inv_vector[i + 25 * j];
			}
		}
		product = 0;
		for (int i = 0; i < 25; i++){
			product += mult_temp[i] * sub_temp[i];
		}
		product = exp(-pow(b*product, gama));
		if (product > max) {
			max = product;
			max_k = k / 2.0 - 1;
		}
	}
	std::cout << "BLIINDS score: " << max_k << std::endl;
	// stop timer
	QueryPerformanceCounter(&t2);

	// compute and print the elapsed time in millisec
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	std::cout << elapsedTime << " ms.\n\n";
	cudaDeviceSynchronize();
	cudaProfilerStop();

}