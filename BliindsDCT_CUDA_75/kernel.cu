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

__global__ void transposeForCuFFT(cufftComplex const*read, cufftComplex *write) {
	// threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	float temp1;// temp2;
	temp1 = (read[y * 50 + x % 5 * 10 + x / 5].x * cos(PI * (x / 5) / 10.0) + read[y * 50 + x % 5 * 10 + x / 5].y * sin(PI * (x / 5) / 10.0)) / 2.0 * (x / 5 == 0 ? sqrt(0.2) : sqrt(0.4));
	//temp2 = (read[y * 50 + x % 5 * 10 + x / 5].y * cos(PI * (x / 5) / 10.0) - read[y * 50 + x % 5 * 10 + x / 5].x * sin(PI * (x / 5) / 10.0)) / 2.0 * (x % 5 == 0 ? sqrt(0.2) : sqrt(0.4));
	//__syncthreads();
	write[y * 50 + x / 5 * 10 + x % 5].x = temp1;
	write[y * 50 + x / 5 * 10 + x % 5].y = 0;
	write[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].x = temp1;
	write[y * 50 + (x / 5 + 1) * 10 - x % 5 - 1].y = 0;
}

__global__ void copyDCT(cufftComplex const*dct_img, float *dctImg) {
	//threads = 25, blocks = (512/3 +1)^2
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	//dctImg[y * 25 + x] = dct_img[y * 50 + x / 5 * 10 + x % 5].x; 
	dctImg[y * 25 + x] = (fabsf(dct_img[y * 50 + x / 5 * 10 + x % 5].x) > 0.0001 ? dct_img[y * 50 + x / 5 * 10 + x % 5].x : 0);
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
			/*if(y == 450) {
				printf("%0.20f\n", fabsf(dctBlock[i]));
			}
			*/
			mean_abs += fabsf(dctBlock[i]);
		}
		
		mean_abs = mean_abs / 24.0;
		for (int i = 1; i < 25; i++) {
			float temp = fabsf(dctBlock[i]) - mean_abs;
			std_gauss += temp * temp;
		}
		std_gauss = sqrtf(std_gauss / 23.0);
		coeff_freq_var[y] = std_gauss / (mean_abs + 0.0000001);
		/*if (y == 450) {
			printf("std_gauss: %0.20f, \tmean_abs: %0.20f, \tcoeff: %0.20f\n", std_gauss, mean_abs, coeff_freq_var[y]);
		}
		*/
	}
		
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

__global__ void gama_dct6_3(float * d_rho, float const * g_vector, float const * r_vector, float * d_gama, int max) {
	int const pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos < max) {
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
	}	
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
			dctBlock[0] = fabsf(d_dctImg[blockIdx.x * 25 + 1]);
			dctBlock[1] = fabsf(d_dctImg[blockIdx.x * 25 + 2]);
			dctBlock[2] = fabsf(d_dctImg[blockIdx.x * 25 + 7]);
			dctBlock[3] = fabsf(d_dctImg[blockIdx.x * 25 + 3]);
			dctBlock[4] = fabsf(d_dctImg[blockIdx.x * 25 + 8]);
			dctBlock[5] = fabsf(d_dctImg[blockIdx.x * 25 + 4]);
			dctBlock[6] = fabsf(d_dctImg[blockIdx.x * 25 + 9]);
			dctBlock[7] = fabsf(d_dctImg[blockIdx.x * 25 + 14]);
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
	float mean = 0.0, std_gauss = 0.0;
	if (x == 0) {
		for (int i = 0; i < 8; i++) {
			mean += dctBlock[i];
		}
		mean /= 8.0;
		for (int i = 0; i < 8; i++) {
			float temp = dctBlock[i] - mean;
			std_gauss += temp * temp;
		}
		std_gauss = sqrtf(std_gauss / 7.0);
		ori_rho[y] = std_gauss / (mean + 0.0000001);
	}
}

__global__ void oriented_dct_final(const float * ori1_rho, const float * ori2_rho, const float * ori3_rho, float * ori_rho) {
	//plan grids = (512/3 + 1)^2, threads = 1
	int const x = threadIdx.x;
	int const y = blockIdx.x;
	
	float num1 = ori1_rho[y], num2 = ori2_rho[y], num3 = ori3_rho[y];
	const float mean = (num1 + num2 + num3) / 3.0;
	const float variance = ((num1 - mean) * (num1 - mean) + (num2 - mean) * (num2 - mean) + (num3 - mean) * (num3 - mean)) / 2;

	ori_rho[y] = variance;
}

__global__ void subband_energy(const float * d_dctImg, float * freq_bands) {
	//plan grids = (512/3 + 1)^2, threads = 25
	int const x = threadIdx.x;
	int const y = blockIdx.x;

	__shared__ float dctBlock[28];
	dctBlock[x] = d_dctImg[y * 25 + x];
	__syncthreads();
	if (x == 0) {
		const float num1 = dctBlock[1], num2 = dctBlock[2], num3 = dctBlock[5],
			num4 = dctBlock[6], num5 = dctBlock[10];
		const float mean = (num1 + num2 + num3 + num4 + num5) / 5.0;
		dctBlock[25] = ((num1 - mean) * (num1 - mean) + (num2 - mean) * (num2 - mean) +
			(num3 - mean) * (num3 - mean) + (num4 - mean) * (num4 - mean) + (num5 - mean) * (num5 - mean)) / 4.0;
	}
	if (x == 1) {
		const float num1 = dctBlock[15], num2 = dctBlock[20], num3 = dctBlock[11],
			num4 = dctBlock[16], num5 = dctBlock[21], num6 = dctBlock[7], num7 = dctBlock[12], num8 = dctBlock[17], num9 = dctBlock[3],
			num10 = dctBlock[8], num11 = dctBlock[13], num12 = dctBlock[4], num13 = dctBlock[9];
		const float mean = (num1 + num2 + num3 + num4 + num5 + num6 + num7 + num8 + num9 + num10 + num11 + num12 + num13) / 13.0;
		dctBlock[26] = ((num1 - mean) * (num1 - mean) + (num2 - mean) * (num2 - mean) +
			(num3 - mean) * (num3 - mean) + (num4 - mean) * (num4 - mean) + (num5 - mean) * (num5 - mean) +
			(num6 - mean) * (num6 - mean) + (num7 - mean) * (num7 - mean) +
			(num8 - mean) * (num8 - mean) + (num9 - mean) * (num9 - mean) + (num10 - mean) * (num10 - mean) +
			(num11 - mean) * (num11 - mean) + (num12 - mean) * (num12 - mean) + (num13 - mean) * (num13 - mean)) / 12.0;
	}
	if (x == 2) {
		const float num1 = dctBlock[14], num2 = dctBlock[18], num3 = dctBlock[22],
			num4 = dctBlock[19], num5 = dctBlock[23], num6 = dctBlock[24];
		const float mean = (num1 + num2 + num3 + num4 + num5 + num6) / 6.0;
		dctBlock[27] = ((num1 - mean) * (num1 - mean) + (num2 - mean) * (num2 - mean) +
			(num3 - mean) * (num3 - mean) + (num4 - mean) * (num4 - mean) + (num5 - mean) * (num5 - mean) + (num6 - mean) * (num6 - mean)) / 5.0;
	}
	__syncthreads();
	if (x == 0) {
		const float var_band1 = dctBlock[25], var_band2 = dctBlock[26], var_band3 = dctBlock[27];
		const float r1 = fabsf(var_band3 - (var_band1 + var_band2) / 2.0) / (var_band3 + (var_band1 + var_band2) / 2.0 + 0.00000001);
		const float r2 = fabsf(var_band2 - var_band1) / (var_band3 + var_band1 + 0.00000001);
		freq_bands[y] = (r1 + r2) / 2.0;
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
	const float h[3] = { 0.106506978919200, 0.786986042161605, 0.106506978919200 };
	float val = 0.0;
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
	const float h[3] = { 0.106506978919200, 0.786986042161605, 0.106506978919200 };
	float val = 0.0;
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
	// cuFFT settings for DCT
	cufftHandle p;
	int rank = 1;
	int lengthOfDFT = 10;
	int howmany = 5 * (512 / 3 + 1) * (512 / 3 + 1);
	int odist;
	int idist = odist = lengthOfDFT;
	int ostride;
	int istride = ostride = 1; /* array is contiguous in memory */
	cufftPlanMany(&p, rank, &lengthOfDFT, NULL, istride, idist, NULL, ostride,
		odist, CUFFT_C2C, howmany);
	
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


	// Create R2C & C2R FFT plans
	//cufftHandle fftPlanFwd; cufftPlan2d(&fftPlanFwd, 512, 512, CUFFT_R2C);
	//cufftHandle fftPlanInv; cufftPlan2d(&fftPlanInv, 512, 512, CUFFT_C2R);

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
	cufftComplex *d_rearr_in;									cudaMalloc((void **)&d_rearr_in, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	cufftComplex *d_dct_inter;									cudaMalloc((void **)&d_dct_inter, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	cufftComplex *d_dct_in;										cudaMalloc((void **)&d_dct_in, 50 * (512 / 3 + 1) * (512 / 3 + 1) * sizeof(cufftComplex));
	float * d_dctImg;											cudaMalloc((void **)&d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float));
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

	int square = (512 / 3 + 1) * (512 / 3 + 1);
	rearrangeForCuFFT << <square, 25, 0, 0 >> >(d_in_pad, 512, d_rearr_in);
	
	cufftExecC2C(p, d_rearr_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25, 0, 0 >> >(d_dct_in, d_dct_in);
	cufftExecC2C(p, d_dct_in, d_dct_in, CUFFT_FORWARD);
	transposeForCuFFT << <square, 25, 0, 0 >> >(d_dct_in, d_dct_in);

	copyDCT << <square, 25 >> >(d_dct_in, d_dctImg);
	cudaDeviceSynchronize();

	float * h_dctImg = (float*)malloc((512 / 3 + 1) * (512 / 3 + 1) * sizeof(float));
	//cudaMemcpy(h_dctImg, d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float), cudaMemcpyDeviceToHost);


	rho_dct << <square, 25 >> >(d_dctImg, d_coeff_freq_var_L1);

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

	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L1, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L1, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L1, 3);

	oriented_dct_final << <square, 1, 0 >> >(d_ori1_rho_L1, d_ori2_rho_L1, d_ori3_rho_L1, d_ori_rho_L1);

	subband_energy << <square, 25 >> >(d_dctImg, d_freq_bands);

	gama_dct6 << <square, 25 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L1);
	//gama_dct5 << <square, 1024 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L1);

	//thrust::device_ptr<float> keys(d_coeff_freq_var_L1);
	thrust::sort(thrust::device, d_coeff_freq_var_L1, d_coeff_freq_var_L1 + square);
	int mean10_size = ceil((square) / 10.0);
	features[0] = thrust::reduce(thrust::device, d_coeff_freq_var_L1, d_coeff_freq_var_L1 + square) / square;
	features[1] = thrust::reduce(thrust::device, d_coeff_freq_var_L1 + square - mean10_size, d_coeff_freq_var_L1 + square) / mean10_size;

	//cudaMemcpy(h_dctImg, d_gama_L1, (512 / 3 + 1) * (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);

	//mean_100 << <171, 171 >> >(d_coeff_freq_var_L1, d_mean_array, 171);
	//mean_100 << <1, 171 >> >(d_mean_array, d_mean_array, 171);

	//float * h_preFeature_block = (float*)malloc((512 / 3 + 1) * sizeof(float));
	//cudaMemcpy(h_preFeature_block, d_mean_array, (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);

	//mean_100 << <171, 171 >> >(d_freq_bands, d_mean_array, 171);
	//mean_100 << <1, 171 >> >(d_mean_array, d_mean_array, 171);
	//cudaMemcpy(h_preFeature_block, d_mean_array, (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
	thrust::sort(thrust::device, d_freq_bands, d_freq_bands + square);
	features[4] = thrust::reduce(thrust::device, d_freq_bands, d_freq_bands + square) / square;
	features[5] = thrust::reduce(thrust::device, d_freq_bands + square - mean10_size, d_freq_bands + square) / mean10_size;

	//mean_100 << <171, 171 >> >(d_ori_rho_L1, d_mean_array, 171);
	//mean_100 << <1, 171 >> >(d_mean_array, d_mean_array, 171);
	//cudaMemcpy(h_preFeature_block, d_mean_array, (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
	thrust::sort(thrust::device, d_ori_rho_L1, d_ori_rho_L1 + square);
	features[6] = thrust::reduce(thrust::device, d_ori_rho_L1, d_ori_rho_L1 + square) / square;
	features[7] = thrust::reduce(thrust::device, d_ori_rho_L1 + square - mean10_size, d_ori_rho_L1 + square) / mean10_size;

	//mean_100 << <171, 171 >> >(d_gama_L1, d_mean_array, 171);
	//mean_100 << <1, 171 >> >(d_mean_array, d_mean_array, 171);
	//cudaMemcpy(h_preFeature_block, d_mean_array, (512 / 3 + 1) * sizeof(float), cudaMemcpyDeviceToHost);
	thrust::sort(thrust::device, d_gama_L1, d_gama_L1 + square);
	gama_dct6_3 << <square / 1024 + 1, 1024 >> >(d_gama_L1, d_g_info, d_r_info, d_gama_L1, square);
	features[2] = thrust::reduce(thrust::device, d_gama_L1, d_gama_L1 + square) / square;
	features[3] = thrust::reduce(thrust::device, d_gama_L1 + square - mean10_size, d_gama_L1 + square) / mean10_size;


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

	//h_dctImg = (float*)malloc(512 * 512 * sizeof(float));
	
	/*
	cudaMemcpy(h_dctImg, d_in_convolve_L2, 512 * 512 * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile3("convolve_L2GPU.txt");
	for (int j = 0; j < 512 * 512; j++) {
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
	howmany = 5 * square;
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

	//h_dctImg = (float*)malloc(25 * square * sizeof(float));

	rho_dct << <square, 25 >> >(d_dctImg, d_coeff_freq_var_L2);
	subband_energy << <square, 25 >> >(d_dctImg, d_freq_bands_L2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L2, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L2, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L2, 3);
	oriented_dct_final << <square, 1 >> >(d_ori1_rho_L2, d_ori2_rho_L2, d_ori3_rho_L2, d_ori_rho_L2);
	thrust::sort(thrust::device, d_coeff_freq_var_L2, d_coeff_freq_var_L2 + square);
	mean10_size = ceil((square) / 10.0);
	features[9] = thrust::reduce(thrust::device, d_coeff_freq_var_L2 + square - mean10_size, d_coeff_freq_var_L2 + square) / mean10_size;
	features[8] = thrust::reduce(thrust::device, d_coeff_freq_var_L2, d_coeff_freq_var_L2 + square) / square;

	//mean_100 << <86, 86 >> >(d_coeff_freq_var_L2, d_mean_array, 86);
	//mean_100 << <1, 86 >> >(d_mean_array, d_mean_array, 86);

	//cudaMemcpy(h_preFeature_block, d_mean_array, 86 * sizeof(float), cudaMemcpyDeviceToHost);

	thrust::sort(thrust::device, d_freq_bands_L2, d_freq_bands_L2 + square);
	features[13] = thrust::reduce(thrust::device, d_freq_bands_L2 + square - mean10_size, d_freq_bands_L2 + square) / mean10_size;
	features[12] = thrust::reduce(thrust::device, d_freq_bands_L2, d_freq_bands_L2 + square) / square;

	thrust::sort(thrust::device, d_ori_rho_L2, d_ori_rho_L2 + square);
	features[15] = thrust::reduce(thrust::device, d_ori_rho_L2 + square - mean10_size, d_ori_rho_L2 + square) / mean10_size;
	features[14] = thrust::reduce(thrust::device, d_ori_rho_L2, d_ori_rho_L2 + square) / square;

	gama_dct6 << <square, 25 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L2);
	//gama_dct5 << <square, 1024 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L2);
	thrust::sort(thrust::device, d_gama_L2, d_gama_L2 + square);
	gama_dct6_3 << <square / 1024 + 1, 1024 >> >(d_gama_L2, d_g_info, d_r_info, d_gama_L2, square);
	features[11] = thrust::reduce(thrust::device, d_gama_L2 + square - mean10_size, d_gama_L2 + square) / mean10_size;
	features[10] = thrust::reduce(thrust::device, d_gama_L2, d_gama_L2 + square) / square;

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

	howmany = 5 * square;
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

	rho_dct << <square, 25 >> >(d_dctImg, d_coeff_freq_var_L3);
	subband_energy << <square, 25 >> >(d_dctImg, d_freq_bands_L3);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori1_rho_L3, 1);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori2_rho_L3, 2);
	oriented_dct_rho << <square, 1 >> >(d_dctImg, d_ori3_rho_L3, 3);
	oriented_dct_final << <square, 1 >> >(d_ori1_rho_L3, d_ori2_rho_L3, d_ori3_rho_L3, d_ori_rho_L3);
	thrust::sort(thrust::device, d_coeff_freq_var_L3, d_coeff_freq_var_L3 + square);
	mean10_size = ceil((square) / 10.0);
	features[17] = thrust::reduce(thrust::device, d_coeff_freq_var_L3 + square - mean10_size, d_coeff_freq_var_L3 + square) / mean10_size;
	features[16] = thrust::reduce(thrust::device, d_coeff_freq_var_L3, d_coeff_freq_var_L3 + square) / square;

	thrust::sort(thrust::device, d_freq_bands_L3, d_freq_bands_L3 + square);
	features[21] = thrust::reduce(thrust::device, d_freq_bands_L3 + square - mean10_size, d_freq_bands_L3 + square) / mean10_size;
	features[20] = thrust::reduce(thrust::device, d_freq_bands_L3, d_freq_bands_L3 + square) / square;

	thrust::sort(thrust::device, d_ori_rho_L3, d_ori_rho_L3 + square);
	features[23] = thrust::reduce(thrust::device, d_ori_rho_L3 + square - mean10_size, d_ori_rho_L3 + square) / mean10_size;
	features[22] = thrust::reduce(thrust::device, d_ori_rho_L3, d_ori_rho_L3 + square) / square;

	gama_dct6 << <square, 25 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L3);
	//gama_dct5 << <square, 1024 >> >(d_dctImg, d_g_info, d_r_info, d_gama_L3);
	thrust::sort(thrust::device, d_gama_L3, d_gama_L3 + square);
	gama_dct6_3 << <square/1024 + 1, 1024 >> >(d_gama_L3, d_g_info, d_r_info, d_gama_L3, square);
	features[19] = thrust::reduce(thrust::device, d_gama_L3 + square - mean10_size, d_gama_L3 + square) / mean10_size;
	features[18] = thrust::reduce(thrust::device, d_gama_L3, d_gama_L3 + square) / square;

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
		//product = b * product;
		//product = -pow(product, gama);
		//product = exp(product);
		product = exp(-pow(b*product, gama));
		if (product > max) {
			max = product;
			max_k = k / 2.0 - 1;
		}
	}
	std::cout << std::endl << "BLIINDS score: " << max_k << std::endl <<std::endl;

	// stop timer
	QueryPerformanceCounter(&t2);

	// compute and print the elapsed time in millisec
	elapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
	std::cout << elapsedTime << " ms.\n";
	cudaDeviceSynchronize();
	cudaProfilerStop();
	
	/*int temp = 171;
	temp = (temp | (~temp));
	std::cout << "round up: " << temp << std::endl;
	*/
	/********Print results in file**********************
	****************************************************/
	/*
	cudaMemcpy(h_dctImg, d_dctImg, 25 * square * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile("dct_L2GPU.txt");
	for (int j = 0; j < 5 * square; j++) {
		for (int i = 0; i < 5; i++) {
			outfile << h_dctImg[j * 5 + i] << ", ";
			//if ((i + 1) % 5 == 0){
			//}
		}
			outfile << std::endl;
	}
	outfile.close();
	*/
	
	
	/*cudaMemcpy(h_dctImg, d_g_info, 9970 * sizeof(float), cudaMemcpyDeviceToHost);
	std::ofstream outfile2("g_infoGPU.txt");
	for (int j = 0; j < 9970; j++) {
		//for (int i = 0; i < 25; i++) {
		outfile2 << h_dctImg[j];
		//if ((i + 1) % 5 == 0){
		outfile2 << std::endl;
		//}
		//}
	}
	outfile2.close();
	/*****************************************************
	**/
	/*std::cout << "Here is the dctImg:\n";
	for (int j = 0; j < 171*171; j++) {
		for (int i = 0; i < 25; i++) {
			std::cout << h_dctImg[j * 25 + i] << "\t";
		}
		std::cout << "\n";
	}*/
	//std::cout << "Here is the rho_dct:\n";
	//for (int j = 0; j < 170; j++) {
	//	for (int i = 0; i < 170; i++) {
	//		std::cout << h_preFeature_block[j * 170 + i] << "\t";
	//	}
	//	std::cout << "\n";
	//}
	
	
	//========================================================================================
	//Test for correct DCT output
	/*
	float * h_dctImg = (float*)malloc((512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float));
	cudaDeviceSynchronize();
	rearrangeTest << <2, 25 >> >(d_rearr_in);
	cudaDeviceSynchronize();
	copyTest << <5, 20 >> >(d_rearr_in, d_dctImg);
	cudaDeviceSynchronize();
	cudaMemcpy(h_dctImg, d_dctImg, 4 * 25 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	std::cout << "Here is the input:\n";
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 50; i++) {
			std::cout << h_dctImg[j * 50 + i] << "\t";
		}
		std::cout << "\n";
	}
	cufftExecC2C(p, d_rearr_in, d_rearr_in, CUFFT_FORWARD);
	transposeForCuFFT << <(512 / 3 + 1) ^ 2, 25 >> >(d_rearr_in, d_dct_inter);

	copyDCT << <(512 / 3 + 1) ^ 2, 25 >> >(d_dct_inter, d_dctImg);
	cudaMemcpy(h_dctImg, d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	std::cout << "\nHere is after 1st dct:\n";
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 25; i++) {
			if (i % 5 == 0)
				std::cout << std::endl;
			std::cout << h_dctImg[j * 25 + i] << "\t";
		}
		//std::cout << "\n";
	}

	cufftExecC2C(p, d_dct_inter, d_dct_inter, CUFFT_FORWARD);
	transposeForCuFFT << <(512 / 3 + 1) ^ 2, 25 >> >(d_dct_inter, d_dct_in);

	copyDCT << <(512 / 3 + 1) ^ 2, 25 >> >(d_dct_in, d_dctImg);

	
	cudaMemcpy(h_dctImg, d_dctImg, (512 / 3 + 1) * (512 / 3 + 1) * 25 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	std::cout << "\nHere is the dct:\n";
	for (int j = 0; j < 2; j++) {
		for (int i = 0; i < 25; i++) {
			if (i % 5 == 0)
				std::cout << std::endl;
			std::cout << h_dctImg[j * 25 + i] << "\t";
		}
		//std::cout << "\n";
	}
	*/
	
}