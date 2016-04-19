//=============================================================================
#include "header.h"
//=============================================================================
int main()
{
	/*
	//images array
	std::string images[] = { "baby.bmp", "baby.BLR.bmp", "baby.DCQ.bmp", "baby.FLT.bmp", "baby.JP2.bmp", "baby.JPG.bmp", "baby.NOZ.bmp",
		"horse.bmp", "horse.BLR.bmp", "horse.DCQ.bmp", "horse.FLT.bmp", "horse.JP2.bmp", "horse.JPG.bmp", "horse.NOZ.bmp",
		"harbour.bmp", "harbour.BLR.bmp", "harbour.DCQ.bmp", "harbour.FLT.bmp", "harbour.JP2.bmp", "harbour.JPG.bmp", "harbour.NOZ.bmp" };
	
	int n = sizeof(images) / sizeof(images[0]);
	cv::Mat mat_ref;
	mat_ref = cv::imread(images[2], CV_8UC1);
	for (int i = 0; i < n; i++) {
		std::cout << "\n*" << i << "* = " << images[i] << std::endl;
		mat_ref = cv::imread(images[i], CV_8UC1);
		kernel_wrapper(mat_ref);
	}
	*/
	
	
	// Read in image and allocate matrices for processed image
	cv::Mat mat_ref = cv::imread("baby.bmp", CV_8UC1);

	// Call function in .cu file
	//kernel_wrapper(mat_ref);
	mat_ref = cv::imread("baby.JPG.bmp", CV_8UC1);
	kernel_wrapper(mat_ref);
	
	printf("Exit with zero errors...\n");
	
	device_rst();
	getchar();
	return 0;
}