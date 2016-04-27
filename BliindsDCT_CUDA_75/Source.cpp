//=============================================================================
#include "header.h"
//=============================================================================
int main()
{
	
	//images array
	std::string images[] = { "baby.bmp", "babyA.BLR.bmp", "babyA.DCQ.bmp", "babyA.FLT.bmp", "babyA.JP2.bmp", "babyA.JPG.bmp", "babyA.NOZ.bmp",
		"babyB.BLR.bmp", "babyB.DCQ.bmp", "babyB.FLT.bmp", "babyB.JP2.bmp", "babyB.JPG.bmp", "babyB.NOZ.bmp",
		"babyC.BLR.bmp", "babyC.DCQ.bmp", "babyC.FLT.bmp", "babyC.JP2.bmp", "babyC.JPG.bmp", "babyC.NOZ.bmp",
		"horse.bmp", "horseA.BLR.bmp", "horseA.DCQ.bmp", "horseA.FLT.bmp", "horseA.JP2.bmp", "horseA.JPG.bmp", "horseA.NOZ.bmp",
		"horseB.BLR.bmp", "horseB.DCQ.bmp", "horseB.FLT.bmp", "horseB.JP2.bmp", "horseB.JPG.bmp", "horseB.NOZ.bmp",
		"horseC.BLR.bmp", "horseC.DCQ.bmp", "horseC.FLT.bmp", "horseC.JP2.bmp", "horseC.JPG.bmp", "horseC.NOZ.bmp",
		"harbour.bmp", "harbourA.BLR.bmp", "harbourA.DCQ.bmp", "harbourA.FLT.bmp", "harbourA.JP2.bmp", "harbourA.JPG.bmp", "harbourA.NOZ.bmp",
		"harbourB.BLR.bmp", "harbourB.DCQ.bmp", "harbourB.FLT.bmp", "harbourB.JP2.bmp", "harbourB.JPG.bmp", "harbourB.NOZ.bmp",
		"harbourC.BLR.bmp", "harbourC.DCQ.bmp", "harbourC.FLT.bmp", "harbourC.JP2.bmp", "harbourC.JPG.bmp", "harbourC.NOZ.bmp" };
	
	int n = sizeof(images) / sizeof(images[0]);
	cv::Mat mat_ref;
	mat_ref = cv::imread(images[2], CV_8UC1);
	for (int i = 0; i < n; i++) {
		std::cout << "*" << i << "* = " << images[i] << std::endl;
		mat_ref = cv::imread(images[i], CV_8UC1);
		kernel_wrapper(mat_ref);
	}
	
	
	/*
	// Read in image and allocate matrices for processed image
	cv::Mat mat_ref = cv::imread("baby.bmp", CV_8UC1);

	// Call function in .cu file
	//kernel_wrapper(mat_ref);
	//mat_ref = cv::imread("baby.JPG.bmp", CV_8UC1);
	kernel_wrapper(mat_ref);
	
	printf("Exit with zero errors...\n");
	*/
	device_rst();
	//getchar();
	return 0;
}