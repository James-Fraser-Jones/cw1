// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <iostream>

using namespace cv;
using namespace std;

void convolute(
	cv::Mat &input,
	int size,
	cv::Mat &convolvedOutput,
	double (&kernelArray)[3][3]);

int main( int argc, char** argv ){

	char* imageName = argv[1];

	Mat image;
	image = imread( imageName, 1 );

	if( argc != 2 || !image.data )
	{
	 printf( " No image data \n " );
	 return -1;
	}

	double sobelX[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
	double sobelY[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

	Mat grey_image, x_image, y_image, magnitude, angle, angleC;
	cvtColor( image, grey_image, CV_BGR2GRAY );
	magnitude.create(grey_image.size(), grey_image.type());
	angle.create(grey_image.size(), grey_image.type());
	angleC.create(grey_image.size(), CV_8UC3);

	convolute(grey_image, 3, x_image, sobelX);
	convolute(grey_image, 3, y_image, sobelY);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for ( int i = 0; i < (x_image.rows); i++){
		for( int j = 0; j < (x_image.cols); j++){

			double xval = ((double) x_image.at<uchar>(i, j)) - 127.5;
			double yval = ((double) y_image.at<uchar>(i, j)) - 127.5;

			double mag = sqrt((xval*xval) + (yval*yval));
			mag = mag * (255/sqrt(2*127.5*127.5)); //normalization
			magnitude.at<uchar>(i, j) = (uchar) (mag);

			double ang = atan2(yval,xval);
			ang = ang * (127.5/M_PI) + 127.5; //normalization
			//double ang = atan(yval/xval); //tried using normal atan function but it didn't look nearly as nice, probably because it has half the precision
			//ang = ang * (127.5/(M_PI/2)) + 127.5; //normalization
			angle.at<uchar>(i, j) = (uchar) (ang);

			angleC.at<Vec3b>(i, j)[0] = (mag/255)*(ang);
			angleC.at<Vec3b>(i, j)[1] = (mag/255)*((angle.at<uchar>(i, j) + 85)%256);
			angleC.at<Vec3b>(i, j)[2] = (mag/255)*((angle.at<uchar>(i, j) + 2*85)%256);
		}
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	namedWindow( "Original", CV_WINDOW_AUTOSIZE );
	imshow( "Original", image);

	namedWindow( "X-gradient", CV_WINDOW_AUTOSIZE );
	imshow( "X-gradient", x_image);

	namedWindow( "Y-gradient", CV_WINDOW_AUTOSIZE );
	imshow( "Y-gradient", y_image);

	namedWindow( "Magnitude", CV_WINDOW_AUTOSIZE );
	imshow( "Magnitude", magnitude);

	namedWindow( "Angle", CV_WINDOW_AUTOSIZE );
	imshow( "Angle", angle);

	namedWindow( "AngleC", CV_WINDOW_AUTOSIZE );
	imshow( "AngleC", angleC);

	waitKey(0);

	return 0;
}

void convolute(cv::Mat &input, int size, cv::Mat &convolvedOutput, double (&kernelArray)[3][3]){

	convolvedOutput.create(input.size(), input.type());

	cv::Mat kernel = (Mat_<double>(3,3) << kernelArray[0][0], kernelArray[0][1], kernelArray[0][2], kernelArray[1][0], kernelArray[1][1], kernelArray[1][2], kernelArray[2][0], kernelArray[2][1], kernelArray[2][2]);

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{
		for( int j = 0; j < input.cols; j++ )
		{
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			sum = (sum/8) + 127.5; //normalization for sobel kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// set the output value as the sum of the convolution
			convolvedOutput.at<uchar>(i, j) = (uchar) sum;
		}
	}
}

/*
// create the Gaussian kernel in 1D
cv::Mat kX = cv::getGaussianKernel(size, -1);
cv::Mat kY = cv::getGaussianKernel(size, -1);

//cout << "Kernel = "<< endl << " "  << kernel << endl << endl;
//cout << "Sobel = "<< endl << " "  << sobelKernel << endl << endl;

//imwrite( "blur.jpg", carBlurred );

//cout << "Image = "<< endl << " "  << gray_image << endl << endl;
//cout << "Sobel = "<< endl << " "  << carBlurred << endl << endl;

double min, max;
cv::minMaxLoc(gray_image, &min, &max);
printf("Image min: %f max: %f", min, max);

double imageA[image.rows][image.cols];
for (int row; row < image.rows; row++){
	for (int col; col < image.cols; col++){
		imageA[row][col] = (double) image.at<double>(row, col);
	}
}

printf("%f\n", imageA[305][193]);
*/
