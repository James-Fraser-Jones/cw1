#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#include <string>
#include <fstream>

using namespace std;
using namespace cv;

void sobel(cv::Mat &grey_image, cv::Mat &x_image, cv::Mat &y_image){

  double sobelX[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};
  double sobelY[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};

  cv::Mat kernelX = (Mat_<double>(3,3) << sobelX[0][0], sobelX[0][1], sobelX[0][2], sobelX[1][0], sobelX[1][1], sobelX[1][2], sobelX[2][0], sobelX[2][1], sobelX[2][2]);
  cv::Mat kernelY = (Mat_<double>(3,3) << sobelY[0][0], sobelY[0][1], sobelY[0][2], sobelY[1][0], sobelY[1][1], sobelY[1][2], sobelY[2][0], sobelY[2][1], sobelY[2][2]);

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernelX.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( grey_image, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	// now we can do the convoltion
	for ( int i = 0; i < grey_image.rows; i++ )
	{
		for( int j = 0; j < grey_image.cols; j++ )
		{
			double sumX = 0.0;
      double sumY = 0.0;
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
					double kernalvalX = kernelX.at<double>( kernelx, kernely );
          double kernalvalY = kernelY.at<double>( kernelx, kernely );

					// do the multiplication
					sumX += imageval * kernalvalX;
          sumY += imageval * kernalvalY;
				}
			}

			sumX = (sumX/8) + 127.5; //normalization for sobel kernel
      sumY = (sumY/8) + 127.5; //normalization for sobel kernel
			x_image.at<uchar>(i, j) = (uchar) sumX;
      y_image.at<uchar>(i, j) = (uchar) sumY;
		}
	}
}

void magn(cv::Mat &x_image, cv::Mat &y_image, cv::Mat &mag_image){
  for ( int i = 0; i < (x_image.rows); i++){
		for( int j = 0; j < (x_image.cols); j++){

			double xval = ((double) x_image.at<uchar>(i, j)) - 127.5; //centre values around 0
			double yval = ((double) y_image.at<uchar>(i, j)) - 127.5;

			double mag = sqrt((xval*xval) + (yval*yval));
			mag = mag * (255/sqrt(2*127.5*127.5)); //normalization
			mag_image.at<uchar>(i, j) = (uchar) (mag);
		}
	}
}

void angl(cv::Mat &x_image, cv::Mat &y_image, cv::Mat &ang_image){
  for ( int i = 0; i < (x_image.rows); i++){
		for( int j = 0; j < (x_image.cols); j++){

			double xval = ((double) x_image.at<uchar>(i, j)) - 127.5; //centre values around 0
			double yval = ((double) y_image.at<uchar>(i, j)) - 127.5;

			double ang = atan(yval/xval);
			int angN = cvRound(ang * (255/CV_PI)); //normalization
      angN = (angN + 255)%256; //normalization
			ang_image.at<uchar>(i, j) = (uchar) (angN);
	   }
   }
}

int main( int argc, const char** argv ){ //6, 200 first best guess of arguments

	for (int i = 0; i < 16; i++){

		// 1. Read Input Image
		char * file = new char[16];
		sprintf(file, "tests/%d.jpg", i);

		Mat image, grey_image;
    image = imread(file, CV_LOAD_IMAGE_COLOR);
    cvtColor(image, grey_image, CV_BGR2GRAY);
    equalizeHist( grey_image, grey_image ); //not sure whether to use this or not

    //Make a bunch of images for storage
    Mat x_image, y_image, mag_image, ang_image;
    x_image.create(grey_image.size(), grey_image.type());
    y_image.create(grey_image.size(), grey_image.type());
    mag_image.create(grey_image.size(), grey_image.type());
    ang_image.create(grey_image.size(), grey_image.type());

    sobel(grey_image, x_image, y_image);
    magn(x_image, y_image, mag_image);
    angl(x_image, y_image, ang_image);

		string sfile (file);
		sfile.erase(0,6); //strip off "tests/" directory from filename
		size_t lastindex = sfile.find_last_of("."); //strip off ".jpg" file extension
		string filename = sfile.substr(0, lastindex); //strip off ".jpg" file extension
    string mag ("m");
    string ang ("a");
		string directory ("mangs/");
		string extension (".jpg");
		string magname = directory + filename + mag + extension;
    string angname = directory + filename + ang + extension;

		imwrite(magname, mag_image);
    imwrite(angname, ang_image);
	}

	return 0;
}
