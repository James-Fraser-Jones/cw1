#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <iostream>

using namespace cv;
using namespace std;

void sobel(
  cv::Mat &grey_image,
  cv::Mat &x_image,
  cv::Mat &y_image);

void magn(
  cv::Mat &x_image,
  cv::Mat &y_image,
  cv::Mat &mag_image);

void angl(
  cv::Mat &x_image,
  cv::Mat &y_image,
  cv::Mat &mag_image,
  cv::Mat &ang_image);

void thresh(
  cv::Mat &mag_image,
  cv::Mat &thresh_image,
  int threshold);

bool houghline(
  cv::Mat &thresh_image,
  cv::Mat &line_image,
  double factor,
  int threshold);

bool check(
  cv::Mat &grey_image,
  int magthresh,
  double factor,
  int linethresh);

void houghtest(
  cv::Mat &hough_image,
  int x,
  int y);

void hough(cv::Mat &thresh_image, cv::Mat &hough_image){
  double yscale = 0.6;
  int brightness = 4;

  for ( int y = 0; y < thresh_image.rows; y++ ){
		for( int x = 0; x < thresh_image.cols; x++ ){
      int val = thresh_image.at<int>(x, y);
      if (val == 255){

        double dx = x;
        double cosFactor = (-2)*(dx/(hough_image.rows-1)) + 1;
        double dy = y;
        double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1;

        for(int y2 = 0; y2 < hough_image.rows; y2++){
          double theta = y2 * (CV_PI/hough_image.rows);

          //int x2 = cvRound(hough_image.cols/2 - hough_image.cols*sinFactor*sin(theta)/2);
          int x2 = cvRound(hough_image.cols/2 - yscale*hough_image.cols*(cosFactor*cos(theta)+sinFactor*sin(theta))/2);

          if (x2 < 0){
            x2 = 0;
          }
          else if (x2 > hough_image.cols-1){
            x2 = hough_image.cols-1;
          }

          //printf("x2 := %d, y2 := %d, sinFactor := %d*(-2/(%d-1)) + 1 = %f\n", x2, y2, y, hough_image.rows, sinFactor);
          //int val = hough_image.at<int>(x2, y2);
          //hough_image.at<uchar>(x2, y2) = (uchar) (val + 30);
          hough_image.at<uchar>(x2, y2) += (uchar) (brightness);
        }

      }
	  }
  }
}

void houghtest(cv::Mat &hough_image, int x, int y){
  double dx = x;
  double cosFactor = (-2)*(dx/(hough_image.rows-1)) + 1;
  double dy = y;
  double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1;

  for(int y2 = 0; y2 < hough_image.rows; y2++){
    double theta = y2 * (2*CV_PI/hough_image.rows);

    //int x2 = cvRound(hough_image.cols/2 - hough_image.cols*sinFactor*sin(theta)/2);
    int x2 = cvRound(hough_image.cols/2 - hough_image.cols*(cosFactor*cos(theta)+sinFactor*sin(theta))/2);

    if (x2 < 0){
      x2 = 0;
    }
    else if (x2 > hough_image.cols-1){
      x2 = hough_image.cols-1;
    }

    //printf("x2 := %d, y2 := %d, sinFactor := %d*(-2/(%d-1)) + 1 = %f\n", x2, y2, y, hough_image.rows, sinFactor);
    //int val = hough_image.at<int>(x2, y2);
    hough_image.at<uchar>(x2, y2) += (uchar) (100);
  }
}

int main( int argc, char** argv ){

  //read in the original image
	char* imageName = argv[1];
	Mat image;
	image = imread( imageName, 1 );
	if( argc != 2 || !image.data )
	{
	 printf( " No image data \n " );
	 return -1;
	}

////////////////////////////////////////////////////////////////////////////////////////////////

  //create all our images for manipulating
  Mat grey_image, x_image, y_image, mag_image, ang_image, thresh_image, line_image, hough_image;
  cvtColor( image, grey_image, CV_BGR2GRAY );
  cvtColor( image, ang_image, CV_BGR2HSV );
  cvtColor( grey_image, line_image, CV_GRAY2BGR );
  ///////////////////////////////////////////////////////
  x_image.create(grey_image.size(), grey_image.type());
  y_image.create(grey_image.size(), grey_image.type());
  mag_image.create(grey_image.size(), grey_image.type());
  thresh_image.create(grey_image.size(), grey_image.type());
  hough_image.create(grey_image.size(), grey_image.type());
  //*

  //use sobel convolution to get x and y derivative images
  sobel(grey_image, x_image, y_image);

  //use x and y derivative images to get magnitude and angle images
  magn(x_image, y_image, mag_image);
  angl(x_image, y_image, mag_image, ang_image);

  //threshold values from magnitude image to get thresholded image
  thresh(mag_image, thresh_image, 8);

  hough(thresh_image, hough_image);
  /*
  houghtest(hough_image, 0, 0);
  houghtest(hough_image, 100, 100);
  houghtest(hough_image, 283, 283);
  houghtest(hough_image, 400, 400);
  houghtest(hough_image, 566, 566);
  //*/

  /*
  houghtest(hough_image, 0, 566);
  houghtest(hough_image, 283, 0);
  houghtest(hough_image, 283, 283);
  houghtest(hough_image, 283, 566);
  houghtest(hough_image, 566, 0);
  houghtest(hough_image, 566, 283);
  houghtest(hough_image, 566, 566);
  //*/

////////////////////////////////////////////////////////////////////////////////////////////////

  //show all our images before returning
  namedWindow( "Original", CV_WINDOW_AUTOSIZE );
	imshow( "Original", image);

  /*
  namedWindow( "X-gradient", CV_WINDOW_AUTOSIZE );
	imshow( "X-gradient", x_image);
	namedWindow( "Y-gradient", CV_WINDOW_AUTOSIZE );
	imshow( "Y-gradient", y_image);
  namedWindow( "Magnitude", CV_WINDOW_AUTOSIZE );
	imshow( "Magnitude", mag_image);
	cvtColor(ang_image, ang_image, CV_HSV2BGR);
	namedWindow("Angle-Colour", CV_WINDOW_AUTOSIZE);
	imshow( "Angle-Colour", ang_image);
  //*/

  namedWindow("Thresholded Magnitude", CV_WINDOW_AUTOSIZE);
	imshow( "Thresholded Magnitude", thresh_image);
  //namedWindow("Detected Lines", CV_WINDOW_AUTOSIZE);
	//imshow( "Detected Lines", line_image);
  namedWindow("Hough Space", CV_WINDOW_AUTOSIZE);
	imshow( "Hough Space", hough_image);
  waitKey(0);

	return 0;
}

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

void angl(cv::Mat &x_image, cv::Mat &y_image, cv::Mat &mag_image, cv::Mat &ang_image){
  for ( int i = 0; i < (x_image.rows); i++){
		for( int j = 0; j < (x_image.cols); j++){

			double xval = ((double) x_image.at<uchar>(i, j)) - 127.5; //centre values around 0
			double yval = ((double) y_image.at<uchar>(i, j)) - 127.5;

			double ang = atan2(yval,xval);
			ang = ang * (90/CV_PI) + 90; //normalization
			ang_image.at<Vec3b>(i, j)[0] = ang;
			ang_image.at<Vec3b>(i, j)[1] = 255;
			ang_image.at<Vec3b>(i, j)[2] = (uchar) mag_image.at<uchar>(i, j);
	   }
   }
}

void thresh(cv::Mat &mag_image, cv::Mat &thresh_image, int threshold){
  double cutoff = threshold*mean(mag_image)[0];

  for ( int i = 0; i < mag_image.rows; i++ ){
		for( int j = 0; j < mag_image.cols; j++ ){
			if (((int) mag_image.at<uchar>(i,j)) > cutoff){
				thresh_image.at<uchar>(i,j) = (uchar) 255;
			}
			else{
				thresh_image.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
}

bool houghline(cv::Mat &thresh_image, cv::Mat &line_image, double factor, int threshold){
	vector<Vec2f> lines;
	HoughLines(thresh_image, lines, 1, CV_PI/180, cvRound(factor*thresh_image.rows), 0, 0 );

	printf("Number of lines:%d\n", lines.size());

	for( size_t i = 0; i < lines.size(); i++ ){
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a*rho, y0 = b*rho;
			pt1.x = cvRound(x0 + 1000*(-b));
			pt1.y = cvRound(y0 + 1000*(a));
			pt2.x = cvRound(x0 - 1000*(-b));
			pt2.y = cvRound(y0 - 1000*(a));
			line(line_image, pt1, pt2, Scalar(0,0,255), 2, CV_AA);
	}

	if (lines.size() > threshold){
		return true;
	}
	return false;
}

bool check(cv::Mat &grey_image, int magthresh, double factor, int linethresh){ //40, 0.5, 15 seem like sensible parameters for now

  Mat x_image, y_image, mag_image;
  x_image.create(grey_image.size(), grey_image.type());
  y_image.create(grey_image.size(), grey_image.type());
  mag_image.create(grey_image.size(), grey_image.type());

  sobel(grey_image, x_image, y_image);
  magn(x_image, y_image, mag_image);
  thresh(mag_image, mag_image, magthresh);

  vector<Vec2f> lines;
	HoughLines(mag_image, lines, 1, CV_PI/180, cvRound(factor*mag_image.rows), 0, 0 );

  if (lines.size() > linethresh){
		return true;
	}
	return false;
}
