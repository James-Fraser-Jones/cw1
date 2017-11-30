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

void houghtest(
  cv::Mat &hough_image,
  float xpercent,
  float ypercent);

void hough(
  cv::Mat &thresh_image,
  cv::Mat &hough_image);

void linetest(
  cv::Mat &hough_image);

Point points(
  cv::Mat &thresh_image,
  cv::Mat &hough_image,
  cv::Mat &points_image);

int main( int argc, char** argv ){

  //read in the original image
	char* imageName = argv[1];
	Mat image;
	image = imread( imageName, 1 );
	if( argc != 2 || !image.data )
	{
	 printf( " Error. \n " );
	 return -1;
	}

  //create all our images for manipulating
  Mat grey_image, x_image, y_image, mag_image, ang_image, thresh_image, hough_image, points_image;
  cvtColor( image, grey_image, CV_BGR2GRAY );
  //cvtColor( grey_image, points_image, CV_GRAY2BGR );
  cvtColor( image, ang_image, CV_BGR2HSV );
  x_image.create(grey_image.size(), grey_image.type());
  y_image.create(grey_image.size(), grey_image.type());
  mag_image.create(grey_image.size(), grey_image.type());
  thresh_image.create(grey_image.size(), grey_image.type());
  hough_image.create(grey_image.size(), grey_image.type());
  points_image.create(grey_image.size(), grey_image.type());
  //*

  sobel(grey_image, x_image, y_image);
  magn(x_image, y_image, mag_image);
  angl(x_image, y_image, mag_image, ang_image);
  thresh(mag_image, thresh_image, 6);
  //linetest(hough_image);
  hough(thresh_image, hough_image);
  Point maxLoc = points(thresh_image, hough_image, points_image);
  cvtColor( points_image, points_image, CV_GRAY2BGR );
  circle(points_image, maxLoc, 2, Scalar( 0, 0, 255 ), 2);

  /*
  houghtest(hough_image, 0, 0); // cos + sin
  houghtest(hough_image, 0, 0.5); // cos + 0
  houghtest(hough_image, 0, 1); // cos - sin
  houghtest(hough_image, 0.5, 0); // 0 + sin
  houghtest(hough_image, 0.5, 0.5); // 0 + 0
  houghtest(hough_image, 0.5, 1); // 0 - sin
  houghtest(hough_image, 1, 0); // -cos + sin
  houghtest(hough_image, 1, 0.5); // -cos + 0
  houghtest(hough_image, 1, 1); // -cos - sin
  linetest(hough_image);
  */

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
  namedWindow("Hough Space", CV_WINDOW_AUTOSIZE);
	imshow( "Hough Space", hough_image);
  namedWindow("Detected Line Intersections", CV_WINDOW_AUTOSIZE);
	imshow( "Detected Line Intersections", points_image);
  waitKey(0);

	return 0;
}

Point points(cv::Mat &thresh_image, cv::Mat &hough_image, cv::Mat &points_image){
  int valSum;

  for (int y = 0; y < points_image.rows; y++){

    double percent = y;
    percent = 100*y/points_image.rows;
    printf("%lf\%\n", percent);

    double dy = y; // y: [0, (hough_image.rows-1)]
    double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1; // sinFactor: [1, -1]

		for( int x = 0; x < points_image.cols; x++ ){

      int tval = (int) thresh_image.at<uchar>(y, x);
      if (tval > 128){
        valSum = -hough_image.cols;
      }
      else{
        valSum = 0;
      }

      double dx = x; // x: [0, (hough_image.cols-1)]
      double cosFactor = (-2)*(dx/(hough_image.cols-1)) + 1; // cosFactor: [1, -1]

      for(int x2 = 0; x2 < hough_image.cols; x2++){
        double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
        double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

        int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

        //read value from sinusodial curve and add to sum accumulator
        int val = (int) hough_image.at<uchar>(y2, x2);
        valSum = valSum + val;

      }

      int pval = (int) points_image.at<uchar>(y, x);
      double pointAvg = valSum;
      pointAvg = pointAvg/(hough_image.cols);
      points_image.at<uchar>(y, x) = (uchar) (pointAvg);

	  }
  }

  double min, max;
  Point minLoc, maxLoc;
  cv::minMaxLoc(points_image, &min, &max, &minLoc, &maxLoc);
  points_image = (points_image - min) * (255/(max-min));
  return maxLoc;
  //circle(points_image, Point(minLoc, maxLoc), 2, Scalar( 0, 0, 255 ), 2);
}

//*
void hough(cv::Mat &thresh_image, cv::Mat &hough_image){
  for ( int y = 0; y < thresh_image.rows; y++ ){
		for( int x = 0; x < thresh_image.cols; x++ ){

      int val = (int) thresh_image.at<uchar>(y, x);
      if (val > 128){
        double dy = y; // y: [0, (hough_image.rows-1)]
        double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1; // sinFactor: [1, -1]

        double dx = x; // x: [0, (hough_image.cols-1)]
        double cosFactor = (-2)*(dx/(hough_image.cols-1)) + 1; // cosFactor: [1, -1]

        for(int x2 = 0; x2 < hough_image.cols; x2++){
          double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
          double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

          int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

          //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
          int val = (int) hough_image.at<uchar>(y2, x2);
          if ((val + 1) <= 255){
            hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
          }
          else{
            hough_image.at<uchar>(y2, x2) = (uchar) 255;
          }
        }
      }

	  }
  }

  //double min, max;
  //cv::minMaxLoc(hough_image, &min, &max);
  //hough_image = (hough_image - min) * (255/(max-min));
}

//*
void houghtest(cv::Mat &hough_image, float xpercent, float ypercent){

  int x = cvRound(xpercent*(hough_image.cols-1));
  int y = cvRound(ypercent*(hough_image.rows-1));

  ///////////////////////////////////////////////

  int brightness = 128;

  double dx = x; // x: [0, (hough_image.cols-1)]
  double dy = y; // y: [0, (hough_image.rows-1)]
  double cosFactor = (-2)*(dx/(hough_image.cols-1)) + 1; // cosFactor: [1, -1]
  double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1; // sinFactor: [1, -1]

  for(int x2 = 0; x2 < hough_image.cols; x2++){
    double theta = x2 * (2*CV_PI/hough_image.cols); // theta: [0, 2pi]
    double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

    int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

    //out of bounds handling (this shouldn't be necessary)
    if (y2 < 0){
      y2 = 0;
    }
    else if (y2 > hough_image.rows-1){
      y2 = hough_image.rows-1;
    }

    //add to value accumulator (preventing overflowing above max value 255)
    int val = (int) hough_image.at<uchar>(y2, x2);
    if (val + brightness <= 255){
      hough_image.at<uchar>(y2, x2) = (uchar) (val + brightness);
    }
    else{
      hough_image.at<uchar>(y2, x2) = 255;
    }
  }

}
//*/

void linetest(cv::Mat &hough_image){
  printf("rows := %d, cols := %d\n",hough_image.rows, hough_image.cols);

  for(int y = 0; y < hough_image.rows; y++){

    int x = cvRound(hough_image.cols/2);
    hough_image.at<uchar>(y, x) = 255;

  }
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
	for ( int i = 0; i < grey_image.rows; i++ ){
		for( int j = 0; j < grey_image.cols; j++ ){
			double sumX = 0.0;
      double sumY = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ){
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ){
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
