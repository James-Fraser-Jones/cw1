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

void reverse(cv::Mat &hough_image, cv::Mat &points_image){
  int valSum;

  for (int y = 0; y < points_image.rows; y++){

    double percent = y;
    percent = 100*y/points_image.rows;
    printf("%lf\%\n", percent);

    double dy = y; // y: [0, (hough_image.rows-1)]
    double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1; // sinFactor: [1, -1]

		for( int x = 0; x < points_image.cols; x++ ){

      valSum = 0;

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
}

int main( int argc, const char** argv ){

	for (int i = 0; i < 16; i++){

		// 1. Read Input Image
		char * file = new char[16];
		sprintf(file, "houghs/%dh.jpg", i);

		Mat hough_image = imread(file, CV_LOAD_IMAGE_GRAYSCALE);

    //Make a bunch of images for storage
    Mat reverse_image;
    reverse_image.create(hough_image.size(), hough_image.type());

    reverse(hough_image, reverse_image);

		string sfile (file);
		sfile.erase(0,7); //strip off "houghs/" directory from filename
		size_t lastindex = sfile.find_last_of("."); //strip off ".jpg" file extension
		string filename = sfile.substr(0, lastindex-1); //strip off "h.jpg"
    string reverse ("r");
		string directory ("reverse/");
		string extension (".jpg");
		string reversename = directory + filename + reverse + extension;

		imwrite(reversename, reverse_image);
	}

	return 0;
}
