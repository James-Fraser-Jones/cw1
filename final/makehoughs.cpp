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

void thresh(cv::Mat &mag_image, cv::Mat &thresh_image, int threshold){

  double cutoff = threshold*mean(mag_image)[0];

  for ( int i = 1; i < (mag_image.rows-1); i++ ){
    for( int j = 1; j < (mag_image.cols-1); j++ ){
      if (((int) mag_image.at<uchar>(i,j)) > cutoff){
        thresh_image.at<uchar>(i,j) = (uchar) 255;
      }
      else{
        thresh_image.at<uchar>(i,j) = (uchar) 0;
      }
    }
  }
}

void hough(cv::Mat &thresh_image, cv::Mat &ang2_image, double angprop, cv::Mat &hough_image){

  Mat largeHough;
  largeHough.create(hough_image.size(), CV_16UC1);

  for ( int y = 0; y < hough_image.rows; y++ ){
		for( int x = 0; x < hough_image.cols; x++ ){
      hough_image.at<uchar>(y, x) = (uchar) 0;
      largeHough.at<ushort>(y, x) = (ushort) 0;
    }
  }

  for ( int y = 0; y < thresh_image.rows; y++ ){
		for( int x = 0; x < thresh_image.cols; x++ ){

      int val = (int) thresh_image.at<uchar>(y, x);
      if (val > 128){
        double dy = y; // y: [0, (hough_image.rows-1)]
        double sinFactor = (-2)*(dy/(hough_image.rows-1)) + 1; // sinFactor: [1, -1]

        double dx = x; // x: [0, (hough_image.cols-1)]
        double cosFactor = (-2)*(dx/(hough_image.cols-1)) + 1; // cosFactor: [1, -1]

        int minCol, maxCol;
        if (angprop == 1){
          minCol = 0;
          maxCol = (hough_image.cols-1);
        }
        else{
          double ang = (double) ang2_image.at<uchar>(y, x);
          int col = cvRound((ang/255)*(hough_image.cols-1));
          int range = cvRound(angprop*(hough_image.cols-1)/2);
          minCol = (col - range)%hough_image.cols + ((col - range)%hough_image.cols < 0 ? hough_image.cols : 0);
          maxCol = (col + range)%hough_image.cols;
        }

        if (maxCol >= minCol){
          for(int x2 = minCol; x2 < (maxCol+1); x2++){
            double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
            double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

            int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

            //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
            unsigned int houghval = (unsigned int) largeHough.at<ushort>(y2, x2);
            if ((houghval + 1) <= 65535){
              largeHough.at<ushort>(y2, x2) = (ushort) (houghval + 1);
            }
            else{
              largeHough.at<ushort>(y2, x2) = (ushort) (65535);
            }
          }
        }
        else{
          for(int x2 = minCol; x2 < hough_image.cols; x2++){
            double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
            double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

            int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

            //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
            unsigned int houghval = (unsigned int) largeHough.at<ushort>(y2, x2);
            if ((houghval + 1) <= 65535){
              largeHough.at<ushort>(y2, x2) = (ushort) (houghval + 1);
            }
            else{
              largeHough.at<ushort>(y2, x2) = (ushort) (65535);
            }
          }
          for(int x2 = 0; x2 < (maxCol+1); x2++){
            double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
            double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

            int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

            //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
            unsigned int houghval = (unsigned int) largeHough.at<ushort>(y2, x2);
            if ((houghval + 1) <= 65535){
              largeHough.at<ushort>(y2, x2) = (ushort) (houghval + 1);
            }
            else{
              largeHough.at<ushort>(y2, x2) = (ushort) (65535);
            }
          }
        }

      }

	  }
  }

  double min, max;
  cv::minMaxLoc(largeHough, &min, &max);
  largeHough = 255*(largeHough - min)/(max-min);

  largeHough.convertTo(hough_image, CV_8UC1);
}

int main( int argc, const char** argv ){

  /*
  if( argc != 3 ){
	 printf("Error: Incorrect arguments.\n");
	 return -1;
	}

  int threshold;
  double angleprop;
	sscanf(argv[1],"%d", &threshold);
  sscanf(argv[2],"%lf", &angleprop);
  */

  int threshold = 4;
  double angleprop = 0.1;

	for (int i = 0; i < 16; i++){

		// 1. Read Input Image
		char * magfile = new char[16];
    char * angfile = new char[16];
		sprintf(magfile, "mangs/%dm.png", i);
    sprintf(angfile, "mangs/%da.png", i);

		Mat mag_image, ang_image;
    mag_image = imread(magfile, CV_LOAD_IMAGE_GRAYSCALE);
    ang_image = imread(angfile, CV_LOAD_IMAGE_GRAYSCALE);

    //Make a bunch of images for storage
    Mat thresh_image, hough_image;
    thresh_image.create(mag_image.size(), mag_image.type());
    hough_image.create(mag_image.size(), mag_image.type());

    thresh(mag_image, thresh_image, threshold);
    hough(thresh_image, ang_image, angleprop, hough_image);

		string sfile (magfile);
		sfile.erase(0,6); //strip off "mangs/" directory from filename
		size_t lastindex = sfile.find_last_of("."); //strip off ".jpg" file extension
		string filename = sfile.substr(0, lastindex-1); //strip off "m.jpg"
    string hough ("h");
    string thresh ("t");
		string directory ("houghs/");
		string extension (".png");
		string houghname = directory + filename + hough + extension;
    string threshname = directory + filename + thresh + extension;

    imwrite(threshname, thresh_image);
		imwrite(houghname, hough_image);
	}

	return 0;
}
