/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

#include <string>
#include <fstream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, Mat filteredframe, int magthresh, double factor, int linethresh, ofstream& myfile);

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

void thresh(cv::Mat &mag_image, cv::Mat &thresh_image, int threshold){
  for ( int i = 0; i < mag_image.rows; i++ ){
		for( int j = 0; j < mag_image.cols; j++ ){
			if (((int) mag_image.at<uchar>(i,j)) > threshold){
				thresh_image.at<uchar>(i,j) = (uchar) 255;
			}
			else{
				thresh_image.at<uchar>(i,j) = (uchar) 0;
			}
		}
	}
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

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

using namespace std;

/** @function main */
int main( int argc, const char** argv ){

	//Read command line arguments
	int magthresh, linethresh;
	double factor;
	sscanf(argv[1],"%d", &magthresh);
	sscanf(argv[2],"%lf", &factor);
	sscanf(argv[3],"%d", &linethresh);

  //open bounding text file
  ofstream myfile;
  myfile.open ("bounding.txt");

	for (int i = 0; i < 16; i++){

		// 1. Read Input Image
		char * file = new char[16];
		sprintf(file, "tests/dart%d.jpg", i);
		printf("image: %d\n", i);

		Mat frame = imread(file, CV_LOAD_IMAGE_COLOR);
		Mat filteredframe = frame.clone();

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Faces and Display Result
		detectAndDisplay( frame, filteredframe, magthresh, factor, linethresh, myfile);
    myfile << "\n";

		string argv1 (file);
		argv1.erase(0,6); //strip off "tests/" from filename
		size_t lastindex = argv1.find_last_of("."); //strip off ".jpg" file extension
		string filename = argv1.substr(0, lastindex); //strip off ".jpg" file extension
		string subtask3 ("subtask3/");
		string detected ("_D.jpg");
		string filtered ("_DF.jpg");
		string outputname = subtask3 + filename + detected;
		string outputname2 = subtask3 + filename + filtered;

		// 4. Save Result Image
		//imwrite(outputname, frame);
		imwrite(outputname2, filteredframe);
	}

  //close bounding text file
  myfile.close();

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, Mat filteredframe, int magthresh, double factor, int linethresh, ofstream& myfile)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

  // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

  char * buffer = new char[30];

  // 4. Draw box around faces found
	int count = 0;
	for( int i = 0; i < faces.size(); i++ ){

		//get region of interest given by detected boudning box
		Rect r(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
		Mat roi = frame_gray(r);

		//run region through hough filter to check if there are sufficient lines to classify as a dartboard
		if (check(roi, magthresh, factor, linethresh)){
			rectangle(filteredframe, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
			count++;
      sprintf(buffer, "%d %d %d %d ", faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
      myfile << buffer;
		}
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	//Print number of faces after filtering
	std::cout << count << std::endl;
}

/*
string detection ("detection_");
char * iter = new char[2];
string jpg (".jpg");
string outputname;
//*/

/*
sprintf(iter, "%d", i);
outputname = detection + iter + jpg;
imwrite( outputname, roi);
//*/

//Rect R(Point(faces[i].x, faces[i].y),Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height)); //Create a rect
//Mat ROI = src(R) //Crop the region of interest using above rect
//imwrite( "detection.jpg", draw );
//Mat draw = Mat(frame_gray.size(), frame_gray.type(), Scalar::all(0));
//frame_gray(r).copyTo(draw(r));
