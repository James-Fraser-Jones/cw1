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

#include <string>
#include <fstream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, Mat filteredframe, int magthresh, int checkthresh, double angprop, ofstream& myfile);

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

void angl2(cv::Mat &x_image, cv::Mat &y_image, cv::Mat &ang2_image){
  for ( int i = 0; i < (x_image.rows); i++){
		for( int j = 0; j < (x_image.cols); j++){

			double xval = ((double) x_image.at<uchar>(i, j)) - 127.5; //centre values around 0
			double yval = ((double) y_image.at<uchar>(i, j)) - 127.5;

			double ang = atan(yval/xval);
			int angN = cvRound(ang * (255/CV_PI)); //normalization
      angN = (angN + 255)%256; //normalization
			ang2_image.at<uchar>(i, j) = (uchar) (angN);
	   }
   }
}

void hough(cv::Mat &thresh_image, cv::Mat &ang2_image, cv::Mat &hough_image, double angprop){
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
            int val = (int) hough_image.at<uchar>(y2, x2);
            //hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            //*
            if ((val + 1) <= 255){
              hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            }
            else{
              hough_image.at<uchar>(y2, x2) = (uchar) 255;
            }
            //*/
          }
        }
        else{
          for(int x2 = minCol; x2 < hough_image.cols; x2++){
            double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
            double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

            int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

            //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
            int val = (int) hough_image.at<uchar>(y2, x2);
            //hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            //*
            if ((val + 1) <= 255){
              hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            }
            else{
              hough_image.at<uchar>(y2, x2) = (uchar) 255;
            }
            //*/
          }
          for(int x2 = 0; x2 < (maxCol+1); x2++){
            double theta = x2 * (CV_PI/hough_image.cols); // theta: [0, pi]
            double rho = (1/sqrt(2))*(cosFactor*cos(theta)+sinFactor*sin(theta)); // rho: [1, -1]

            int y2 = cvRound((hough_image.rows-1)/2 - ((hough_image.rows-1)/2)*rho); // y2: [0, (hough_image.rows-1)]

            //add to value accumulator (this needs to be changed to automatically scale everything correctly rather than truncating)
            int val = (int) hough_image.at<uchar>(y2, x2);
            //hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            //*
            if ((val + 1) <= 255){
              hough_image.at<uchar>(y2, x2) = (uchar) (val + 1);
            }
            else{
              hough_image.at<uchar>(y2, x2) = (uchar) 255;
            }
            //*/
          }
        }

      }

	  }
  }
}

void points(cv::Mat &thresh_image, cv::Mat &hough_image, cv::Mat &points_image){
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
}

cv::Mat getPointsImage(cv::Mat &grey_image, int magthresh, double angprop){

  Mat x_image, y_image, mag_image, ang_image, thresh_image, hough_image, points_image, ang2_image;
  x_image.create(grey_image.size(), grey_image.type());
  y_image.create(grey_image.size(), grey_image.type());
  mag_image.create(grey_image.size(), grey_image.type());
  thresh_image.create(grey_image.size(), grey_image.type());
  hough_image.create(grey_image.size(), grey_image.type());
  points_image.create(grey_image.size(), grey_image.type());
  ang2_image.create(grey_image.size(), grey_image.type());

  sobel(grey_image, x_image, y_image);
  magn(x_image, y_image, mag_image);
  angl2(x_image, y_image, ang2_image);
  thresh(mag_image, thresh_image, magthresh);
  hough(thresh_image, ang2_image, hough_image, angprop);
  points(thresh_image, hough_image, points_image);

  return points_image;

}

bool check(cv::Mat &points_image, cv::Rect &roi, int checkthresh){
  Mat subimage = points_image(roi);

  /*
  double min, max;
  cv::minMaxLoc(subimage, &min, &max);
  */

  if (mean(subimage)[0] > checkthresh){
    return true;
  }
  else{
    return false;
  }
}

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

using namespace std;

/** @function main */
int main( int argc, const char** argv ){ //6, 200 first best guess of arguments

	//Read command line arguments
	int magthresh, checkthresh;
	double angprop;
	sscanf(argv[1],"%d", &magthresh);
	sscanf(argv[2],"%d", &checkthresh);
	sscanf(argv[3],"%lf", &angprop);

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

		// 3. Detect Dartboards and Display Result
		detectAndDisplay( frame, filteredframe, magthresh, checkthresh, angprop, myfile);
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
void detectAndDisplay( Mat frame, Mat filteredframe, int magthresh, int checkthresh, double angprop, ofstream& myfile){
	std::vector<Rect> darts;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
  std::cout << darts.size() << std::endl;

  char * buffer = new char[30];

  // 3. Draw box around dartboards found
	for( int i = 0; i < darts.size(); i++ ){
    //draw box on original image
    rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
    /*
    sprintf(buffer, "%d %d %d %d ", darts[i].x, darts[i].y, darts[i].x + darts[i].width, darts[i].y + darts[i].height);
    myfile << buffer;
    //*/
	}

  //*
  // 4. Filter out all boxes except for the top *checkthresh* number of them
  Mat points_image = getPointsImage(frame_gray, magthresh, angprop);
  int size = darts.size();

  //array to store max value of each roi, in the same order as the array of rois
  int vals[size];
  for (int i = 0; i<size; i++){

    Mat subimage = points_image(darts[i]);

    double min, max;
    cv::minMaxLoc(subimage, &min, &max);

    vals[i] = max;
  }

  vector<Rect> orderedDarts; //same as darts array but ordered (high to low) by size of max value, given by vals
  for (int i = 0; i<size; i++){
    int maxIndex = distance(vals, max_element(vals, vals + size));
    orderedDarts.push_back(darts[maxIndex]);
    vals[maxIndex] = 0; //don't choose the same highest value twice
  }

  for (int i = 0; i < min(size, checkthresh); i++){
    rectangle(filteredframe, Point(orderedDarts[i].x, orderedDarts[i].y), Point(orderedDarts[i].x + orderedDarts[i].width, orderedDarts[i].y + orderedDarts[i].height), Scalar( 0, 255, 0 ), 2);
    sprintf(buffer, "%d %d %d %d ", orderedDarts[i].x, orderedDarts[i].y, orderedDarts[i].x + orderedDarts[i].width, orderedDarts[i].y + orderedDarts[i].height);
    myfile << buffer;
  }

  //Print number of dartboards after filtering
  int count = min(size, checkthresh);
  std::cout << count << std::endl;
  //*/
}

/*
//get region of interest given by detected boudning box
//Rect r(darts[i].x,darts[i].y,darts[i].width,darts[i].height);

//using the points_image and the roi rectangle, get the max pixel value from each detected box and store these values in their own (ordered) array.

/*
//run region through hough filter to check if there are sufficient lines to classify as a dartboard
if (check(points_image, r, checkthresh)){
  rectangle(filteredframe, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
  count++;
  sprintf(buffer, "%d %d %d %d ", faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
  myfile << buffer;
}
*/

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
