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
void detectAndDisplay(Mat frame, Mat filteredframe, Mat reverse, double checkthresh, ofstream& myfile);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv ){

  /*
  if( argc != 2 ){
   printf("Error: Incorrect arguments.\n");
   return -1;
  }

	//Read command line arguments
	double checkthresh;
	sscanf(argv[1],"%lf", &checkthresh);
  */

  double checkthresh = 0.2;

  //open bounding text file
  ofstream myfile;
  myfile.open ("bounding.txt");

	for (int i = 0; i < 16; i++){

    //0. Get Reverse Image
    char * reversefile = new char[16];
		sprintf(reversefile, "reverse/%dr.png", i);
    Mat reverse_image = imread(reversefile, CV_LOAD_IMAGE_GRAYSCALE);

		// 1. Read Input Image
		char * file = new char[16];
		sprintf(file, "tests/%d.jpg", i);
		printf("image: %d, ", i);

		Mat frame = imread(file, CV_LOAD_IMAGE_COLOR);
		Mat filteredframe = frame.clone();

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

		// 3. Detect Dartboards and Display Result
		detectAndDisplay( frame, filteredframe, reverse_image, checkthresh, myfile);
    myfile << "\n";

		string argv1 (file);
		argv1.erase(0,6); //strip off "tests/" from filename
		size_t lastindex = argv1.find_last_of("."); //strip off ".jpg" file extension
		string filename = argv1.substr(0, lastindex); //strip off ".jpg" file extension
		string detections ("detections/");
		string detected ("_D.jpg");
		string filtered ("_DF.jpg");
		string outputname = detections + filename + detected;
		string outputname2 = detections + filename + filtered;

		// 4. Save Result Image
		imwrite(outputname, frame);
		imwrite(outputname2, filteredframe);
	}

  //close bounding text file
  myfile.close();

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, Mat filteredframe, Mat points_image, double checkthresh, ofstream& myfile){
	std::vector<Rect> darts;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, darts, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
  std::cout << darts.size() << " ";

  char * buffer = new char[30];

  // 3. Draw box around dartboards found
	for( int i = 0; i < darts.size(); i++ ){
    rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
	}

  // 4. Filter out all boxes except for the top *checkthresh* number of them
  int size = darts.size();

  //array to store max value of each roi, in the same order as the array of rois
  int vals[size];
  double checkVal = 0;
  printf("\nTest 1\n");
  for (int i = 0; i<size; i++){

    Mat subimage = points_image(darts[i]);

    double min, max;
    cv::minMaxLoc(subimage, &min, &max);

    if (max > checkVal){
      checkVal = max;
    }
    vals[i] = max;
  }
  printf("\nTest 2\n");

  checkVal = checkVal*(1-checkthresh);

  int count = 0;
  for (int i = 0; i < size; i++){
    if (vals[i] > checkVal){
      rectangle(filteredframe, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
      sprintf(buffer, "%d %d %d %d ", darts[i].x, darts[i].y, darts[i].x + darts[i].width, darts[i].y + darts[i].height);
      myfile << buffer;
      count++;
    }
  }

  //printf("\nTest 3\n");

  //Print number of dartboards after filtering
  std::cout << count << std::endl;
}
