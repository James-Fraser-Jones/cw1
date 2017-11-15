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

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

using namespace std;

/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	string argv1 (argv[1]);
	argv1.erase(0,6); //remove "tests/" from argument
	string detected ("ground_");
	string outputname = detected + argv1;

	// 4. Save Result Image
	imwrite(outputname, frame);

	return 0;
}

void test4(Mat frame){
	rectangle(frame, Point(350, 126), Point(472, 251), Scalar(0, 255, 255), 2);
}
void test5(Mat frame){
	rectangle(frame, Point(63, 138), Point(120, 197), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(48, 243), Point(120, 322), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(185, 213), Point(255, 286), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(239, 157), Point(312, 235), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(287, 242), Point(353, 311), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(370, 181), Point(442, 254), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(418, 230), Point(492, 304), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(499, 169), Point(574, 242), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(553, 243), Point(624, 319), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(637, 174), Point(705, 247), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(668, 246), Point(738, 316), Scalar(0, 255, 255), 2);
}
void test13(Mat frame){
	rectangle(frame, Point(412, 125), Point(524, 253), Scalar(0, 255, 255), 2);
}
void test14(Mat frame){
	rectangle(frame, Point(459, 220), Point(556, 316), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(725, 202), Point(828, 299), Scalar(0, 255, 255), 2);
}
void test15(Mat frame){
	rectangle(frame, Point(55, 120), Point(151, 216), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(341, 101), Point(437, 197), Scalar(0, 255, 255), 2);
	rectangle(frame, Point(517, 117), Point(614, 222), Scalar(0, 255, 255), 2);
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	test15(frame);
}
