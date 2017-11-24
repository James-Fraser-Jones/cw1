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

void test0(Mat frame){
	//(423,1,620,218)
	rectangle(frame, Point(423, 1), Point(620, 218), Scalar(0, 255, 255), 2); //good
}
void test1(Mat frame){
	//(166, 106, 420, 356)
	rectangle(frame, Point(166, 106), Point(420, 356), Scalar(0, 255, 255), 2); //good
}
void test2(Mat frame){
	//(88, 88, 201, 196)
	rectangle(frame, Point(88, 88), Point(201, 196), Scalar(0, 255, 255), 2); //good
	//(317, 50, 378, 102)
	rectangle(frame, Point(317, 50), Point(378, 102), Scalar(0, 255, 255), 2); //good
}
void test3(Mat frame){
	//(311, 138, 400, 233)
	rectangle(frame, Point(311, 138), Point(400, 233), Scalar(0, 255, 255), 2); //good
}
void test4(Mat frame){
	//(157, 68, 417, 321)
	rectangle(frame, Point(157, 68), Point(417, 321), Scalar(0, 255, 255), 2); //good
}
void test5(Mat frame){
	//(417, 126, 550, 252)
	rectangle(frame, Point(417, 126), Point(550, 252), Scalar(0, 255, 255), 2); //good
}
void test6(Mat frame){
	//(204, 108, 281, 190)
	rectangle(frame, Point(204, 108), Point(281, 190), Scalar(0, 255, 255), 2); //good
}
void test7(Mat frame){
	//(235, 155, 415, 334)
	rectangle(frame, Point(235, 155), Point(415, 334), Scalar(0, 255, 255), 2); //good
}
void test8(Mat frame){
	//(824, 203, 975, 355)
	rectangle(frame, Point(824, 203), Point(975, 355), Scalar(0, 255, 255), 2); //good
	//(63, 243, 134, 351)
	rectangle(frame, Point(63, 243), Point(134, 351), Scalar(0, 255, 255), 2); //good
}
void test9(Mat frame){
	//(164, 17, 465, 318)
	rectangle(frame, Point(164, 17), Point(465, 318), Scalar(0, 255, 255), 2); //good
	//(141, 532, 225, 587)
	rectangle(frame, Point(141, 532), Point(225, 587), Scalar(0, 255, 255), 2); //good
}
void test10(Mat frame){
	//(77, 90, 99, 230)
	rectangle(frame, Point(77, 90), Point(199, 230), Scalar(0, 255, 255), 2); //good
	//(577, 119, 646, 224)
	rectangle(frame, Point(577, 119), Point(646, 224), Scalar(0, 255, 255), 2); //good
	//(912, 142, 957, 224)
	rectangle(frame, Point(912, 142), Point(957, 224), Scalar(0, 255, 255), 2); //good
}
void test11(Mat frame){
	//(163, 94, 243, 187)
	rectangle(frame, Point(163, 94), Point(243, 187), Scalar(0, 255, 255), 2); //good
	//(436, 104, 498, 192)
	rectangle(frame, Point(436, 104), Point(498, 192), Scalar(0, 255, 255), 2); //good
}
void test12(Mat frame){
	//(147, 59, 227, 236)
	rectangle(frame, Point(147, 59), Point(227, 236), Scalar(0, 255, 255), 2); //good
}
void test13(Mat frame){
	//(252, 101, 423, 271)
	rectangle(frame, Point(252, 101), Point(423, 271), Scalar(0, 255, 255), 2); //good
}
void test14(Mat frame){
	//(102, 87, 265, 247)
	rectangle(frame, Point(102, 87), Point(265, 247), Scalar(0, 255, 255), 2); //good
	//(968, 79, 1131, 240)
	rectangle(frame, Point(968, 79), Point(1131, 240), Scalar(0, 255, 255), 2); //good

}
void test15(Mat frame){
	//(130, 32, 304, 216)
	rectangle(frame, Point(130, 32), Point(304, 216), Scalar(0, 255, 255), 2); //good
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	test9(frame);
}

/*
rect1 = {
  left: x1,
  right: x1 + x2,
  top: y1,
  bottom: y1 + y2,
}
rect2 = {
  left: x1,
  right: x1 + x2,
  top: y1,
  bottom: y1 + y2,
}
area1 = (rect1.right - rect1.left) * (rect1.bottom - rect1.top)
area2 = (rect2.right - rect2.left) * (rect2.bottom - rect2.top)

x_overlap = Math.max(0, Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
y_overlap = Math.max(0, Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
intersection = x_overlap * y_overlap;
union = area1 + area2 - intersection
uoi = union/intersection
match = (uoi >= 0.5)
*/
