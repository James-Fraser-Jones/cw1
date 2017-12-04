/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
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
void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;


int min_x[256], min_y[256], max_x[256], max_y[256];
int x_point[256], y_point[256], postion_group[256];

//find feature points group
void find(int num, int group, int range, int all)
{
	postion_group[num] = group;
	if (x_point[num] > max_x[group]) max_x[group] = x_point[num];
	if (y_point[num] > max_y[group]) max_y[group] = y_point[num];
	if (x_point[num] < min_x[group] || min_x[group] <= 0) min_x[group] = x_point[num];
	if (y_point[num] < min_y[group] || min_y[group] <= 0) min_y[group] = y_point[num];

	for (int n = 1; n <= all; n++)
		if (postion_group[n] == -1 &&
			x_point[n] > x_point[num] - range && x_point[n] < x_point[num] + range &&
			y_point[n] > y_point[num] - range && y_point[n] < y_point[num] + range)
		{
			postion_group[n] = group;
			find(n, group, range, all);
		}
}



/** @function main */
int main( int argc, const char** argv ){

 /* if( argc != 2 ){
   printf("Error: Incorrect arguments.\n");
   return -1;
  }

	//Read command line arguments
	double checkthresh;
	sscanf(argv[1],"%lf", &checkthresh);
 */

	//threshold of hessian for SURF
	int threshold_hessian = 1000; 

	Mat input = imread("dartt.jpg", 1); // load detect object's image
	if (!input.data)
	{
		printf("NO IMAGE \n"); return false;
	}

	Mat src_gray_1, src_gaussian_1,
		sobelx, sobely, gradient, bla_whi, processed_1;
	//convert_to_gray
	cvtColor(input, src_gray_1, COLOR_BGR2GRAY);
	//reduce_noise and reduce the edge strength
	GaussianBlur(src_gray_1, 12, src_gaussian_1);
	input = src_gaussian_1;
	vector<Rect> dart;

 //open bounding text file
  ofstream myfile;
  myfile.open ("bounding.txt");

  for (int i = 0; i < 16; i++) {

	  char * file = new char[16];
	  sprintf(file, "tests/%d.jpg", i);
	  printf("image: %d, ", i);
	  Mat frame = imread(file, CV_LOAD_IMAGE_COLOR);
	  Mat filteredframe = frame.clone();
	  //----------------------------------------------------------------------
	  //detect feature points

		  int minHessian = threshold_hessian;  // the hessian threshold in SURF algorithm
		  SurfFeatureDetector detector(minHessian);//Define one SurfFeatureDetector（SURF）detecter
		  std::vector<KeyPoint> keyPoint1, keyPoint2;//vector class，save the dynamic array

          //detect SURF feature points，saved in vector
		  detector.detect(input, keyPoint1);
		  detector.detect(frame, keyPoint2);

		  //Calculate feature vectors
		  SurfDescriptorExtractor extractor;
		  Mat descriptors1, descriptors2;
		  extractor.compute(input, keyPoint1, descriptors1);
		  extractor.compute(frame, keyPoint2, descriptors2);

		  // using BruteForce for matching
		  // Instantiate a matcher
		  BruteForceMatcher< L2<float> > matcher;
		  std::vector< DMatch > matches;
		  //match the descriptors
		  matcher.match(descriptors1, descriptors2, matches);

		  //提取强特征点
		  double minMatch = 1;
		  double maxMatch = 0;

		  for (int i = 0; i < matches.size(); i++)
		  {
			  //Get the maximum and minimum value
			  minMatch = minMatch > matches[i].distance ? matches[i].distance : minMatch;
			  maxMatch = maxMatch < matches[i].distance ? matches[i].distance : maxMatch;
		  }
		  // Output best and worst match
		  cout << "Best match of " << file << "：" << minMatch << endl;
		  cout << "Worst match of " << file << "：" << maxMatch << endl;





		  vector<DMatch> goodMatchePoints;
		  int count_match = 0, bestgroup = 0;
		  for (int i = 0; i < matches.size(); i++)
		  {
			  if (matches[i].distance < 0.2 || matches[i].distance == minMatch)
			  {
				  count_match += 1;
				  goodMatchePoints.push_back(matches[i]);
				  x_point[count_match] = (int)keyPoint2[matches[i].trainIdx].pt.x;
				  y_point[count_match] = (int)keyPoint2[matches[i].trainIdx].pt.y;
				  postion_group[count_match] = -1;
				  if (matches[i].distance == minMatch) bestgroup = count_match;
			  }
		  }

		  int count_group = 0, range = (int)MIN(frame.rows - 1, frame.cols - 1) / 10; //search the <range> near the point
		  for (int i = 1; i <= count_match; i++)
		  {
			  if (postion_group[i] == -1)
			  {
				  count_group += 1;
				  min_x[count_group] = 0; min_y[count_group] = 0;
				  max_x[count_group] = 0; max_y[count_group] = 0;
				  find(i, count_group, range, count_match);
			  }
			  if (i == bestgroup) bestgroup = postion_group[i];
		  }

		  //Find the target that meets the requirements based on match points group
		  char * buffer = new char[30];
		  int count_d = 0;
		  for (int i = 1; i <= count_group; i++)
		  {
			  if (min_x[i] != max_x[i] && min_y[i] != max_y[i] && (max_x[i] - min_x[i]) >= range / 2 && (max_y[i] - min_y[i]) >= range / 2)
			  {
				  count_d += 1;
				  int dis_x = max_x[i] - min_x[i];
				  int dis_y = max_y[i] - min_y[i];
				  Rect d = Rect(min_x[i] - dis_y / 4, min_y[i] - dis_x / 4, dis_x + dis_y / 2, dis_y + dis_x / 2);

				  rectangle(filteredframe, d, cvScalar(0, 0, 255), 3, 4, 0);
				  sprintf(buffer, "%d %d %d %d ", d.x, d.y, d.x + d.width, d.y + d.height);
				  myfile << buffer;
			  }
		  }

		  /*
		  //drawing match points
		  Mat imageOutput;
		  drawMatches(input, keyPoint1, frame, keyPoint2, goodMatchePoints, imageOutput, Scalar::all(-1),
			  Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		  imshow("Mathch Points", imageOutput);*/


		  //if there are non good results then output the optimal solution
		  //----------------------------------------------------------------------
	  if (count_d == 0)
	  {
		  //0. Get Reverse Image
		  char * reversefile = new char[16];
		  sprintf(reversefile, "reverse/%dr.jpg", i);
		  Mat reverse_image = imread(reversefile, CV_LOAD_IMAGE_GRAYSCALE);



		  // 2. Load the Strong Classifier in a structure called `Cascade'
		  if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

		  // 3. Detect Dartboards and Display Result
		  detectAndDisplay(frame, filteredframe, reverse_image, 0.2, myfile);
	  }
			myfile << "\n";
			string argv1(file);
			argv1.erase(0, 6); //strip off "tests/" from filename
			size_t lastindex = argv1.find_last_of("."); //strip off ".jpg" file extension
			string filename = argv1.substr(0, lastindex); //strip off ".jpg" file extension
			string detections("detections/");
			string detected("_D.jpg");
			string filtered("_DF.jpg");
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
    //draw box on original image
    rectangle(frame, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
    /*
    sprintf(buffer, "%d %d %d %d ", darts[i].x, darts[i].y, darts[i].x + darts[i].width, darts[i].y + darts[i].height);
    myfile << buffer;
    //*/
	}

  //*
  // 4. Filter out all boxes except for the top *checkthresh* number of them
  int size = darts.size();

  //array to store max value of each roi, in the same order as the array of rois
  int* vals = new int[size];
  double checkVal = 0;
  for (int i = 0; i<size; i++){

    Mat subimage = points_image(darts[i]);

    double min, max;
    cv::minMaxLoc(subimage, &min, &max);

    if (max > checkVal){
      checkVal = max;
    }
    vals[i] = max;
  }
  checkVal = checkVal*(1-checkthresh);

  /*
  vector<Rect> orderedDarts; //same as darts array but ordered (high to low) by size of max value, given by vals
  for (int i = 0; i<size; i++){
    int maxIndex = distance(vals, max_element(vals, vals + size));
    orderedDarts.push_back(darts[maxIndex]);
    vals[maxIndex] = 0; //don't choose the same highest value twice
  }
  */

  int count = 0;
  for (int i = 0; i < size; i++){
    if (vals[i] > checkVal){
      rectangle(filteredframe, Point(darts[i].x, darts[i].y), Point(darts[i].x + darts[i].width, darts[i].y + darts[i].height), Scalar( 0, 255, 0 ), 2);
      sprintf(buffer, "%d %d %d %d ", darts[i].x, darts[i].y, darts[i].x + darts[i].width, darts[i].y + darts[i].height);
      myfile << buffer;
      count++;
    }
  }

  //Print number of dartboards after filtering
  std::cout << count << std::endl;
  //*/
}

void GaussianBlur(cv::Mat &input, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());

	// create the Gaussian kernel in 1 D
	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	//CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	//TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);

	// the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			blurredOutput.at<uchar>(i, j) = (uchar)sum;
		}
	}
}