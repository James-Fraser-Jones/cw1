#include <stdio.h>
#include <iostream>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int main(){

  //*
  //Initialization of testings conditions
  int actualVals[5] = {5,2,63,6,7};

  Rect d0,d1,d2,d3,d4;
  d0 = Rect(22,41,0,9);
  d1 = Rect(1,3,3,7);
  d2 = Rect(6,10,3,8);
  d3 = Rect(2,90,64,1);
  d4 = Rect(6,7,9,76);

  vector<Rect> darts;
  darts.push_back(d0);
  darts.push_back(d1);
  darts.push_back(d2);
  darts.push_back(d3);
  darts.push_back(d4);
  //*/

  int checkthresh = 5;

  int size = darts.size();

  //array to store max value of each roi, in the same order as the array of rois
  int vals[size];
  for (int i = 0; i<size; i++){
    vals[i] = actualVals[i]; //run function to get max values here
  }

  vector<Rect> orderedDarts; //same as darts array but ordered (high to low) by size of max value, given by vals
  for (int i = 0; i<size; i++){
    int maxIndex = distance(vals, max_element(vals, vals + size));
    orderedDarts.push_back(darts[maxIndex]);
    vals[maxIndex] = 0; //don't choose the same highest value twice
  }

  for (int i = 0; i < min(size, checkthresh); i++){
    printf("orderedDarts[%d] := Rect(%d,%d,%d,%d)\n",i,orderedDarts[i].x,orderedDarts[i].y,orderedDarts[i].width,orderedDarts[i].height);
  }

  return 0;
}
