#ifndef STEPDETECTION_H
#define STEPDETECTION_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv/cv.h>
#include <QRect>
#include <QDir>



using namespace std;
using namespace cv;

typedef struct {
    Mat img;
    Mat labels;
    bool flag;
    Mat area_filter;
} frame_out ;

void paintRectangles(Mat &img, map<int, Rect>&bboxes);
void getBlobs(Mat labls, map<int, Rect>&bboxes);
void getFeet(Mat fg, map<int, Rect>&bboxes, Mat labels, Mat labels2, map<int, Rect>&fboxes);

void getFileInput (ofstream &);

Mat  stepDetection_1(Mat img);
frame_out  stepDetection_2(Mat img, ofstream &fileout, string substring, bool start);


static int xl1 = 0, yl1 = 0, xr1 = 0, yr1 = 0;
static int xl2 = 0, yl2 = 0, xr2 = 0, yr2 = 0;
static int f1r = 0, f1l = 0, f2r = 0, f2l = 0;

#endif // STEPDETECTION_H
