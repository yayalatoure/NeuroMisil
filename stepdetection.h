
#ifndef STEPDETECTION_H
#define STEPDETECTION_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <math.h>

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
    map<int, Rect> fboxes;
} frame_out ;

void paintRectangles(Mat &img, map<int, Rect>&bboxes);
void getBlobs(Mat labls, map<int, Rect>&bboxes);
void getFeet(Mat fg, map<int, Rect>&bboxes, Mat labels, Mat labels2, map<int, Rect>&fboxes);

//// New Functions ////
void getFileInput (ofstream &);
void KalmanInit(cv::KalmanFilter kf);
double distance(cv::Point center_kalman, cv::Point center_measured);
frame_out FindBoxes(Mat img, ofstream &fileout, bool start);

//// New Variables ////
static int Xk0 = 0, Xk1 = 0, Lk1 = 0, Rk1 = 0;
static string flag_direc;

//// Kalman Variables ////
static unsigned int type = CV_32F;
static int stateSize = 6, measSize  = 4, contSize = 0;

static cv::KalmanFilter kf_R(stateSize, measSize, contSize, type); // NOLINT
static cv::Mat state_R(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]  // NOLINT
static cv::Mat meas_R(measSize, 1, type);    // [z_x,z_y,z_w,z_h]  // NOLINT




#endif // STEPDETECTION_H