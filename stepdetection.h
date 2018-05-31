
#ifndef STEPDETECTION_H
#define STEPDETECTION_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cmath>

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
    bool found;
    double errork1;
    cv::Mat img;
    cv::Mat seg;
    cv::Mat state;
    cv::Mat measure;
    cv::Point center;
    cv::Rect predRect;
    map<int, Rect> fboxes;
} frame_out;

static frame_out img_out = frame_out();

void paintRectangles(Mat &img, map<int, Rect> &bboxes);
void getBlobs(Mat labls, map<int, Rect>&bboxes);
void getFeet(Mat fg, map<int, Rect> &bboxes, Mat labels, Mat labels2, map<int, Rect> &fboxes);

//// New Functions ////
void getFileInput (ofstream &);
void KalmanInit(cv::KalmanFilter kf);
void KalmanPredict(frame_out *img_out, cv::KalmanFilter kf, cv::Mat state, cv::Rect *predRect, cv::Point *center_kalman, int dT);
void KalmanResetAndStep(frame_out *img_out, cv::Point *center_kalman, cv::Point *center_measured, cv::Rect *predRect, double *errork1, bool *found);





void KalmanUpdate(frame_out *img_out, cv::KalmanFilter kf, int *notFoundCount, cv::Mat *state, cv::Mat *measure, bool *found);







void FindBoxes(frame_out *img_out, cv::Mat img, bool start, bool *found);
double distance(cv::Point *center_kalman, cv::Point *center_measured);

//// New Variables ////
static int Xk0 = 0, Xk1 = 0;
static string flag_direc;

//// Rectangulo y centro kalman ////
static cv::Rect predRect_R;
static cv::Point center_kalman_R, center_measured_R;


//// Kalman Variables ////
static unsigned int type = CV_32F;
static int stateSize = 6, measSize  = 4, contSize = 0;
static int notFoundCount = 0;

static cv::KalmanFilter kf_R(stateSize, measSize, contSize, type); // NOLINT
static cv::Mat state_R(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]  // NOLINT
static cv::Mat meas_R(measSize, 1, type);    // [z_x,z_y,z_w,z_h]  // NOLINT



#endif // STEPDETECTION_H