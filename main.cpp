#include "stepdetection.h"
#include <QApplication>
#include <iostream>
#include <fstream>

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


int main(int argc, char *argv[]){

    ///////////////////////////////////////////////////////////
    ////////////////////// Kalman Filter //////////////////////
    ///////////////////////////////////////////////////////////

    // Kalman Filter sizes
    int stateSize = 6;
    int measSize  = 4;
    int contrSize = 0;


    // Kalman Filter
    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, measSize, contrSize, type);

    cv::Mat state(stateSize, 1, type);  // [x,y,v_x,v_y,w,h]
    cv::Mat meas(measSize, 1, type);    // [z_x,z_y,z_w,z_h]

    //cv::Mat procNoise(stateSize, 1, type)
    // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

    // Transition State Matrix A
    // Note: set dT at each processing step!
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    cv::setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // [ 1 0 0 0 0 0 ]
    // [ 0 1 0 0 0 0 ]
    // [ 0 0 0 0 1 0 ]
    // [ 0 0 0 0 0 1 ]
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
//    cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 1e-2;// 5.0f
    kf.processNoiseCov.at<float>(21) = 1e-2;// 5.0f
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));


    int notFoundCount = 0;
    bool found = false;

    cv::Mat img_cal, img_test, img_proc, labels, labels2;
    frame_out img_out = frame_out();

    // Images Reading
    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";

    // Flag start detection
    bool start = false;

    int count_test = 195, count_cal = 0, limit = 150;
    vector<String> filenames_cal, filenames_test;

    glob(path_test, filenames_test);
    glob(path_cal , filenames_cal);

    //// Logging pasos (frames) ////
//    int digits = 5;
//    string fileName, substring;
//    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil_Lalo/NeuroMisil/Logging/pasos1.csv";
//    ofstream ofStream(fileName);
//    size_t pos = filenames_test[count_test].find(".jpg");
//    ofStream << "Frame" << "," << "CenterX" << "," << "CenterY" << "," << "BottomY" << "," << "Pie" << "\n";

    //// Logging error Kalman (frames) ////
    int digits = 5;
    string fileName, substring;
    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil_Lalo/NeuroMisil/Logging/error_KalmanRight.csv";
    ofstream ofStream(fileName);
    size_t pos = filenames_test[count_test].find(".jpg");
    ofStream << "Frame" << "," << "CX_Kalman" << "," << "CY_Kalman" << "," << "CX_Measured" << "," << "CY,Measured" << "," << "Pie" << "\n";


    int frames_pasos = 0;
    char ch = 0;
    int  dT = 1;

    // measured center
    cv::Point center_measured, center_kalman;
    double error = 0;


    //////////////////////////////////////////////////////////
    /////////////////////// Algoritmo ////////////////////////
    //////////////////////////////////////////////////////////

    while(ch != 'q' && ch != 'Q'){

        ///////////////////////////////////
        //////// Kalman Prediction ////////
        ///////////////////////////////////

        //double ticks = 0;
        //double precTick = ticks;
        //ticks = (double) cv::getTickCount();
        //double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds

        ////////// Frame Acquisition /////////
        if (count_cal < limit){
            img_cal   = imread(filenames_cal[count_cal], CV_LOAD_IMAGE_COLOR);
            img_test  = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos-digits);
            img_proc  = img_cal;

        }else{
            img_test  = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos-digits);
            img_proc  = img_test;
            start = true;
            found = true;
        }
        // Obtain feet's rectangles
        if(img_proc.data) {

                img_out = stepDetection_2(img_proc, ofStream, substring, start);
                if (count_cal < limit) {
                    if (img_test.data) cv::imshow("Video", img_test);
                } else {
                    cv::imshow("Video", img_out.img);
                }
        /////// Frame Acquisition Finish //////


            if (found){
                // >>>> Matrix A
                kf.transitionMatrix.at<float>(2) = dT;
                kf.transitionMatrix.at<float>(9) = dT;
                // <<<< Matrix A

                // cout << "dT:" << endl << dT << endl;

                /////// Prediction ///////
                state = kf.predict();
                //cout << "State post:" << endl << state << endl;

                ////// Predicted Rect //////
                cv::Rect predRect;
                predRect.width = static_cast<int>(state.at<float>(4));
                predRect.height = static_cast<int>(state.at<float>(5));
                predRect.x = static_cast<int>(state.at<float>(0) - state.at<float>(4)/2);
                predRect.y = static_cast<int>(state.at<float>(1) - state.at<float>(5)/2);
                cv::rectangle(img_out.img, predRect, CV_RGB(255,0,0), 2);
                //// Predicted Point ////
                center_kalman.x = static_cast<int>(state.at<float>(0));
                center_kalman.y = static_cast<int>(state.at<float>(1));
                cv::circle(img_out.img, center_kalman, 2, CV_RGB(255,0,0), -1);

                //// Logging ////
                if(start)
                    ofStream << substring << "," << center_kalman.x << "," << center_kalman.y << ",";

            }
            ///////////////////////////////////
            ///// Kalman Prediction Finish ////
            ///////////////////////////////////

            ///////////////////////////////////
            /////////// Kalman Reset //////////
            ///////////////////////////////////

            //  Si existe una diferencia muy grande entre el  centro x,y predicho por kalman
            //  y el x,y medido con la segmentacion se reseteara el filtro y se considerara que se obtuvo un paso


            if(img_out.fboxes.size() == 1) {
                center_measured.x = img_out.fboxes[1].x + img_out.fboxes[1].width / 2;
                center_measured.y = img_out.fboxes[1].y + img_out.fboxes[1].height / 2;
            }else{
                center_measured.x = img_out.fboxes[2].x + img_out.fboxes[2].width / 2;
                center_measured.y = img_out.fboxes[2].y + img_out.fboxes[2].height / 2;
            }

            error = distance(center_kalman, center_measured);

            if ( abs(error) > 1){
                found = false;
            }

//            if(state.at<float>(0) > center_measured.x) {
//                if ((state.at<float>(0) - center_measured.x) > 5){
//                    found = false;
//                }
//            }
//
//            if(state.at<float>(0) < center_measured.x){
//                if( (center_measured.x - state.at<float>(0)) > 5) {
//                    found = false;
//                }
//            }

            ///////////////////////////////////
            /////// Kalman Reset Finish ///////
            ///////////////////////////////////
            if(start)
                ofStream << center_measured.x << "," << center_measured.y << "," << "Rigth" << "\n";

            cout << "Frame actual: " << filenames_test[count_test].substr(pos-digits) << endl;
            cout << "Posicion X predecida: " << state.at<float>(0) << endl;
            cout << "Posicion X medida: " << center_measured.x << endl;


            ///////////////////////////////////
            ////////// Kalman Update //////////
            ///////////////////////////////////

            if (img_out.fboxes[1].width <= 0){
                notFoundCount++;
                //cout << "notFoundCount:" << notFoundCount << endl;
                if( notFoundCount >= 100 ){
                    found = false;
                }
                else
                    kf.statePost = state;
            }
            else{
                if(img_out.fboxes.size() == 1){
                    meas.at<float>(0) = img_out.fboxes[1].x + float(img_out.fboxes[1].width);
                    meas.at<float>(1) = img_out.fboxes[1].y + float(img_out.fboxes[1].height)/2;
                    meas.at<float>(2) = (float)state.at<float>(4);
                    meas.at<float>(3) = (float)state.at<float>(5);
                }else{
                    meas.at<float>(0) = img_out.fboxes[2].x + float(img_out.fboxes[2].width)/2;
                    meas.at<float>(1) = img_out.fboxes[2].y + float(img_out.fboxes[2].height)/2;
                    meas.at<float>(2) = (float)img_out.fboxes[2].width;
                    meas.at<float>(3) = (float)img_out.fboxes[2].height;
                }
                notFoundCount = 0;

                if (!found) { // First detection!

                    // >>>> Initialization
                    kf.errorCovPre.at<float>(0) = 1; // px
                    kf.errorCovPre.at<float>(7) = 1; // px
                    kf.errorCovPre.at<float>(14) = 1;
                    kf.errorCovPre.at<float>(21) = 1;
                    kf.errorCovPre.at<float>(28) = 1; // px
                    kf.errorCovPre.at<float>(35) = 1; // px

                    state.at<float>(0) = meas.at<float>(0);
                    state.at<float>(1) = meas.at<float>(1);
                    state.at<float>(2) = 0;
                    state.at<float>(3) = 0;
                    state.at<float>(4) = meas.at<float>(2);
                    state.at<float>(5) = meas.at<float>(3);
                    // <<<< Initialization

                    kf.statePost = state;

                    found = true;

                }else
                    kf.correct(meas); // Kalman Correction

                //cout << "Measure matrix:" << endl << meas << endl;
            }
            ////////////////////////////////////
            /////// Kalman Update Finish ///////
            ////////////////////////////////////

        }

        count_cal++;
        count_test++;

        if(start)  imshow("Kalman", img_out.img);

        ch = char(cv::waitKey(0));

    }


    return 0;

}