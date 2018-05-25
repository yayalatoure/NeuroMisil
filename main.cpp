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


    //// Kalmar Init ////
    KalmanInit(kf_R);

    int notFoundCount = 0;
    bool found = false;

    cv::Mat img_cal, img_test, img_proc, labels, labels2;
    frame_out img_out = frame_out();

    // Images Reading
    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/CALIBRACION01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/TEST01/*.jpg";
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
    bool ocult;
    cv::Rect predRect;

    // measured center
    cv::Point center_measured, center_kalman;
    double errork = 0, errork1 = 0, errork2 = 0, errorp = 0;


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
                kf_R.transitionMatrix.at<float>(2) = dT;
                kf_R.transitionMatrix.at<float>(9) = dT;
                // <<<< Matrix A

                // cout << "dT:" << endl << dT << endl;

                /////// Prediction ///////
                state_R = kf_R.predict();
                //cout << "State post:" << endl << state << endl;

                ////// Predicted Rect Red //////
                predRect.width = static_cast<int>(state_R.at<float>(4));
                predRect.height = static_cast<int>(state_R.at<float>(5));
                predRect.x = static_cast<int>(state_R.at<float>(0) - state_R.at<float>(4)/2);
                predRect.y = static_cast<int>(state_R.at<float>(1) - state_R.at<float>(5)/2);
                cv::rectangle(img_out.img, predRect, CV_RGB(255,0,0), 2);
                //// Predicted Point ////
                center_kalman.x = static_cast<int>(state_R.at<float>(0));
                center_kalman.y = static_cast<int>(state_R.at<float>(1));
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
            //  y el centro x,y medido con la segmentacion se reseteara el filtro y se considerara que se obtuvo un paso

            ocult = bool(img_out.fboxes.size() == 1);
            if(ocult) {
                center_measured.x = img_out.fboxes[1].x + img_out.fboxes[1].width / 2;
                center_measured.y = img_out.fboxes[1].y + img_out.fboxes[1].height / 2;
            }else{
                center_measured.x = img_out.fboxes[2].x + img_out.fboxes[2].width / 2;
                center_measured.y = img_out.fboxes[2].y + img_out.fboxes[2].height / 2;
            }

            errork2 = distance(center_kalman, center_measured);

            if ( abs(errork1) > 1 ){
                found = false;
            }

            if(!ocult & found){
                errorp = (errork2 + errork1)/2;
                if(errorp < 3)
                    cv::rectangle(img_out.img, predRect, CV_RGB(0,0,255), 2);
            }

//
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
            cout << "Posicion X predecida: " << state_R.at<float>(0) << endl;
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
                    kf_R.statePost = state_R;
            }
            else{
                if(img_out.fboxes.size() == 1){
                    meas_R.at<float>(0) = img_out.fboxes[1].x + float(img_out.fboxes[1].width);
                    meas_R.at<float>(1) = img_out.fboxes[1].y + float(img_out.fboxes[1].height)/2;
                    meas_R.at<float>(2) = (float)state_R.at<float>(4);
                    meas_R.at<float>(3) = (float)state_R.at<float>(5);
                }else{
                    meas_R.at<float>(0) = img_out.fboxes[2].x + float(img_out.fboxes[2].width)/2;
                    meas_R.at<float>(1) = img_out.fboxes[2].y + float(img_out.fboxes[2].height)/2;
                    meas_R.at<float>(2) = (float)img_out.fboxes[2].width;
                    meas_R.at<float>(3) = (float)img_out.fboxes[2].height;
                }
                notFoundCount = 0;

                if (!found) { // First detection!

                    // >>>> Initialization
                    kf_R.errorCovPre.at<float>(0) = 1; // px
                    kf_R.errorCovPre.at<float>(7) = 1; // px
                    kf_R.errorCovPre.at<float>(14) = 1;
                    kf_R.errorCovPre.at<float>(21) = 1;
                    kf_R.errorCovPre.at<float>(28) = 1; // px
                    kf_R.errorCovPre.at<float>(35) = 1; // px

                    state_R.at<float>(0) = meas_R.at<float>(0);
                    state_R.at<float>(1) = meas_R.at<float>(1);
                    state_R.at<float>(2) = 0;
                    state_R.at<float>(3) = 0;
                    state_R.at<float>(4) = meas_R.at<float>(2);
                    state_R.at<float>(5) = meas_R.at<float>(3);
                    // <<<< Initialization

                    kf_R.statePost = state_R;

                    found = true;

                }else
                    kf_R.correct(meas_R); // Kalman Correction

                //cout << "Measure matrix:" << endl << meas << endl;
            }
            ////////////////////////////////////
            /////// Kalman Update Finish ///////
            ////////////////////////////////////

        }

        errork1 = errork2;

        count_cal++;
        count_test++;

        if(start)  imshow("Kalman", img_out.img);

        ch = char(cv::waitKey(0));

    }


    return 0;

}