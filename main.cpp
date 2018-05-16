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


    // >>>> Kalman Filter
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;

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
    //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));
    // <<<< Kalman Filter

    int notFoundCount = 0;
    bool found = false;

    double ticks = 0;


    cv::Mat img_cal, img_test, img_proc;
    frame_out img_out;
    std::map<int, Rect> bboxes;
    cv::Mat labels,labels2;

    // para lectura de imagenes
    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";
    // Flag para comenzar la detección
    bool start = false;

    int count_test = 195, count_cal = 0, limit = 150;
    vector<String> filenames_cal, filenames_test;

    glob(path_test, filenames_test);
    glob(path_cal , filenames_cal);

    string fileName, substring;
    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil/NeuroMisil_CLion/pasos1.csv";
    ofstream ofStream(fileName);
    size_t pos = filenames_test[count_test].find(".jpg");
    int digits = 5;
    ofStream << "Frame" << "," << "CenterX" << "," << "CenterY" << "," << "BottomY" << "," << "Pie" << "\n";

    cv::Mat res;
    int frames_pasos=0;

    char ch = 0;

    while(ch != 'q' && ch != 'Q'){

        ///////////////////////
        // Kalman Prediction //
        ///////////////////////

        double precTick = ticks;
        ticks = (double) cv::getTickCount();

        double dT = (ticks - precTick) / cv::getTickFrequency(); //seconds


        //Frame Acquisitionat

        res = img_test;

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
        }

        if(img_proc.data) {

            img_out = stepDetection_2(img_proc, ofStream, substring, start);

            if (count_cal < limit) {
                if (img_test.data) cv::imshow("Video", img_test);
//                cv::imshow("Video", img_out.img);
            } else {
                cv::imshow("Video", img_out.img);
            }





        }

        //Frame Acquisition



        //frame.copyTo( res );

        if (found)
        {
            // >>>> Matrix A
            dT = 1;
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A

//            cout << "dT:" << endl << dT << endl;

            state = kf.predict();
            //cout << "State post:" << endl << state << endl;

            cv::Rect predRect;
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;

            cv::Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            //cv::circle(res, center, 2, CV_RGB(255,0,0), -1);
            cv::rectangle(img_out.img, predRect, CV_RGB(255,0,0), 2);

        }
        ///////////////////////////////
        // Kalman Prediction Finish //
        //////////////////////////////

        //  img_out:fboxes[1].x
        //  img_out:fboxes[1].y
        //  img_out:fboxes[1].width
        //  img_out:fboxes[1].height
        //  Si existe una diferencia muy grande entre el  centro x,y predicho por kalman
        //  y el x,y medido con la segmentacion se reseteara el filtro y se considerara que se obtuvo un paso


        int centro_x = img_out.fboxes[1].x + img_out.fboxes[1].width/ 2;

        if(state.at<float>(0) > centro_x) {
            if ((state.at<float>(0) - centro_x) > 7){
                found = false;
                frames_pasos = frames_pasos + 1;
            }
        }


        if(state.at<float>(0) < centro_x){
            if( (centro_x - state.at<float>(0)) > 7) {
                found = false;
                frames_pasos = frames_pasos + 1;
            }
        }

        cout << "Frame actual: " << filenames_test[count_test].substr(pos-digits) << endl;
        cout << "Posicion X medida: " << centro_x << endl;
        cout << "Posicion X predecida: " << state.at<float>(0) << endl;
        cout << "Pasos Encontrados: " << frames_pasos << endl;

        // >>>>> Kalman Update
        if (img_out.fboxes[1].width <= 0)
        {
            notFoundCount++;
            //cout << "notFoundCount:" << notFoundCount << endl;
            if( notFoundCount >= 100 )
            {
                found = false;
            }
            /*else
                kf.statePost = state;*/
        }
        else
        {
            notFoundCount = 0;

            meas.at<float>(0) = img_out.fboxes[1].x + img_out.fboxes[1].width/ 2;
            meas.at<float>(1) = img_out.fboxes[1].y + img_out.fboxes[1].height / 2;
            meas.at<float>(2) = (float)img_out.fboxes[1].width;
            meas.at<float>(3) = (float)img_out.fboxes[1].width;

            if (!found) // First detection!
            {
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
            }
            else
                kf.correct(meas); // Kalman Correction

            //cout << "Measure matrix:" << endl << meas << endl;
        }
        // <<<<< Kalman Update

        count_cal++;
        count_test++;


        if(start)imshow("Kalman",img_out.img);

        ch = char(cv::waitKey(0));

    }


    return 0;

}