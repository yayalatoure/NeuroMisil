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

    cv::Mat img_cal, img_test, img_proc, img_show, labels, labels2;

    // Images Reading
    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/CALIBRACION01/*.jpg";
    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/TEST01/*.jpg";
//    std::string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/2_CAMARAS/FEED1/*.jpg";
//    std::string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/2_CAMARAS/FEED1/*.jpg";

    // Flag start detection
    bool start = false;

//    int count_test = 195+145, count_cal = 0, limit = 150-145;
    int count_test = 195+145, count_cal = 0, limit = 150-145;
    vector<String> filenames_cal, filenames_test;

    glob(path_test, filenames_test);
    glob(path_cal , filenames_cal);

    //// Logging error Kalman (frames) ////
    int digits = 5;
    string fileName, substring;
    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil_Lalo/NeuroMisil/Logging/pasos_result.csv";
    ofstream ofStream(fileName);
    size_t pos = filenames_test[count_test].find(".jpg");
    ofStream << "Frame" << "," << "CX_Paso" << "," << "CY_Paso" << "," << "W_Paso" << "." << "H_Paso" << "," << "Pie" << "\n";

    char ch = 0;
    int  dT = 0;
    bool found = false;

    double errork1_R = 0, errork1_L=0, errork2_R = 0, errork2_L=0;

    //// Kalmar Init ////
    KalmanInit(*(&kf_R));
    KalmanInit(*(&kf_L));

    while(ch != 'q' && ch != 'Q'){

        ////////// Frame Acquisition /////////
        if (count_cal < limit){
            img_cal   = imread(filenames_cal[count_cal], CV_LOAD_IMAGE_COLOR);
            img_test  = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos-digits);
            img_proc  = img_cal;

        }else{
            img_test  = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            img_show = imread(filenames_test[count_test], CV_LOAD_IMAGE_COLOR);
            substring = filenames_test[count_test].substr(pos-digits);
            img_proc  = img_test;
            start = true;

            cout << substring << '\n' << endl;

        }


        ///// Algoritmo /////


        if(img_proc.data) {

            //////// 2D Feet Boxes ////////
            FindBoxes(&img_out, img_proc, start, &found);

            //////// Kalman Prediction ////////
            if(found){
                KalmanPredict(&img_out, *(&kf_R), &state_R, &predRect_R, &center_kalman_R, dT);
                KalmanPredict(&img_out, *(&kf_L), &state_L, &predRect_L, &center_kalman_L, dT);
            }

            //// Kalman Reset & Step ///////
            MeasureError(&img_out, &center_kalman_R, &center_measured_R, &errork2_R, Right);
            KalmanResetStep(ofStream, substring, &img_out, &errork1_R, errork2_R, &Reset_R, Right);

            MeasureError(&img_out, &center_kalman_L, &center_measured_L, &errork2_L, Left);
            KalmanResetStep(ofStream, substring, &img_out, &errork1_L, errork2_L, &Reset_L, Left);

//            KalmanResetAndStep(&img_out, &center_kalman_R, &center_measured_R, &predRect_R, &errork1_R, &Reset_R, Right);
//            KalmanResetAndStep(&img_out, &center_kalman_L, &center_measured_L, &predRect_L, &errork1_L, &Reset_L, Left);


            ////////// Kalman Update //////////
            KalmanUpdate(&img_out, *(&kf_R), &notFoundCount, &state_R, &meas_R, &found, &Reset_R, Right);
            KalmanUpdate(&img_out, *(&kf_L), &notFoundCount, &state_L, &meas_L, &found, &Reset_L, Left);

        }

        /////// Visualize ///////

        if (count_cal < limit)
            if (img_test.data) cv::imshow("Algoritmo", img_test);

        if(start && (img_out.img.data) && img_show.data){
            imshow("Input", img_show);
            imshow("Algoritmo", img_out.img);
        }

        count_cal++;
        count_test++;
        ch = char(cv::waitKey(0));

    }

    return 0;

}