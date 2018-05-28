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

    cv::Mat img_cal, img_test, img_proc, labels, labels2;
    frame_out img_out = frame_out();

    // Images Reading
    string path_cal  = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/CALIBRACION01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
    string path_test = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/1_CAMARA/TEST01/*.jpg";
    // std::string path_cal = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";

    // Flag start detection
    bool start = false;

    int count_test = 195+145, count_cal = 0, limit = 150-145;
    vector<String> filenames_cal, filenames_test;

    glob(path_test, filenames_test);
    glob(path_cal , filenames_cal);

    //// Logging error Kalman (frames) ////
    int digits = 5;
    string fileName, substring;
    fileName = "/home/lalo/Dropbox/Proyecto IPD441/NeuroMisil_Lalo/NeuroMisil/Logging/error_KalmanRight.csv";
    ofstream ofStream(fileName);
    size_t pos = filenames_test[count_test].find(".jpg");
    ofStream << "Frame" << "," << "CX_Kalman" << "," << "CY_Kalman" << "," << "CX_Measured" << "," << "CY,Measured" << "," << "Pie" << "\n";

    char ch = 0;
    int  dT = 1;
    int notFoundCount = 0;
    bool ocult;
    bool found = false;

    cv::Rect predRect_R;

    cv::Point center_measured_R, center_kalman_R;
    double errork1 = 0, errork2 = 0, errorp = 0;

    ///// Algoritmo /////

    while(ch != 'q' && ch != 'Q'){

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

            img_out = FindBoxes(img_proc, ofStream, start);

            //////// Kalman Prediction ////////
            if(found){
                img_out = KalmanPredict(kf_R, state_R, img_out, dT);
                center_kalman_R = img_out.center;
                predRect_R      = img_out.predRect;
            }
            ///// Kalman Prediction Finish /////

            //// Logging ////
            if(start)
                ofStream << substring << "," << center_kalman_R.x << "," << center_kalman_R.y << ",";

            ////// Kalman Evaluate Reset ///////

            //  Si existe una diferencia muy grande entre el  centro x,y predicho por kalman
            //  y el centro x,y medido con la segmentacion se reseteara el filtro y se considerara que se obtuvo un paso

            ocult = bool(img_out.fboxes.size() == 1);
            if(ocult) {
                center_measured_R.x = img_out.fboxes[1].x + img_out.fboxes[1].width / 2;
                center_measured_R.y = img_out.fboxes[1].y + img_out.fboxes[1].height / 2;
            }else{
                center_measured_R.x = img_out.fboxes[2].x + img_out.fboxes[2].width / 2;
                center_measured_R.y = img_out.fboxes[2].y + img_out.fboxes[2].height / 2;
            }

            errork2 = distance(center_kalman_R, center_measured_R);

            if ( abs(errork1) > 1 ){
                found = false;
            }

            if(!ocult & found){
                errorp = (errork2 + errork1)/2;
                if(errorp < 3)
                    cv::rectangle(img_out.img, predRect_R, CV_RGB(0,0,255), 2);
            }

            ///// Logging /////

            if(start)
                ofStream << center_measured_R.x << "," << center_measured_R.y << "," << "Rigth" << "\n";

            cout << "Frame actual: " << filenames_test[count_test].substr(pos-digits) << endl;
            cout << "Posicion X predecida: " << state_R.at<float>(0) << endl;
            cout << "Posicion X medida: " << center_measured_R.x << endl;

            ////////// Kalman Update //////////

            // Cuando no encuentra caja
            if (img_out.fboxes[1].width <= 0){
                notFoundCount++;
                if( notFoundCount >= 100 ){
                    found = false;
                }else
                    kf_R.statePost = state_R;
            }else{
                // Si se encuemtra una caja, realiza medición
                // Si hay ocultamiento asigna a medida derecha centro de primer cuadro detectado
                if (img_out.fboxes.size() == 1) {
                    meas_R.at<float>(0) = img_out.fboxes[1].x + float(img_out.fboxes[1].width);
                    meas_R.at<float>(1) = img_out.fboxes[1].y + float(img_out.fboxes[1].height) / 2;
                    meas_R.at<float>(2) = (float) state_R.at<float>(4);
                    meas_R.at<float>(3) = (float) state_R.at<float>(5);
                // Si no hay ocultamiento adigna a medida derecha centro de segundo cuadro detectado
                } else {
                    meas_R.at<float>(0) = img_out.fboxes[2].x + float(img_out.fboxes[2].width) / 2;
                    meas_R.at<float>(1) = img_out.fboxes[2].y + float(img_out.fboxes[2].height) / 2;
                    meas_R.at<float>(2) = (float) img_out.fboxes[2].width;
                    meas_R.at<float>(3) = (float) img_out.fboxes[2].height;
                }

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
                    found = true;
                    kf_R.statePost = state_R;

                }else{
                    kf_R.correct(meas_R); // Kalman Correction
                }
                notFoundCount = 0;
            }

            ////////////////////////////////////
            /////// Kalman Update Finish ///////
            ////////////////////////////////////
        }

        /////// Visualize ///////

        if (count_cal < limit)
            if (img_test.data) cv::imshow("Algoritmo", img_test);

        if(start && (img_out.img.data)){
            imshow("Algoritmo", img_out.img);
            // imshow("Segmentación", img_out.seg);
        }

        count_cal++;
        count_test++;
        ch = char(cv::waitKey(0));
        errork1 = errork2;

    }

    return 0;

}