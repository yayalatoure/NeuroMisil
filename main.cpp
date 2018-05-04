#include "stepdetection.h"
#include <QApplication>


int main(int argc, char *argv[]){

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

    while(true){

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

        if(img_proc.data){

            img_out = stepDetection_2(img_proc, ofStream, substring, start);

            if(count_cal < limit){
                if(img_test.data) cv::imshow("Video", img_test);
//                cv::imshow("Video", img_out.img);
            }else{
                cv::imshow("Video", img_out.img);
            }

//            cv::waitKey(0);

        }

        count_cal++;
        count_test++;

        if(cv::waitKey(20) != -1) break;

    }

//    cv::waitKey(0);
    return 0;

}
