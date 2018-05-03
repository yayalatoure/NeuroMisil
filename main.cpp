#include "stepdetection.h"
#include <QApplication>


int main(int argc, char *argv[]){


    cv::Mat img, gray_image;
    frame_out img_out;
    std::map<int, Rect> bboxes;
    cv::Mat labels,labels2;

    // para lectura de imagenes
    std::string path;
    int count = 0, limit = 150;
    vector<String> filenames;

    while(true){

        if (count < limit){
            // path dell pc
            // path = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/NeuroMisil/NeuroMisil_CLion/CALIBRACION01/0";
            path = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/CALIBRACION01/*.jpg";
        }else{
            // path dell pc
            // path = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/NeuroMisil/NeuroMisil_CLion/TEST01/0";
            path = "/home/lalo/Dropbox/Proyecto IPD441/Data/Videos/TEST01/*.jpg";
            if (count == limit) count = 345;
        }

        glob(path, filenames);
        img = imread(filenames[count], CV_LOAD_IMAGE_COLOR);


        if(img.data){

            img_out = stepDetection_2(img);

            cv::imshow("Video", img_out.img);

//            cv::waitKey(0);
        }



        //num_im++;

//        printf("\n %s ",im_name.c_str());

        count++;
        if(cv::waitKey(10) != -1) break;

    }


    cv::waitKey(0);
    return 0;

}
