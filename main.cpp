#include "stepdetection.h"
#include <QApplication>

int main(int argc, char *argv[]){

    cv::VideoCapture vid;
    vid.open(0); //Puede ser un string de ruta a camara (e.g. IP)

    if(!vid.isOpened()) {
        cerr << "Error opening input." << endl;
        return 1;
    }

    cv::Mat img, gray_image;
    frame_out img_out;

    long num_im = 1579;

    // getblobs -> cachando que wea hace
    std::map<int, Rect> bboxes;
    cv::Mat labels,labels2, labels_norm;
    double min, max;

    while(1){

//        vid >> img;

        std::string path = "/home/lalo/Desktop/Dropbox/Proyecto IPD441/NeuroMisil/NeuroMisil_CLion/TEST01/0";
        std::string name, im_name;
        im_name = QString::number(num_im).toStdString() + ".jpg";
        name = path + im_name;
        img = cv::imread(name , CV_LOAD_IMAGE_COLOR);

        if(img.data){

            img_out = stepDetection_2(img);

            cv::imshow("Video", img_out.area_filter);


//            for (int i = 0; i < img_out.labels.cols; ++i) {
//                printf("%d", img_out.labels.at<int>(0, i));
//            }
//            printf("\n");
//            printf("\n %d", img_out.flag);



//            cv::waitKey(0);
        }



        num_im++;

//        printf("\n %s ",im_name.c_str());

        if(cv::waitKey(10) != -1) break;

    }

    vid.release();

    cv::waitKey(0);
    return 0;

}
