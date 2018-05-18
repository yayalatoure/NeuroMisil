#include "stepdetection.h"

using namespace std;
using namespace cv;


void paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes){
    std::map<int, cv::Rect>::iterator it, it_end = bboxes.end();

    for(it = bboxes.begin(); it != it_end; it++) {
        cv::rectangle(img, it->second, cv::Scalar(0,255,0), 2);
    }

}

void getBlobs(cv::Mat labels, std::map<int, cv::Rect> &bboxes) {

    int ro = labels.rows, co = labels.cols;
    int label, x, y;

    bboxes.clear();
    for(int j=0; j<ro; ++j)
        for(int i=0; i<co; ++i) {
            label = labels.at<int>(j,i);
            if(label > 0) {                    // Not Background?
                if(bboxes.count(label) == 0) { // New label
                    cv::Rect r(i,j,1,1);
                    bboxes[label] = r;
                } else {                       // Update rect
                    cv::Rect &r = bboxes[label];
                    x = r.x + r.width  - 1;
                    y = r.y + r.height - 1;
                    if(i < r.x)  r.x = i;
                    if(i > x)    x = i;
                    if(j < r.y)  r.y = j;
                    if(j > y)    y = j;
                    r.width  = x - r.x + 1;
                    r.height = y - r.y + 1;
                }
            }
        }
}

void getFeet(cv::Mat fg, std::map<int, cv::Rect> &bboxes, cv::Mat labels, cv::Mat labels2, std::map<int, cv::Rect> &fboxes){

    // Selecciona la región conectada más grande
    int Direc = 0, biggestblob = 1;
    string Direccion;

    getBlobs(labels, bboxes);
//    malloc();

    for(unsigned int j=0; j < bboxes.size(); j++){
        if(bboxes[j].area() >= bboxes[biggestblob].area()) biggestblob = j;
    }

    // Crea una ROI en la parte inferior del jugador para visualizar sólo los
    // pies y eliminar el resto del análisis.

    Rect ROI;
    ROI.x = bboxes[biggestblob].x;
    ROI.y = int( bboxes[biggestblob].y + bboxes[biggestblob].height*0.8);
    ROI.height = int(bboxes[biggestblob].height*0.2);
    ROI.width = bboxes[biggestblob].width;

    Xk1 = ROI.x;
    Direc = (Xk1 - Xk0);
    Xk0 = Xk1;

    if(Direc == 0 ){
        Direccion = flag_direc;
    }else{
        Direccion = (Direc < 0) ? "Left" : "Rigth" ;
    }


    cout << "\n Dirección: " <<  Direccion << endl;
    flag_direc = Direccion;


    //
    Mat mask = Mat::zeros(fg.size(), CV_8U);
    rectangle(mask, ROI, Scalar(255), CV_FILLED);
    Mat fgROI = Mat::zeros(fg.size(), CV_8U);

    // copia fg a fgROI donde mask es distinto de cero.
    fg.copyTo(fgROI, mask);
    // aplica componentes conectados otra vez.
    cv::connectedComponents(fgROI, labels2, 8, CV_32S);
    getBlobs(labels2, fboxes);

}

double distance(cv::Point center_kalman, cv::Point center_measured){
    double dx = 0, dy = 0, result=0;
    dx = pow((center_kalman.x - center_measured.x), 2);
    dy = pow((center_kalman.y - center_measured.y), 2);
    result = sqrt(dx + dy);
    return result;
}

frame_out stepDetection_2(Mat img, ofstream &fileout, string substring, bool start){

    /* Inicializacion */
    Mat fg, labels, labels2, stats, centroids;


    double backgroundRatio = 0.7;
    double learningRate = 0.005;
    double varThreshold = 80;
    int    nmixtures = 3;
    int    history = 150;

    map<int, Rect> bboxes;
    map<int, Rect> fboxes;

    static int frameNumber = 0;

    static frame_out  output;

    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);
    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);

    /*Start Segmentation*/
    mog->apply(img, fg, 2*learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,5)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,6))); //(3,6)
    cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);



    /*Start Detection*/

    float xp, yp, height_p, widht_p;

    if(start){

        getFeet(fg, bboxes, labels, labels2, fboxes);
        paintRectangles(img, fboxes);

        /*
        xp = fboxes[1].x;
        yp = fboxes[1].y;
        height_p = fboxes[1].height;
        widht_p = fboxes[1].width;
        */

        /*
        static int v1r = 0, v2r = 0, ar = 0;
        static int v1l = 0, v2l = 0, al = 0;
        int vxl2 = 0, vxr2 = 0;
        int f1 = 0, f2 = 0;

        paintRectangles(img, fboxes);

        QRect r1 = QRect(fboxes[1].x, fboxes[1].y, fboxes[1].height, fboxes[1].width);
        QRect r2 = QRect(fboxes[2].x, fboxes[2].y, fboxes[2].height, fboxes[2].width);

        v2l  = (r1.bottom() - yl2);
        v2r  = (r2.bottom() - yr2);
        vxl2 = (r1.center().x() - xl2);
        vxr2 = (r2.center().x() - xr2);

        al = (v2l - v1l);
        ar = (v2r - v1r);

        if(al <= 0 || v2l == 0) f1 = 1;
        if(ar <= 0 || v2r == 0) f2 = 1;

        if((f1 && (v2l > 1)) || ((al > 1) && (v2l != 0)) || (fboxes[1].x == 0) || (fboxes[1].y == 0) ) f1 = 0;
        if((f2 && (v2r > 1)) || ((ar > 1) && (v2r != 0)) || (fboxes[2].x == 0) || (fboxes[2].y == 0) ) f2 = 0;

        if((vxl2 > 2) || (vxl2 < -2)) f1 = 0;
        if((vxr2 > 2) || (vxr2 < -2)) f2 = 0;

        if(f1){
            rectangle(img, fboxes[1], Scalar(0,255,0), 2);
            fileout << substring << "," << r1.center().x() << "," << r1.center().y() << "," << r1.bottom() << "," << "Id_1" << "\n";
        }
        if(f2){
            rectangle(img, fboxes[2], Scalar(0,255,0), 2);
            fileout << substring << "," << r2.center().x() << "," << r2.center().y() << "," << r2.bottom() << "," << "Id_2" << "\n";
        }

        yl2 = r1.bottom();
        yr2 = r2.bottom();
        xl2 = r1.center().x();
        xr2 = r2.center().x();

        v1l = v2l;
        v1r = v2r;
        */

    }

    bboxes.clear();

    /*End Detection*/

    frameNumber++;

    output.flag = true;
    output.img  = img;
    output.fboxes = fboxes;


    return *(&output);
}
