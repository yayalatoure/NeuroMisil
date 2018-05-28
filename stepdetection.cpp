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
    result = dx; // + dy; //sqrt(dx + dy);
    return result;
}

void KalmanInit(cv::KalmanFilter kf){

    // Kalman Filter Init

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
    kf.processNoiseCov.at<float>(14) = 1e-2;// 5.0f
    kf.processNoiseCov.at<float>(21) = 1e-2;// 5.0f
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));

}

frame_out FindBoxes(Mat img, ofstream &fileout, bool start){

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

    //// Start Segmentation ////
    mog->apply(img, fg, 2*learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,5)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,6))); //(3,6)
    cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);

    if(start){
        getFeet(fg, bboxes, labels, labels2, fboxes);
        paintRectangles(img, fboxes);
    }

    bboxes.clear();

    frameNumber++;

    output.flag = true;
    output.img  = img;
    output.seg  =  fg;
    output.fboxes = fboxes;

    return *(&output);
}
