#include "stepdetection.h"

using namespace std;
using namespace cv;


void KalmanInit(cv::KalmanFilter kf){

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
    kf.processNoiseCov.at<float>(14) = 1.0f;// 5.0f
    kf.processNoiseCov.at<float>(21) = 1.0f;// 5.0f
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));


}

void KalmanPredict(frame_out *img_out, cv::KalmanFilter kf, cv::Mat *state, cv::Rect *predRect, cv::Point *center_kalman, int dT){

    kf.transitionMatrix.at<float>(2) = dT;
    kf.transitionMatrix.at<float>(9) = dT;

    /////// Prediction ///////
    *state = kf.predict();

    ////// Predicted Rect Red //////
    (*predRect).width = static_cast<int>((*state).at<float>(4));
    (*predRect).height = static_cast<int>((*state).at<float>(5));
    (*predRect).x = static_cast<int>((*state).at<float>(0) - (*state).at<float>(4)/2);
    (*predRect).y = static_cast<int>((*state).at<float>(1) - (*state).at<float>(5)/2);
    cv::rectangle((*img_out).img, *predRect, CV_RGB(255,0,0), 2);
    //// Predicted Point ////
    (*center_kalman).x = static_cast<int>((*state).at<float>(0));
    (*center_kalman).y = static_cast<int>((*state).at<float>(1)); ////static_cast<int>((*state).at<float>(1) + (*state).at<float>(5)/2);
    cv::circle((*img_out).img, *center_kalman, 2, CV_RGB(255,0,0), -1);


}

void MeasureError(frame_out *img_out, cv::Point *center_kalman, cv::Point *center_measured, double *errork2, int pie){

    double error;
    bool ocultamiento;

    ocultamiento = bool((*img_out).fboxes.size() == 1);

    if(ocultamiento){
        if(pie == Right) {
            (*center_measured).x = (*img_out).fboxes[Right].x + (*img_out).fboxes[Right].width / 4;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }else{
            (*center_measured).x = (*img_out).fboxes[Right].x + ((*img_out).fboxes[Right].width*3) / 4;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }
    }else{
        if(pie == Right) {
            (*center_measured).x = (*img_out).fboxes[Right].x + (*img_out).fboxes[Right].width / 2;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }else{
            (*center_measured).x = (*img_out).fboxes[Left].x + (*img_out).fboxes[Left].width / 2;
            (*center_measured).y = (*img_out).fboxes[Left].y + (*img_out).fboxes[Left].height / 2;
        }
    }

    error = distance(&(*center_kalman), &(*center_measured));
    *errork2 = error;

}

double distance(cv::Point *center_kalman, cv::Point *center_measured){
    double dx = 0, dy = 0, result=0;
    dx = pow(((*center_kalman).x - (*center_measured).x), 2);
    dy = pow(((*center_kalman).y - (*center_measured).y), 2);
    result = dx + dy;
    return result;
}

void KalmanResetStep(ofstream &fileout, string substring, frame_out *img_out, double *errork1, double errork2, bool *reset, int pie){

    double error;
    bool ocultamiento;

    ocultamiento = bool((*img_out).fboxes.size() == 1);


    if ( abs(errork2) > 2.5 ){
        *reset = true;
    }

    if(!ocultamiento & !(*reset)){
        error = errork2; //(errork2 + (*errork1))/2;
        if(abs(error) < 2.5) {
            cv::rectangle((*img_out).img, (*img_out).fboxes[pie], CV_RGB(0, 0, 255), 2);

            //// Logging

            QRect r = QRect((*img_out).fboxes[pie].x, (*img_out).fboxes[pie].y, (*img_out).fboxes[pie].height, (*img_out).fboxes[pie].width);

            if(pie == Right){
                fileout << substring << "," << r.center().x() << "," << r.center().y() << "," <<
                        (*img_out).fboxes[pie].height <<","<<  (*img_out).fboxes[pie].width <<","<<"Rigth"<<"\n";
            }else{
                fileout << substring << "," << r.center().x() << "," << r.center().y() << "," <<
                        (*img_out).fboxes[pie].height <<","<<  (*img_out).fboxes[pie].width <<","<<"Left"<<"\n";
            }

        }
    }

    *errork1 = errork2;

}

void KalmanResetAndStep(frame_out *img_out, cv::Point *center_kalman, cv::Point *center_measured, cv::Rect *predRect, double *errork1, bool *reset, int pie){

    double errork2, errorp;
    bool ocultamiento;

    ocultamiento = bool((*img_out).fboxes.size() == 1);

    if(ocultamiento){
        if(pie == Right) {
            (*center_measured).x = (*img_out).fboxes[Right].x + (*img_out).fboxes[Right].width / 4;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }else{
            (*center_measured).x = (*img_out).fboxes[Right].x + ((*img_out).fboxes[Right].width*3) / 4;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }
    }else{
        if(pie == Right) {
            (*center_measured).x = (*img_out).fboxes[Right].x + (*img_out).fboxes[Right].width / 2;
            (*center_measured).y = (*img_out).fboxes[Right].y + (*img_out).fboxes[Right].height / 2;
        }else{
            (*center_measured).x = (*img_out).fboxes[Left].x + (*img_out).fboxes[Left].width / 2;
            (*center_measured).y = (*img_out).fboxes[Left].y + (*img_out).fboxes[Left].height / 2;
        }
    }

    errork2 = distance(&(*center_kalman), &(*center_measured));

    if ( abs(*errork1) > 1.5 ){
        *reset = true;
    }

    if(!ocultamiento & !(*reset)){
        errorp = (errork2 ); // (errork2 + (*errork1))/2;
        if(errorp < 2)
           cv::rectangle((*img_out).img, *predRect, CV_RGB(0,0,255), 2);
    }

    *errork1 = errork2;

}


void KalmanUpdate(frame_out *img_out, cv::KalmanFilter kf, int *notFoundCount, cv::Mat *state, cv::Mat *measure, bool *found, bool *reset, int pie){

    bool ocultamiento;
    ocultamiento = bool((*img_out).fboxes.size() == 1);

    // Cuando no encuentra caja
    if ((*img_out).fboxes[1].width <= 0){
        (*notFoundCount)++;
        if( (*notFoundCount) >= 100 ){
            *found = false;
        }else{
            kf.statePost.at<float>(0) = (*state).at<float>(0);
            kf.statePost.at<float>(1) = (*state).at<float>(1); ////(*state).at<float>(1) + (*state).at<float>(5)/2;
            kf.statePost.at<float>(2) = (*state).at<float>(2);
            kf.statePost.at<float>(3) = (*state).at<float>(3);
            kf.statePost.at<float>(4) = (*state).at<float>(4);
            kf.statePost.at<float>(5) = (*state).at<float>(5);
        }
    }else{
        // Si se encuentra una caja, realiza medición

        if (ocultamiento) {
//            (*measure).at<float>(0) = (*img_out).fboxes[Right].x + float((*img_out).fboxes[Right].width);
//            (*measure).at<float>(1) = (*img_out).fboxes[Right].y + float((*img_out).fboxes[Right].height) / 2;
//            (*measure).at<float>(2) = (float) (*state).at<float>(4);
//            (*measure).at<float>(3) = (float) (*state).at<float>(5);
            if(pie == Right) {
                (*measure).at<float>(0) = (*img_out).fboxes[Right].x + float((*img_out).fboxes[Right].width) / 4;
                (*measure).at<float>(1) = (*img_out).fboxes[Right].y + float((*img_out).fboxes[Right].height) / 4;
                (*measure).at<float>(2) = (float) (*img_out).fboxes[Right].width / 3;
                (*measure).at<float>(3) = (float) ((*img_out).fboxes[Right].height*3) /4 ;
            }else if(pie == Left){
                (*measure).at<float>(0) = (*img_out).fboxes[Right].x + (float((*img_out).fboxes[Right].width)*3) / 4;
                (*measure).at<float>(1) = (*img_out).fboxes[Right].y + (float((*img_out).fboxes[Right].height)*3) / 4;
                (*measure).at<float>(2) = (float) (*img_out).fboxes[Right].width / 3;
                (*measure).at<float>(3) = (float) ((*img_out).fboxes[Right].height*3) /4 ;
            }
        } else {
            if(pie == Right) {
                (*measure).at<float>(0) = (*img_out).fboxes[Right].x + float((*img_out).fboxes[Right].width) / 2;
                (*measure).at<float>(1) = (*img_out).fboxes[Right].y + float((*img_out).fboxes[Right].height) / 2;
                (*measure).at<float>(2) = (float) (*img_out).fboxes[Right].width;
                (*measure).at<float>(3) = (float) (*img_out).fboxes[Right].height;
            }else if(pie == Left){
                (*measure).at<float>(0) = (*img_out).fboxes[Left].x + float((*img_out).fboxes[Left].width) / 2;
                (*measure).at<float>(1) = (*img_out).fboxes[Left].y + float((*img_out).fboxes[Left].height) / 2;
                (*measure).at<float>(2) = (float) (*img_out).fboxes[Left].width;
                (*measure).at<float>(3) = (float) (*img_out).fboxes[Left].height;
            }
        }
        // cambie flag reset por found
        if (*reset){ // First detection!
            // >>>> Initialization
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px

            (*state).at<float>(0) = (*measure).at<float>(0);
            (*state).at<float>(1) = (*measure).at<float>(1);
            (*state).at<float>(2) = 0;
            (*state).at<float>(3) = 0;
            (*state).at<float>(4) = (*measure).at<float>(2);
            (*state).at<float>(5) = (*measure).at<float>(3);
            // <<<< Initialization

            kf.statePost.at<float>(0) = (*state).at<float>(0);
            kf.statePost.at<float>(1) = (*state).at<float>(1);
            kf.statePost.at<float>(2) = (*state).at<float>(2);
            kf.statePost.at<float>(3) = (*state).at<float>(3);
            kf.statePost.at<float>(4) = (*state).at<float>(4);
            kf.statePost.at<float>(5) = (*state).at<float>(5);

            *reset = false;

        }else{
            kf.correct(*measure); // Kalman Correction
        }
        *notFoundCount = 0;
    }

}

void FindBoxes(frame_out *img_out, Mat img, bool start, bool *found){

    /* Inicializacion */
    Mat fg, labels, labels2, stats, centroids;

    double backgroundRatio = 0.7;
    double learningRate = 0.005; ////0.005
    double varThreshold = 80;
    int    nmixtures = 3;
    int    history = 150;

    map<int, Rect> bboxes;
    map<int, Rect> fboxes;

    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);
    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);

    //// Start Segmentation ////
    mog->apply(img, fg, 2*learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,6)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,5))); ////(4,6)
    cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);

    if(start){
        getFeet(fg, bboxes, labels, labels2, fboxes);
        paintRectangles(img, fboxes);
    }

    bboxes.clear();

    if (fboxes[1].width > 0)
        *found = true;
    else
        *found = false;

    (*img_out).img  = img;
    (*img_out).seg  =  fg;
    (*img_out).fboxes = fboxes;

}

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

    getBlobs(labels, bboxes); // NOLINT

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

    flag_direc = Direccion;

    Mat mask = Mat::zeros(fg.size(), CV_8U);
    rectangle(mask, ROI, Scalar(255), CV_FILLED);
    Mat fgROI = Mat::zeros(fg.size(), CV_8U);

    // copia fg a fgROI donde mask es distinto de cero.
    fg.copyTo(fgROI, mask);
    // aplica componentes conectados otra vez.
    cv::connectedComponents(fgROI, labels2, 8, CV_32S);
    getBlobs(labels2, fboxes);

}
