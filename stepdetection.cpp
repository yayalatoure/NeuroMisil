#include "stepdetection.h"

using namespace std;
using namespace cv;


void paintRectangles(cv::Mat &img, std::map<int, cv::Rect> &bboxes){
    std::map<int, cv::Rect>::iterator it, it_end = bboxes.end();

    for(it = bboxes.begin(); it != it_end; it++) {
        cv::rectangle(img, it->second, cv::Scalar(0,0,255), 2);
    }

}

// Recive
void getBlobs(cv::Mat labels, std::map<int, cv::Rect> &bboxes) {

    int r = labels.rows, c = labels.cols;
    int label, x, y;

    bboxes.clear();
    for(int j=0; j<r; ++j)
        for(int i=0; i<c; ++i) {
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

void getFeet(cv::Mat img, cv::Mat fg, std::map<int, cv::Rect> &bboxes, cv::Mat labels, cv::Mat labels2, std::map<int, cv::Rect> &fboxes){

    // Selecciona la región conectada más grande
    int biggestblob = 1;
    getBlobs(labels, bboxes);

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

Mat stepDetection_1(Mat img){

    Mat fg,labels,labels2;
    double backgroundRatio = 0.7;
    double learningRate = 0.0025;
    double varThreshold = 80;
    int    nmixtures = 3;
    int    history = 200;
    map<int, Rect>bboxes;
    map<int, Rect>fboxes;

    // Guardando Video por frame en directorio name
    static long frameNumber = 0; //01579
//    std::string path = "../NeuroMisilQt/FEED/";
//    std::string name;
//    name = path + QString::number(frameNumber).toStdString() + ".png";
//    cv::imwrite(name,img);

    cv::imshow("feed image", img);

    static int v1r = 0, v2r = 0, ar = 0;
    static int v1l = 0, v2l = 0, al = 0;
    int vxl2 = 0, vxr2 = 0;
    int f1 = 0, f2 = 0;
    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);

    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);

    /* Start Segmentation */

    mog->apply(img,fg,learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,6)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,6)));
    cv::connectedComponents(fg, labels, 8, CV_32S);

    /* End Segmentation */

    /* Start Detection */
    getFeet(img, fg, bboxes, labels, labels2, fboxes);

    paintRectangles(img,fboxes);
    QRect r1 = QRect(fboxes[1].x,fboxes[1].y,fboxes[1].height,fboxes[1].width);
    QRect r2 = QRect(fboxes[2].x,fboxes[2].y,fboxes[2].height,fboxes[2].width);

    v2l = (r1.bottom() - yl1);
    v2r = (r2.bottom() - yr1);
    vxl2 = (r1.center().x() - xl1);
    vxr2 = (r2.center().x() - xr1);

    al = (v2l - v1l);
    ar = (v2r - v1r);

    if(al <= 0 || v2l == 0) f1 = 1;
    if(ar <= 0 || v2r == 0) f2 = 1;

    if((f1 && (v2l > 1)) || ((al > 1) && (v2l != 0)))f1 = 0;
    if((f2 && (v2r > 1)) || ((ar > 1) && (v2r != 0)))f2 = 0;

    if((vxl2 > 2) || (vxl2 < -2)) f1 = 0;
    if((vxr2 > 2) || (vxr2 < -2)) f2 = 0;

    if(f1)rectangle(img,fboxes[1],Scalar(0,255,0),2);
    if(f2)rectangle(img,fboxes[2],Scalar(0,255,0),2);

    yl1 = r1.bottom();
    yr1 = r2.bottom();
    xl1 = r1.center().x();
    xr1 = r2.center().x();

    v1l = v2l;
    v1r = v2r;

    std::string name2 = "SEGMENTED1/";
    bboxes.clear();
    /*End Detection*/

//    name2+=QString::number(frameNumber).toStdString()+".jpg";
//    cv::imwrite(name2,img);

    frameNumber++;

    return img;
}

frame_out stepDetection_2(Mat img){

    /* Inicializacion */
    Mat fg,labels,labels2, stats, centroids;

    double backgroundRatio = 0.7;
    double learningRate = 0.005;
    double varThreshold = 80;
    int    nmixtures = 3;
    int    history = 150;

    map<int, Rect> bboxes;
    map<int, Rect> fboxes;

    static long frameNumber = 0;
    std::string name1 = "FEED0/";
    name1 += QString::number(frameNumber).toStdString()+".jpg";
//    cv::imwrite(name1,img);

    static frame_out *p_output;
    static frame_out  output;

    static int v1r = 0, v2r = 0, ar = 0;
    static int v1l = 0, v2l = 0, al = 0;
    int vxl2 = 0, vxr2 = 0;
    int f1 = 0, f2 = 0;


    static cv::Ptr<cv::BackgroundSubtractorMOG2> mog = cv::createBackgroundSubtractorMOG2(history, varThreshold, true);

    mog->setNMixtures(nmixtures);
    mog->setBackgroundRatio(backgroundRatio);
    mog->setShadowValue(0);
    /*Start Segmentation*/
    mog->apply(img, fg, 2*learningRate);

    cv::dilate(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,6)));
    cv::erode(fg, fg, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,6)));
    cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);

    // MEJORA N°1: Filtro por Área
    /*
    // Filtro por área de componentes conectados
    Mat sizes = stats.col(4);
    int min_size, nb_components = stats.rows;
    nb_components = nb_components - 1;
    min_size = 25;

    Mat area_filtered(labels.rows, labels.cols, CV_8U, Scalar(255));

    for (int i = 0; i < nb_components; ++i) {
        if (sizes.at<int>(i) >= min_size) {
            compare(labels, i+1, area_filtered, CMP_EQ); //
        }
    }

    for (int j = 0; j < sizes.rows; ++j) {
        printf("\t %d", sizes.at<int>(j));
        printf("\n");
    }
    printf("==================================== \n");
    output.labels = fg;
    output.area_filter = area_filtered;

    */

    /*End Segmentation*/


    /*Start Detection*/
    getFeet(img, fg, bboxes, labels, labels2, fboxes);

    /* Ahora aplica algoritmo */
    paintRectangles(img, fboxes);
    QRect r1 = QRect(fboxes[1].x, fboxes[1].y, fboxes[1].height, fboxes[1].width);
    QRect r2 = QRect(fboxes[2].x, fboxes[2].y, fboxes[2].height, fboxes[2].width);

    v2l = (r1.bottom() - yl2);
    v2r = (r2.bottom() - yr2);
    vxl2 = (r1.center().x() - xl2);
    vxr2 = (r2.center().x() - xr2);

    al = (v2l - v1l);
    ar = (v2r - v1r);

    if(al <= 0 || v2l == 0) f1 = 1;
    if(ar <= 0 || v2r == 0) f2 = 1;

    if((f1 && (v2l > 1)) || ((al > 1) && (v2l != 0)))f1 = 0;
    if((f2 && (v2r > 1)) || ((ar > 1) && (v2r != 0)))f2 = 0;

    if((vxl2 > 2) || (vxl2 < -2)) f1 = 0;
    if((vxr2 > 2) || (vxr2 < -2)) f2 = 0;

    if(f1)rectangle(img, fboxes[1], Scalar(0,255,0), 2);
    if(f2)rectangle(img, fboxes[2], Scalar(0,255,0), 2);

    yl2 = r1.bottom();
    yr2 = r2.bottom();
    xl2 = r1.center().x();
    xr2 = r2.center().x();

    v1l = v2l;
    v1r = v2r;

    std::string name2 = "SEGMENTED2/";
    bboxes.clear();
    /*End Detection*/


    name2+=QString::number(frameNumber).toStdString()+".jpg";
////    cv::imwrite(name2,img);


    frameNumber++;

    output.flag = true;
    output.img  = img;


    p_output = &output;

    return *p_output;
}
