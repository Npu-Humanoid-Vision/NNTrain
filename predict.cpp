#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define POS_LABLE 1
#define NEG_LABLE 0

// 摄像机模式/图片模式
#define CAMERA_MODE
// #define PIC_MODE

// 是否在达尔文上跑
// 可用 CV_MAJOR_VERSION代替
// #define RUN_ON_DARWIN

// 在摄像机模式获得样本
#define GET_SAMPLE_LIVING

#define MODEL_NAME "test.xml"

#define IMG_COLS 32
#define IMG_ROWS 32

// 开启选项之后
#ifdef GET_SAMPLE_LIVING

#define POS_COUNTER_INIT_NUM 384
#define NEG_COUNTER_INIT_NUM 160
#define SAVE_PATH "../../BackUpSource/Ball/Train/"

int pos_counter = POS_COUNTER_INIT_NUM;
int neg_counter = NEG_COUNTER_INIT_NUM;

string GetPath(string save_path, int lable) {
    stringstream t_ss;
    string t_s;

    if (lable == POS_LABLE) {
        save_path += "Pos/";
        t_ss << pos_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;
    }
    else {
        save_path += "Neg/";
        t_ss << neg_counter++;
        t_ss >> t_s;
        t_s = save_path + t_s;
        t_s += ".jpg";
        cout<<t_s<<endl;   
    }
    return t_s;
}
#endif


int main(int argc, char const *argv[]) {
    // load SVM model
#if CV_MAJOR_VERSION < 3
    CvANN_MLP tester;
    tester.load(MODEL_NAME);
#else
    cv::Ptr<cv::ml::SVM> tester = cv::ml::SVM::load(MODEL_NAME);
#endif
#ifdef CAMERA_MODE
    cv::VideoCapture cp(0);
    cv::Mat frame; 
    cv::Rect ROI_Rect(100, 100, 9*IMG_COLS, 9*IMG_ROWS);

    cp >> frame;
    while (frame.empty()) {
        cp >> frame;
    }
    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr << __LINE__ <<"frame empty"<<endl;
            return -1;
        }
#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        cv::Mat ROI = frame(ROI_Rect).clone();
        cv::resize(ROI, ROI, cv::Size(IMG_COLS, IMG_ROWS));
        cv::HOGDescriptor hog_des(Size(IMG_COLS, IMG_ROWS), Size(8,8), Size(4,4), Size(4,4), 9);
        std::vector<float> hog_vec;
        hog_des.compute(ROI, hog_vec);

        cv::Mat t(hog_vec);
        cv::Mat hog_vec_in_mat = t.t();
        hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);
#if CV_MAJOR_VERSION < 3
        cv::Mat outputs;
        tester.predict(hog_vec_in_mat, outputs);
        cout<<outputs<<endl;
        if (outputs.at<float>(0, 0) > 0.5) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#else
        cv::Mat lable;
        tester->predict(hog_vec_in_mat, lable);
        if (lable.at<float>(0, 0) == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        } 
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
#endif
        
        cv::imshow("frame", frame);
        char key = cv::waitKey(20);
        if (key == 'q') {
            break;
        }
        else if (key == 'p') {
            cv::imwrite(GetPath(SAVE_PATH, POS_LABLE), ROI);
        }
        else if (key == 'n') {
            cv::imwrite(GetPath(SAVE_PATH, NEG_LABLE), ROI);
        }

    }

#endif

#ifdef PIC_MODE
#endif
    return 0;
}
